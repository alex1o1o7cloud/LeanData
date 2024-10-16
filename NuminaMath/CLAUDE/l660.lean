import Mathlib

namespace NUMINAMATH_CALUDE_parallel_equal_segment_construction_l660_66024

/-- Represents a point on a 2D grid -/
structure GridPoint where
  x : ℤ
  y : ℤ

/-- Represents a vector on a 2D grid -/
structure GridVector where
  dx : ℤ
  dy : ℤ

/-- Calculates the vector between two grid points -/
def vectorBetween (a b : GridPoint) : GridVector :=
  { dx := b.x - a.x, dy := b.y - a.y }

/-- Translates a point by a vector -/
def translatePoint (p : GridPoint) (v : GridVector) : GridPoint :=
  { x := p.x + v.dx, y := p.y + v.dy }

/-- Calculates the squared length of a vector -/
def vectorLengthSquared (v : GridVector) : ℤ :=
  v.dx * v.dx + v.dy * v.dy

theorem parallel_equal_segment_construction 
  (a b c : GridPoint) : 
  let v := vectorBetween a b
  let d := translatePoint c v
  (vectorBetween c d = v) ∧ 
  (vectorLengthSquared (vectorBetween a b) = vectorLengthSquared (vectorBetween c d)) :=
by sorry


end NUMINAMATH_CALUDE_parallel_equal_segment_construction_l660_66024


namespace NUMINAMATH_CALUDE_product_nine_sum_undetermined_l660_66091

theorem product_nine_sum_undetermined : 
  ∃ (a b c d : ℤ), 
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
    a * b * c * d = 9 ∧
    ¬∃! (s : ℤ), s = a + b + c + d :=
by sorry

end NUMINAMATH_CALUDE_product_nine_sum_undetermined_l660_66091


namespace NUMINAMATH_CALUDE_green_pieces_count_l660_66060

theorem green_pieces_count (amber : ℕ) (clear : ℕ) (green : ℕ) :
  amber = 20 →
  clear = 85 →
  green = (25 : ℚ) / 100 * (amber + green + clear) →
  green = 35 := by
sorry

end NUMINAMATH_CALUDE_green_pieces_count_l660_66060


namespace NUMINAMATH_CALUDE_f_increasing_max_b_value_ln_2_bounds_l660_66034

noncomputable def f (x : ℝ) := Real.exp x - Real.exp (-x) - 2 * x

theorem f_increasing : ∀ x y : ℝ, x < y → f x < f y := by sorry

noncomputable def g (b : ℝ) (x : ℝ) := f (2 * x) - 4 * b * f x

theorem max_b_value : 
  (∀ x : ℝ, x > 0 → g 2 x > 0) ∧ 
  (∀ b : ℝ, b > 2 → ∃ x : ℝ, x > 0 ∧ g b x ≤ 0) := by sorry

theorem ln_2_bounds : 0.692 < Real.log 2 ∧ Real.log 2 < 0.694 := by sorry

end NUMINAMATH_CALUDE_f_increasing_max_b_value_ln_2_bounds_l660_66034


namespace NUMINAMATH_CALUDE_prism_18_edges_has_8_faces_l660_66089

/-- A prism is a polyhedron with a specific structure. -/
structure Prism where
  edges : ℕ
  lateral_faces : ℕ
  total_faces : ℕ

/-- Theorem: A prism with 18 edges has 8 faces. -/
theorem prism_18_edges_has_8_faces :
  ∀ (p : Prism), p.edges = 18 → p.total_faces = 8 := by
  sorry


end NUMINAMATH_CALUDE_prism_18_edges_has_8_faces_l660_66089


namespace NUMINAMATH_CALUDE_beatrice_book_cost_l660_66074

/-- Calculates the total cost of books given the pricing rules and number of books purchased. -/
def book_cost (regular_price : ℕ) (discount : ℕ) (regular_quantity : ℕ) (total_quantity : ℕ) : ℕ :=
  let regular_cost := regular_price * regular_quantity
  let discounted_quantity := total_quantity - regular_quantity
  let discounted_price := regular_price - discount
  let discounted_cost := discounted_quantity * discounted_price
  regular_cost + discounted_cost

/-- Proves that given the specific pricing rules and Beatrice's purchase, the total cost is $370. -/
theorem beatrice_book_cost :
  let regular_price := 20
  let discount := 2
  let regular_quantity := 5
  let total_quantity := 20
  book_cost regular_price discount regular_quantity total_quantity = 370 := by
  sorry

#eval book_cost 20 2 5 20  -- This should output 370

end NUMINAMATH_CALUDE_beatrice_book_cost_l660_66074


namespace NUMINAMATH_CALUDE_mashed_potatoes_tomatoes_difference_l660_66008

/-- The number of students who suggested mashed potatoes -/
def mashed_potatoes : ℕ := 144

/-- The number of students who suggested bacon -/
def bacon : ℕ := 467

/-- The number of students who suggested tomatoes -/
def tomatoes : ℕ := 79

/-- The difference between the number of students who suggested mashed potatoes and tomatoes -/
def difference : ℕ := mashed_potatoes - tomatoes

theorem mashed_potatoes_tomatoes_difference : difference = 65 := by sorry

end NUMINAMATH_CALUDE_mashed_potatoes_tomatoes_difference_l660_66008


namespace NUMINAMATH_CALUDE_proposition_one_is_correct_l660_66098

theorem proposition_one_is_correct (p q : Prop) :
  (¬(p ∧ q) ∧ ¬(p ∨ q)) → (¬p ∧ ¬q) := by sorry

end NUMINAMATH_CALUDE_proposition_one_is_correct_l660_66098


namespace NUMINAMATH_CALUDE_imo_inequality_l660_66073

theorem imo_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a / Real.sqrt (a^2 + 8*b*c)) + (b / Real.sqrt (b^2 + 8*c*a)) + (c / Real.sqrt (c^2 + 8*a*b)) ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_imo_inequality_l660_66073


namespace NUMINAMATH_CALUDE_quadratic_convergence_l660_66009

-- Define the quadratic function
def f (p q : ℝ) (x : ℝ) : ℝ := x^2 + p*x + q

-- Define the property that |f(x)| ≤ 1/2 for all x in [3, 5]
def bounded_on_interval (p q : ℝ) : Prop :=
  ∀ x : ℝ, 3 ≤ x ∧ x ≤ 5 → |f p q x| ≤ 1/2

-- Define the repeated application of f
def f_iterate (p q : ℝ) (n : ℕ) (x : ℝ) : ℝ :=
  match n with
  | 0 => x
  | n+1 => f p q (f_iterate p q n x)

-- State the theorem
theorem quadratic_convergence (p q : ℝ) (h : bounded_on_interval p q) :
  f_iterate p q 2017 ((7 + Real.sqrt 15) / 2) = (7 - Real.sqrt 15) / 2 :=
sorry

end NUMINAMATH_CALUDE_quadratic_convergence_l660_66009


namespace NUMINAMATH_CALUDE_brenda_mice_problem_l660_66018

theorem brenda_mice_problem (total_mice : ℕ) : 
  (∃ (given_to_robbie sold_to_store sold_as_feeder remaining : ℕ),
    given_to_robbie = total_mice / 6 ∧
    sold_to_store = 3 * given_to_robbie ∧
    sold_as_feeder = (total_mice - given_to_robbie - sold_to_store) / 2 ∧
    remaining = total_mice - given_to_robbie - sold_to_store - sold_as_feeder ∧
    remaining = 4 ∧
    total_mice % 3 = 0) →
  total_mice / 3 = 8 := by
sorry

end NUMINAMATH_CALUDE_brenda_mice_problem_l660_66018


namespace NUMINAMATH_CALUDE_pies_from_apples_l660_66090

/-- Given the rate of pies per apples and a new number of apples, calculate the number of pies that can be made -/
def calculate_pies (initial_apples : ℕ) (initial_pies : ℕ) (new_apples : ℕ) : ℕ :=
  (new_apples * initial_pies) / initial_apples

/-- Theorem stating that given 3 pies can be made from 15 apples, 45 apples will yield 9 pies -/
theorem pies_from_apples :
  calculate_pies 15 3 45 = 9 := by
  sorry

end NUMINAMATH_CALUDE_pies_from_apples_l660_66090


namespace NUMINAMATH_CALUDE_annual_bill_calculation_correct_l660_66083

/-- Calculates the total annual bill for Noah's calls to his Grammy -/
def annual_bill_calculation : ℝ :=
  let weekday_duration : ℝ := 25
  let weekend_duration : ℝ := 45
  let holiday_duration : ℝ := 60
  
  let total_weekdays : ℝ := 260
  let total_weekends : ℝ := 104
  let total_holidays : ℝ := 11
  
  let intl_weekdays : ℝ := 130
  let intl_weekends : ℝ := 52
  let intl_holidays : ℝ := 6
  
  let local_weekday_rate : ℝ := 0.05
  let local_weekend_rate : ℝ := 0.06
  let local_holiday_rate : ℝ := 0.07
  
  let intl_weekday_rate : ℝ := 0.09
  let intl_weekend_rate : ℝ := 0.11
  let intl_holiday_rate : ℝ := 0.12
  
  let tax_rate : ℝ := 0.10
  let monthly_service_fee : ℝ := 2.99
  let intl_holiday_discount : ℝ := 0.05
  
  let local_weekday_cost := (total_weekdays - intl_weekdays) * weekday_duration * local_weekday_rate
  let local_weekend_cost := (total_weekends - intl_weekends) * weekend_duration * local_weekend_rate
  let local_holiday_cost := (total_holidays - intl_holidays) * holiday_duration * local_holiday_rate
  
  let intl_weekday_cost := intl_weekdays * weekday_duration * intl_weekday_rate
  let intl_weekend_cost := intl_weekends * weekend_duration * intl_weekend_rate
  let intl_holiday_cost := intl_holidays * holiday_duration * intl_holiday_rate * (1 - intl_holiday_discount)
  
  let total_call_cost := local_weekday_cost + local_weekend_cost + local_holiday_cost + 
                         intl_weekday_cost + intl_weekend_cost + intl_holiday_cost
  
  let total_tax := total_call_cost * tax_rate
  let total_service_fee := monthly_service_fee * 12
  
  total_call_cost + total_tax + total_service_fee

theorem annual_bill_calculation_correct : 
  annual_bill_calculation = 1042.20 := by sorry

end NUMINAMATH_CALUDE_annual_bill_calculation_correct_l660_66083


namespace NUMINAMATH_CALUDE_fraction_relation_l660_66078

theorem fraction_relation (w x y z : ℝ) 
  (h1 : x / y = 5)
  (h2 : y / z = 1 / 2)
  (h3 : z / w = 7)
  (hw : w ≠ 0) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) : 
  w / x = 2 / 35 := by
  sorry

end NUMINAMATH_CALUDE_fraction_relation_l660_66078


namespace NUMINAMATH_CALUDE_sum_of_y_coordinates_l660_66041

theorem sum_of_y_coordinates : ∀ (y₁ y₂ : ℝ),
  (∀ (y : ℝ), (4 - (-1))^2 + (y - (-3))^2 = 15^2 → y = y₁ ∨ y = y₂) →
  y₁ + y₂ = -6 := by
sorry

end NUMINAMATH_CALUDE_sum_of_y_coordinates_l660_66041


namespace NUMINAMATH_CALUDE_inscribed_square_side_length_l660_66094

/-- A right triangle with sides 5, 12, and 13 -/
structure RightTriangle where
  de : ℝ
  ef : ℝ
  df : ℝ
  de_eq : de = 5
  ef_eq : ef = 12
  df_eq : df = 13
  right_angle : de^2 + ef^2 = df^2

/-- A square inscribed in the right triangle -/
structure InscribedSquare (t : RightTriangle) where
  side_length : ℝ
  on_df : side_length ≤ t.df
  on_de : side_length ≤ t.de
  on_ef : side_length ≤ t.ef

/-- The side length of the inscribed square is 30/7 -/
theorem inscribed_square_side_length (t : RightTriangle) (s : InscribedSquare t) :
  s.side_length = 30 / 7 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_square_side_length_l660_66094


namespace NUMINAMATH_CALUDE_quadratic_function_properties_l660_66081

/-- A quadratic function passing through (-2, 1) with exactly one root -/
def f (a b : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + 1

/-- The function g derived from f -/
def g (a b k : ℝ) (x : ℝ) : ℝ := f a b x - k * x

/-- The theorem stating the properties of f and g -/
theorem quadratic_function_properties (a b : ℝ) (h_a : a ≠ 0) :
  (∃ x₁ x₂ : ℝ, 1 ≤ x₁ ∧ x₁ < x₂ ∧ f a b (-2) = 1) →
  (∃! x : ℝ, f a b x = 0) →
  (∀ x : ℝ, f a b x = (x + 1)^2) ∧
  (∀ k : ℝ, (∀ x : ℝ, x ∈ Set.Icc (-1) 2 → Monotone (g a b k)) ↔ k ≤ 0 ∨ k ≥ 6) :=
sorry

end NUMINAMATH_CALUDE_quadratic_function_properties_l660_66081


namespace NUMINAMATH_CALUDE_reflection_yoz_plane_l660_66065

-- Define a point in 3D space
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

-- Define the original point P
def P : Point3D := ⟨3, 1, 5⟩

-- Define the function for reflection across the yOz plane
def reflectYOZ (p : Point3D) : Point3D :=
  ⟨-p.x, p.y, p.z⟩

-- Theorem statement
theorem reflection_yoz_plane :
  reflectYOZ P = ⟨-3, 1, 5⟩ := by
  sorry

end NUMINAMATH_CALUDE_reflection_yoz_plane_l660_66065


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_and_perimeter_l660_66077

theorem right_triangle_hypotenuse_and_perimeter :
  ∀ (a b h p : ℝ),
  a = 24 →
  b = 32 →
  h^2 = a^2 + b^2 →
  p = a + b + h →
  h = 40 ∧ p = 96 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_and_perimeter_l660_66077


namespace NUMINAMATH_CALUDE_train_length_calculation_l660_66020

/-- The length of a train given its speed and time to cross a pole -/
theorem train_length_calculation (speed_kmh : ℝ) (cross_time_s : ℝ) :
  speed_kmh = 56 →
  cross_time_s = 9 →
  ∃ (length_m : ℝ), 139 < length_m ∧ length_m < 141 :=
by
  sorry

#check train_length_calculation

end NUMINAMATH_CALUDE_train_length_calculation_l660_66020


namespace NUMINAMATH_CALUDE_subscription_cost_l660_66058

theorem subscription_cost (reduction_percentage : ℝ) (reduction_amount : ℝ) (original_cost : ℝ) : 
  reduction_percentage = 0.30 →
  reduction_amount = 658 →
  reduction_percentage * original_cost = reduction_amount →
  original_cost = 2193 := by
  sorry

end NUMINAMATH_CALUDE_subscription_cost_l660_66058


namespace NUMINAMATH_CALUDE_log_equality_l660_66063

-- Define the logarithm base 2 (lg)
noncomputable def lg (x : ℝ) := Real.log x / Real.log 2

-- State the theorem
theorem log_equality : lg (5/2) + 2 * lg 2 + 2^(Real.log 3 / Real.log 4) = 1 + Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_log_equality_l660_66063


namespace NUMINAMATH_CALUDE_problem_solution_l660_66032

def set_A (a : ℝ) : Set ℝ := {a - 3, 2 * a - 1, a^2 + 1}
def set_B (x : ℝ) : Set ℝ := {0, 1, x}

theorem problem_solution :
  (∀ a : ℝ, -3 ∈ set_A a → a = 0 ∨ a = -1) ∧
  (∀ x : ℝ, x^2 ∈ set_B x ∧ x ≠ 0 ∧ x ≠ 1 → x = -1) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l660_66032


namespace NUMINAMATH_CALUDE_juniper_bones_ratio_l660_66004

theorem juniper_bones_ratio : 
  ∀ (initial_bones given_bones stolen_bones final_bones : ℕ),
    initial_bones = 4 →
    stolen_bones = 2 →
    final_bones = 6 →
    final_bones = initial_bones + given_bones - stolen_bones →
    (initial_bones + given_bones) / initial_bones = 2 := by
  sorry

end NUMINAMATH_CALUDE_juniper_bones_ratio_l660_66004


namespace NUMINAMATH_CALUDE_james_golden_retrievers_l660_66097

/-- Represents the number of dogs James has for each breed -/
structure DogCounts where
  huskies : Nat
  pitbulls : Nat
  golden_retrievers : Nat

/-- Represents the number of pups each breed has -/
structure PupCounts where
  husky_pups : Nat
  pitbull_pups : Nat
  golden_retriever_pups : Nat

/-- The problem statement -/
theorem james_golden_retrievers (dogs : DogCounts) (pups : PupCounts) : 
  dogs.huskies = 5 →
  dogs.pitbulls = 2 →
  pups.husky_pups = 3 →
  pups.pitbull_pups = 3 →
  pups.golden_retriever_pups = pups.husky_pups + 2 →
  dogs.huskies * pups.husky_pups + 
  dogs.pitbulls * pups.pitbull_pups + 
  dogs.golden_retrievers * pups.golden_retriever_pups = 
  (dogs.huskies + dogs.pitbulls + dogs.golden_retrievers) + 30 →
  dogs.golden_retrievers = 4 := by
sorry

end NUMINAMATH_CALUDE_james_golden_retrievers_l660_66097


namespace NUMINAMATH_CALUDE_smallest_representable_difference_l660_66064

theorem smallest_representable_difference : ∃ (m n : ℕ+), 
  14 = 19^(n : ℕ) - 5^(m : ℕ) ∧ 
  ∀ (k : ℕ+) (m' n' : ℕ+), k < 14 → k ≠ 19^(n' : ℕ) - 5^(m' : ℕ) :=
by sorry

end NUMINAMATH_CALUDE_smallest_representable_difference_l660_66064


namespace NUMINAMATH_CALUDE_five_rooks_on_five_by_five_board_l660_66050

/-- The number of ways to place n distinct rooks on an n×n chess board
    such that no two rooks share the same row or column -/
def rook_placements (n : ℕ) : ℕ := Nat.factorial n

/-- Theorem: The number of ways to place 5 distinct rooks on a 5×5 chess board,
    such that no two rooks share the same row or column, is equal to 5! (120) -/
theorem five_rooks_on_five_by_five_board :
  rook_placements 5 = 120 := by
  sorry

#eval rook_placements 5  -- Should output 120

end NUMINAMATH_CALUDE_five_rooks_on_five_by_five_board_l660_66050


namespace NUMINAMATH_CALUDE_arithmetic_sequence_2005_l660_66021

/-- 
Given an arithmetic sequence with first term a₁ = -1 and common difference d = 2,
prove that the 1004th term is equal to 2005.
-/
theorem arithmetic_sequence_2005 :
  let a : ℕ → ℤ := λ n => -1 + (n - 1) * 2
  a 1004 = 2005 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_2005_l660_66021


namespace NUMINAMATH_CALUDE_bobs_grade_l660_66023

theorem bobs_grade (jenny_grade jason_grade bob_grade : ℕ) : 
  jenny_grade = 95 →
  jason_grade = jenny_grade - 25 →
  bob_grade = jason_grade / 2 →
  bob_grade = 35 := by
sorry

end NUMINAMATH_CALUDE_bobs_grade_l660_66023


namespace NUMINAMATH_CALUDE_divisibility_property_l660_66057

theorem divisibility_property (a b c : ℕ) : 
  a ∈ ({1, 2, 3, 4, 5, 6} : Set ℕ) → 
  b ∈ ({1, 2, 3, 4, 5, 6} : Set ℕ) → 
  c ∈ ({1, 2, 3, 4, 5, 6} : Set ℕ) → 
  a ≠ b → b ≠ c → a ≠ c →
  ∃ k : ℤ, a * b * c + (7 - a) * (7 - b) * (7 - c) = 7 * k := by
sorry

end NUMINAMATH_CALUDE_divisibility_property_l660_66057


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_l660_66096

theorem right_triangle_hypotenuse (a q : ℝ) (ha : a > 0) (hq : q > 0) :
  ∃ (b c : ℝ),
    c > 0 ∧
    a^2 + b^2 = c^2 ∧
    q * c = b^2 ∧
    c = q / 2 + Real.sqrt ((q / 2)^2 + a^2) :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_l660_66096


namespace NUMINAMATH_CALUDE_geometric_series_sum_l660_66069

/-- The sum of a geometric series with first term a, common ratio r, and n terms -/
def geometricSum (a : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

/-- The first term of our geometric series -/
def a : ℚ := 1/2

/-- The common ratio of our geometric series -/
def r : ℚ := 1/2

/-- The number of terms in our geometric series -/
def n : ℕ := 8

theorem geometric_series_sum :
  geometricSum a r n = 255/256 := by
  sorry

end NUMINAMATH_CALUDE_geometric_series_sum_l660_66069


namespace NUMINAMATH_CALUDE_smallest_tangent_circle_radius_l660_66016

theorem smallest_tangent_circle_radius 
  (square_side : ℝ) 
  (semicircle_radius : ℝ) 
  (quarter_circle_radius : ℝ) 
  (h1 : square_side = 4) 
  (h2 : semicircle_radius = 1) 
  (h3 : quarter_circle_radius = 2) : 
  ∃ r : ℝ, r = Real.sqrt 2 - 3/2 ∧ 
  (∀ x y : ℝ, x^2 + y^2 = r^2 → 
    ((x - 2)^2 + y^2 = 1^2 ∨ 
     (x + 2)^2 + y^2 = 1^2 ∨ 
     x^2 + (y - 2)^2 = 1^2 ∨ 
     x^2 + (y + 2)^2 = 1^2) ∧
    x^2 + y^2 = (2 - r)^2) :=
by sorry

end NUMINAMATH_CALUDE_smallest_tangent_circle_radius_l660_66016


namespace NUMINAMATH_CALUDE_complement_of_A_l660_66028

def U : Set Int := {-1, 0, 1, 2}

def A : Set Int := {x : Int | x^2 < 2}

theorem complement_of_A : (U \ A) = {2} := by sorry

end NUMINAMATH_CALUDE_complement_of_A_l660_66028


namespace NUMINAMATH_CALUDE_blue_jelly_bean_probability_l660_66015

/-- The probability of selecting a blue jelly bean from a bag -/
theorem blue_jelly_bean_probability :
  let red : ℕ := 5
  let green : ℕ := 6
  let yellow : ℕ := 7
  let blue : ℕ := 8
  let total : ℕ := red + green + yellow + blue
  (blue : ℚ) / total = 4 / 13 := by
  sorry

end NUMINAMATH_CALUDE_blue_jelly_bean_probability_l660_66015


namespace NUMINAMATH_CALUDE_initial_peaches_calculation_l660_66033

/-- The number of peaches Sally picked from the orchard -/
def peaches_picked : ℕ := 42

/-- The total number of peaches Sally has now -/
def total_peaches_now : ℕ := 55

/-- The initial number of peaches at Sally's roadside fruit dish -/
def initial_peaches : ℕ := total_peaches_now - peaches_picked

theorem initial_peaches_calculation :
  initial_peaches = total_peaches_now - peaches_picked :=
by sorry

end NUMINAMATH_CALUDE_initial_peaches_calculation_l660_66033


namespace NUMINAMATH_CALUDE_intersection_of_logarithmic_functions_l660_66013

theorem intersection_of_logarithmic_functions :
  ∃! x : ℝ, x > 0 ∧ 3 * Real.log x = Real.log (3 * (x - 1)) :=
by sorry

end NUMINAMATH_CALUDE_intersection_of_logarithmic_functions_l660_66013


namespace NUMINAMATH_CALUDE_point_in_first_quadrant_l660_66062

/-- Given a complex equation, prove that the point is in the first quadrant -/
theorem point_in_first_quadrant (x y : ℝ) (h : x + y + (x - y) * Complex.I = 3 - Complex.I) :
  x > 0 ∧ y > 0 := by
  sorry

end NUMINAMATH_CALUDE_point_in_first_quadrant_l660_66062


namespace NUMINAMATH_CALUDE_choose_seven_two_l660_66059

theorem choose_seven_two : Nat.choose 7 2 = 21 := by sorry

end NUMINAMATH_CALUDE_choose_seven_two_l660_66059


namespace NUMINAMATH_CALUDE_smallest_root_of_unity_exponent_l660_66042

/-- The smallest positive integer n such that all roots of z^3 - z + 1 = 0 are n^th roots of unity is 5 -/
theorem smallest_root_of_unity_exponent : ∃ (n : ℕ), n > 0 ∧ 
  (∀ (z : ℂ), z^3 - z + 1 = 0 → z^n = 1) ∧
  (∀ (m : ℕ), m > 0 → (∀ (z : ℂ), z^3 - z + 1 = 0 → z^m = 1) → m ≥ n) ∧
  n = 5 := by
  sorry

end NUMINAMATH_CALUDE_smallest_root_of_unity_exponent_l660_66042


namespace NUMINAMATH_CALUDE_simple_interest_principal_l660_66049

/-- Simple interest calculation -/
theorem simple_interest_principal (SI : ℝ) (R : ℝ) (T : ℝ) (P : ℝ) 
  (h1 : SI = 4.8)
  (h2 : T = 12)
  (h3 : R = 0.05)
  (h4 : SI = P * R * T) :
  P = 8 := by
  sorry

end NUMINAMATH_CALUDE_simple_interest_principal_l660_66049


namespace NUMINAMATH_CALUDE_towel_bleaching_l660_66053

theorem towel_bleaching (L B : ℝ) (hL : L > 0) (hB : B > 0) :
  let L' := L * (1 - x)
  let B' := B * 0.85
  L' * B' = (L * B) * 0.595
  →
  x = 0.3
  := by sorry

end NUMINAMATH_CALUDE_towel_bleaching_l660_66053


namespace NUMINAMATH_CALUDE_sum_of_x_solutions_is_zero_l660_66075

theorem sum_of_x_solutions_is_zero :
  ∀ x₁ x₂ : ℝ,
  (∃ y : ℝ, y = 8 ∧ x₁^2 + y^2 = 169) ∧
  (∃ y : ℝ, y = 8 ∧ x₂^2 + y^2 = 169) ∧
  (∀ x : ℝ, (∃ y : ℝ, y = 8 ∧ x^2 + y^2 = 169) → (x = x₁ ∨ x = x₂)) →
  x₁ + x₂ = 0 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_x_solutions_is_zero_l660_66075


namespace NUMINAMATH_CALUDE_middle_part_of_proportional_division_l660_66030

theorem middle_part_of_proportional_division (total : ℚ) (p1 p2 p3 : ℚ) :
  total = 96 →
  p1 + p2 + p3 = total →
  p2 = (1/4) * p1 →
  p3 = (1/8) * p1 →
  p2 = 17 + 21/44 :=
by sorry

end NUMINAMATH_CALUDE_middle_part_of_proportional_division_l660_66030


namespace NUMINAMATH_CALUDE_quadratic_intercept_problem_l660_66048

/-- Given two quadratic functions with specific y-intercepts and rational x-intercepts, prove that h = 1 -/
theorem quadratic_intercept_problem (j k : ℚ) : 
  (∃ x₁ x₂ : ℚ, 4 * (x₁ - 1)^2 + j = 0 ∧ 4 * (x₂ - 1)^2 + j = 0 ∧ x₁ ≠ x₂) →
  (∃ y₁ y₂ : ℚ, 3 * (y₁ - 1)^2 + k = 0 ∧ 3 * (y₂ - 1)^2 + k = 0 ∧ y₁ ≠ y₂) →
  4 * 1^2 + j = 2021 →
  3 * 1^2 + k = 2022 →
  (∀ h : ℚ, (∃ x₁ x₂ : ℚ, 4 * (x₁ - h)^2 + j = 0 ∧ 4 * (x₂ - h)^2 + j = 0 ∧ x₁ ≠ x₂) →
             (∃ y₁ y₂ : ℚ, 3 * (y₁ - h)^2 + k = 0 ∧ 3 * (y₂ - h)^2 + k = 0 ∧ y₁ ≠ y₂) →
             4 * h^2 + j = 2021 →
             3 * h^2 + k = 2022 →
             h = 1) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_intercept_problem_l660_66048


namespace NUMINAMATH_CALUDE_farm_animals_count_l660_66002

/-- Represents the number of animals on a farm --/
structure FarmAnimals where
  cows : ℕ
  dogs : ℕ
  cats : ℕ
  sheep : ℕ
  chickens : ℕ

/-- Calculates the total number of animals on the farm --/
def totalAnimals (farm : FarmAnimals) : ℕ :=
  farm.cows + farm.dogs + farm.cats + farm.sheep + farm.chickens

/-- Represents the initial state of the farm --/
def initialFarm : FarmAnimals where
  cows := 120
  dogs := 18
  cats := 6
  sheep := 0
  chickens := 288

/-- Applies the changes to the farm as described in the problem --/
def applyChanges (farm : FarmAnimals) : FarmAnimals :=
  let soldCows := farm.cows / 4
  let soldDogs := farm.dogs * 3 / 5
  let remainingDogs := farm.dogs - soldDogs + soldDogs  -- Sell and add back equal number
  { cows := farm.cows - soldCows,
    dogs := remainingDogs,
    cats := farm.cats,
    sheep := remainingDogs / 2,
    chickens := farm.chickens * 3 / 2 }  -- 50% increase

theorem farm_animals_count :
  totalAnimals (applyChanges initialFarm) = 555 :=
by sorry

end NUMINAMATH_CALUDE_farm_animals_count_l660_66002


namespace NUMINAMATH_CALUDE_commission_calculation_l660_66017

/-- The commission percentage for sales exceeding $500 -/
def excess_commission_percentage : ℝ :=
  -- We'll define this value and prove it's equal to 50
  50

theorem commission_calculation (total_sale : ℝ) (total_commission_percentage : ℝ) :
  total_sale = 800 →
  total_commission_percentage = 31.25 →
  (0.2 * 500 + (excess_commission_percentage / 100) * (total_sale - 500)) / total_sale = total_commission_percentage / 100 →
  excess_commission_percentage = 50 := by
  sorry

#check commission_calculation

end NUMINAMATH_CALUDE_commission_calculation_l660_66017


namespace NUMINAMATH_CALUDE_initial_candies_l660_66035

theorem initial_candies (package_size : ℕ) (added_candies : ℕ) (total_candies : ℕ) :
  package_size = 15 →
  added_candies = 4 →
  total_candies = 10 →
  total_candies - added_candies = 6 := by
  sorry

end NUMINAMATH_CALUDE_initial_candies_l660_66035


namespace NUMINAMATH_CALUDE_slope_of_line_l660_66055

theorem slope_of_line (x y : ℝ) (h : x/3 + y/2 = 1) : 
  ∃ m b : ℝ, y = m*x + b ∧ m = -2/3 := by
  sorry

end NUMINAMATH_CALUDE_slope_of_line_l660_66055


namespace NUMINAMATH_CALUDE_island_puzzle_l660_66066

-- Define the types of people
inductive PersonType
| Truthful
| Liar

-- Define the genders
inductive Gender
| Boy
| Girl

-- Define a person
structure Person where
  type : PersonType
  gender : Gender

-- Define the statements made by A and B
def statement_A (a b : Person) : Prop :=
  a.type = PersonType.Truthful → b.type = PersonType.Liar

def statement_B (a b : Person) : Prop :=
  b.gender = Gender.Boy → a.gender = Gender.Girl

-- Theorem to prove
theorem island_puzzle :
  ∃ (a b : Person),
    (statement_A a b ↔ a.type = PersonType.Truthful) ∧
    (statement_B a b ↔ b.type = PersonType.Liar) ∧
    a.type = PersonType.Truthful ∧
    a.gender = Gender.Boy ∧
    b.type = PersonType.Liar ∧
    b.gender = Gender.Boy :=
  sorry

end NUMINAMATH_CALUDE_island_puzzle_l660_66066


namespace NUMINAMATH_CALUDE_intersection_complement_theorem_l660_66047

def U : Set Nat := {1, 2, 3, 4}
def A : Set Nat := {1, 4}
def B : Set Nat := {2, 4}

theorem intersection_complement_theorem :
  A ∩ (U \ B) = {1} := by sorry

end NUMINAMATH_CALUDE_intersection_complement_theorem_l660_66047


namespace NUMINAMATH_CALUDE_sum_of_ages_l660_66051

/-- Given a son who is 27 years old and a woman whose age is three years more
    than twice her son's age, prove that the sum of their ages is 84 years. -/
theorem sum_of_ages (son_age : ℕ) (woman_age : ℕ) : son_age = 27 →
  woman_age = 2 * son_age + 3 → son_age + woman_age = 84 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_ages_l660_66051


namespace NUMINAMATH_CALUDE_new_revenue_is_354375_l660_66072

/-- Calculates the total revenue at the new price given the conditions --/
def calculate_new_revenue (price_increase : ℕ) (sales_decrease : ℕ) (revenue_increase : ℕ) (new_sales : ℕ) : ℕ :=
  let original_sales := new_sales + sales_decrease
  let original_price := (revenue_increase + price_increase * new_sales) / sales_decrease
  let new_price := original_price + price_increase
  new_price * new_sales

/-- Theorem stating that the total revenue at the new price is $354,375 --/
theorem new_revenue_is_354375 :
  calculate_new_revenue 1000 8 26000 63 = 354375 := by
  sorry

end NUMINAMATH_CALUDE_new_revenue_is_354375_l660_66072


namespace NUMINAMATH_CALUDE_primes_not_sum_of_composites_l660_66014

def is_composite (n : ℕ) : Prop := n > 1 ∧ ∃ a b, a > 1 ∧ b > 1 ∧ n = a * b

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d, d ∣ n → d = 1 ∨ d = n

def cannot_be_sum_of_two_composites (p : ℕ) : Prop :=
  is_prime p ∧ ¬∃ a b, is_composite a ∧ is_composite b ∧ p = a + b

theorem primes_not_sum_of_composites :
  {p : ℕ | cannot_be_sum_of_two_composites p} = {2, 3, 5, 7, 11} :=
sorry

end NUMINAMATH_CALUDE_primes_not_sum_of_composites_l660_66014


namespace NUMINAMATH_CALUDE_square_root_and_cube_root_l660_66012

theorem square_root_and_cube_root : 
  (∃ x : ℝ, x^2 = 16 ∧ (x = 4 ∨ x = -4)) ∧ 
  (∃ y : ℝ, y^3 = -2 ∧ y = -Real.rpow 2 (1/3)) := by
  sorry

end NUMINAMATH_CALUDE_square_root_and_cube_root_l660_66012


namespace NUMINAMATH_CALUDE_eighth_term_geometric_sequence_l660_66037

theorem eighth_term_geometric_sequence (a₁ : ℚ) (r : ℚ) (n : ℕ) :
  a₁ = 27 ∧ r = 1/3 ∧ n = 8 →
  a₁ * r^(n - 1) = 1/81 := by
  sorry

end NUMINAMATH_CALUDE_eighth_term_geometric_sequence_l660_66037


namespace NUMINAMATH_CALUDE_bakery_storage_ratio_l660_66044

/-- Proves that the ratio of sugar to flour is 3:8 given the conditions in the bakery storage room --/
theorem bakery_storage_ratio : ∀ (flour baking_soda : ℕ),
  flour = 10 * baking_soda →
  flour = 8 * (baking_soda + 60) →
  (900 : ℕ) / flour = 3 / 8 :=
by
  sorry

#check bakery_storage_ratio

end NUMINAMATH_CALUDE_bakery_storage_ratio_l660_66044


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l660_66005

theorem imaginary_part_of_z (z : ℂ) (h : (1 + Complex.I) * z = Complex.I) :
  z.im = 1/2 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l660_66005


namespace NUMINAMATH_CALUDE_integers_between_cubes_l660_66080

theorem integers_between_cubes : ∃ n : ℕ, n = (⌊(10.6 : ℝ)^3⌋ - ⌈(10.5 : ℝ)^3⌉ + 1) ∧ n = 33 := by
  sorry

end NUMINAMATH_CALUDE_integers_between_cubes_l660_66080


namespace NUMINAMATH_CALUDE_t_range_l660_66085

/-- The quadratic function -/
def f (x : ℝ) : ℝ := -x^2 + 6*x - 7

/-- The maximum value function -/
def y_max (t : ℝ) : ℝ := -(t-3)^2 + 2

/-- Theorem stating the range of t -/
theorem t_range (t : ℝ) :
  (∀ x, t ≤ x ∧ x ≤ t+2 → f x ≤ y_max t) →
  (∃ x, t ≤ x ∧ x ≤ t+2 ∧ f x = y_max t) →
  t ≥ 3 :=
sorry

end NUMINAMATH_CALUDE_t_range_l660_66085


namespace NUMINAMATH_CALUDE_henry_collection_cost_l660_66079

/-- The amount of money Henry needs to finish his action figure collection -/
def money_needed (current : ℕ) (total : ℕ) (cost : ℕ) : ℕ :=
  (total - current) * cost

/-- Theorem stating that Henry needs $30 to finish his collection -/
theorem henry_collection_cost :
  let current := 3  -- Henry's current number of action figures
  let total := 8    -- Total number of action figures needed for a complete collection
  let cost := 6     -- Cost of each action figure in dollars
  money_needed current total cost = 30 := by
sorry

end NUMINAMATH_CALUDE_henry_collection_cost_l660_66079


namespace NUMINAMATH_CALUDE_unique_solution_l660_66040

-- Define the machine's rule
def machineRule (n : ℕ) : ℕ :=
  if n % 2 = 0 then n / 2 else 5 * n + 2

-- Define a function that applies the rule n times
def applyNTimes (n : ℕ) (x : ℕ) : ℕ :=
  match n with
  | 0 => x
  | n + 1 => applyNTimes n (machineRule x)

-- Theorem statement
theorem unique_solution : ∀ n : ℕ, n > 0 → (applyNTimes 6 n = 4 ↔ n = 256) :=
sorry

end NUMINAMATH_CALUDE_unique_solution_l660_66040


namespace NUMINAMATH_CALUDE_total_pencils_is_64_l660_66093

/-- The number of pencils Reeta has -/
def reeta_pencils : ℕ := 20

/-- The number of pencils Anika has -/
def anika_pencils : ℕ := 2 * reeta_pencils + 4

/-- The total number of pencils Anika and Reeta have together -/
def total_pencils : ℕ := reeta_pencils + anika_pencils

/-- Theorem stating that the total number of pencils is 64 -/
theorem total_pencils_is_64 : total_pencils = 64 := by
  sorry

end NUMINAMATH_CALUDE_total_pencils_is_64_l660_66093


namespace NUMINAMATH_CALUDE_range_when_a_b_one_values_of_a_b_for_range_zero_one_max_min_a_squared_plus_b_squared_l660_66025

-- Define the function f
def f (a b x : ℝ) : ℝ := x^2 + a*x + b

-- Part I(i)
theorem range_when_a_b_one (x : ℝ) (h : x ∈ Set.Icc 0 1) :
  f 1 1 x ∈ Set.Icc 1 3 :=
sorry

-- Part I(ii)
theorem values_of_a_b_for_range_zero_one :
  (∀ x ∈ Set.Icc 0 1, f a b x ∈ Set.Icc 0 1) →
  ((a = 0 ∧ b = 0) ∨ (a = -2 ∧ b = 1)) :=
sorry

-- Part II
theorem max_min_a_squared_plus_b_squared 
  (h1 : ∀ x : ℝ, |x| ≥ 2 → f a b x ≥ 0)
  (h2 : ∃ x ∈ Set.Ioo 2 3, ∀ y ∈ Set.Ioo 2 3, f a b x ≥ f a b y)
  (h3 : ∃ x ∈ Set.Ioo 2 3, f a b x = 1) :
  (a^2 + b^2 ≥ 32 ∧ a^2 + b^2 ≤ 74) :=
sorry

end NUMINAMATH_CALUDE_range_when_a_b_one_values_of_a_b_for_range_zero_one_max_min_a_squared_plus_b_squared_l660_66025


namespace NUMINAMATH_CALUDE_locus_of_centers_l660_66038

/-- Circle C1 with center (1,1) and radius 2 -/
def C1 (x y : ℝ) : Prop := (x - 1)^2 + (y - 1)^2 = 4

/-- Circle C2 with center (4,1) and radius 3 -/
def C2 (x y : ℝ) : Prop := (x - 4)^2 + (y - 1)^2 = 9

/-- A circle with center (a,b) and radius r -/
def Circle (a b r : ℝ) (x y : ℝ) : Prop := (x - a)^2 + (y - b)^2 = r^2

/-- External tangency condition -/
def ExternallyTangent (a b r : ℝ) : Prop := (a - 1)^2 + (b - 1)^2 = (r + 2)^2

/-- Internal tangency condition -/
def InternallyTangent (a b r : ℝ) : Prop := (a - 4)^2 + (b - 1)^2 = (3 - r)^2

/-- The locus equation -/
def LocusEquation (a b : ℝ) : Prop := 84*a^2 + 100*b^2 - 336*a - 200*b + 900 = 0

theorem locus_of_centers :
  ∀ a b : ℝ, (∃ r : ℝ, ExternallyTangent a b r ∧ InternallyTangent a b r) ↔ LocusEquation a b :=
sorry

end NUMINAMATH_CALUDE_locus_of_centers_l660_66038


namespace NUMINAMATH_CALUDE_unique_solution_condition_l660_66087

theorem unique_solution_condition (a : ℝ) : 
  (∃! x : ℝ, Real.sqrt (x + 1/2 + Real.sqrt (x + 1/4)) + x = a) ↔ a ≥ 1/4 := by
sorry

end NUMINAMATH_CALUDE_unique_solution_condition_l660_66087


namespace NUMINAMATH_CALUDE_problem_solution_l660_66027

theorem problem_solution (a b : ℝ) (h : (a - 1)^2 + |b + 2| = 0) :
  2 * (5 * a^2 - 7 * a * b + 9 * b^2) - 3 * (14 * a^2 - 2 * a * b + 3 * b^2) = 20 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l660_66027


namespace NUMINAMATH_CALUDE_meaningful_square_root_range_l660_66000

theorem meaningful_square_root_range (x : ℝ) :
  (∃ y : ℝ, y = (Real.sqrt (x + 3)) / x) ↔ x ≥ -3 ∧ x ≠ 0 := by
sorry

end NUMINAMATH_CALUDE_meaningful_square_root_range_l660_66000


namespace NUMINAMATH_CALUDE_trailing_zeros_28_factorial_l660_66056

/-- The number of trailing zeros in n! -/
def trailingZeros (n : ℕ) : ℕ :=
  (n / 5) + (n / 25) + (n / 125) + (n / 625)

/-- Theorem: The number of trailing zeros in 28! is 6 -/
theorem trailing_zeros_28_factorial :
  trailingZeros 28 = 6 := by
  sorry

end NUMINAMATH_CALUDE_trailing_zeros_28_factorial_l660_66056


namespace NUMINAMATH_CALUDE_expression_equals_one_l660_66082

theorem expression_equals_one (a : ℝ) (h : a = Real.sqrt 2) : 
  ((a + 1) / (a + 2) + 1 / (a - 2)) / (2 / (a^2 - 4)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_expression_equals_one_l660_66082


namespace NUMINAMATH_CALUDE_circle_properties_l660_66036

/-- Given a circle with polar equation ρ²-4√2ρcos(θ-π/4)+6=0, prove its properties -/
theorem circle_properties (ρ θ : ℝ) :
  ρ^2 - 4 * Real.sqrt 2 * ρ * Real.cos (θ - π/4) + 6 = 0 →
  ∃ (x y : ℝ),
    -- Standard equation
    x^2 + y^2 - 4*x - 4*y + 6 = 0 ∧
    -- Parametric equations
    x = 2 + Real.sqrt 2 * Real.cos θ ∧
    y = 2 + Real.sqrt 2 * Real.sin θ ∧
    -- Maximum and minimum values of x⋅y
    (∀ (x' y' : ℝ), x'^2 + y'^2 - 4*x' - 4*y' + 6 = 0 → x'*y' ≤ 9) ∧
    (∀ (x' y' : ℝ), x'^2 + y'^2 - 4*x' - 4*y' + 6 = 0 → x'*y' ≥ 1) ∧
    (∃ (x' y' : ℝ), x'^2 + y'^2 - 4*x' - 4*y' + 6 = 0 ∧ x'*y' = 9) ∧
    (∃ (x' y' : ℝ), x'^2 + y'^2 - 4*x' - 4*y' + 6 = 0 ∧ x'*y' = 1) :=
by sorry

end NUMINAMATH_CALUDE_circle_properties_l660_66036


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_l660_66061

/-- Represents a geometric sequence with positive terms -/
structure GeometricSequence where
  a : ℕ → ℝ
  pos : ∀ n, a n > 0
  ratio : ℝ
  ratio_pos : ratio > 0
  geom : ∀ n, a (n + 1) = a n * ratio

/-- Theorem: In a geometric sequence with positive terms, if a₁a₃ = 4 and a₂ + a₄ = 10, then the common ratio is 2 -/
theorem geometric_sequence_ratio 
  (seq : GeometricSequence)
  (h1 : seq.a 1 * seq.a 3 = 4)
  (h2 : seq.a 2 + seq.a 4 = 10) :
  seq.ratio = 2 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_l660_66061


namespace NUMINAMATH_CALUDE_inequality_solution_set_l660_66092

theorem inequality_solution_set (x : ℝ) : 4 * x < 3 * x + 2 ↔ x < 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l660_66092


namespace NUMINAMATH_CALUDE_coin_toss_sequence_count_l660_66076

/-- Represents a coin toss sequence. -/
def CoinSequence := List Bool

/-- Counts the number of occurrences of a given subsequence in a coin sequence. -/
def countSubsequence (seq : CoinSequence) (subseq : List Bool) : Nat :=
  sorry

/-- Checks if a coin sequence satisfies the given conditions. -/
def isValidSequence (seq : CoinSequence) : Prop :=
  seq.length = 20 ∧
  countSubsequence seq [true, true] = 3 ∧
  countSubsequence seq [true, false] = 4 ∧
  countSubsequence seq [false, true] = 5 ∧
  countSubsequence seq [false, false] = 7

/-- The number of valid coin toss sequences. -/
def validSequenceCount : Nat :=
  sorry

theorem coin_toss_sequence_count :
  validSequenceCount = 11550 :=
sorry

end NUMINAMATH_CALUDE_coin_toss_sequence_count_l660_66076


namespace NUMINAMATH_CALUDE_gcd_factorial_eight_and_factorial_six_squared_l660_66001

theorem gcd_factorial_eight_and_factorial_six_squared : 
  Nat.gcd (Nat.factorial 8) ((Nat.factorial 6)^2) = 11520 := by
  sorry

end NUMINAMATH_CALUDE_gcd_factorial_eight_and_factorial_six_squared_l660_66001


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l660_66029

theorem quadratic_equation_roots (p q : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁^2 + p*x₁ + q = 0 ∧ x₂^2 + p*x₂ + q = 0 ∧ 
   x₁ - x₂ = 5 ∧ x₁^3 - x₂^3 = 35) →
  ((p = 1 ∧ q = -6) ∨ (p = -1 ∧ q = -6)) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l660_66029


namespace NUMINAMATH_CALUDE_bus_ticket_impossibility_prove_bus_ticket_impossibility_l660_66031

theorem bus_ticket_impossibility 
  (num_passengers : ℕ) 
  (ticket_price : ℕ) 
  (coin_denominations : List ℕ) 
  (total_coins : ℕ) : Prop :=
  num_passengers = 40 →
  ticket_price = 5 →
  coin_denominations = [10, 15, 20] →
  total_coins = 49 →
  ¬∃ (payment : List ℕ),
    payment.sum = num_passengers * ticket_price ∧
    payment.length ≤ total_coins - num_passengers ∧
    ∀ c ∈ payment, c ∈ coin_denominations

theorem prove_bus_ticket_impossibility : 
  bus_ticket_impossibility 40 5 [10, 15, 20] 49 := by
  sorry

end NUMINAMATH_CALUDE_bus_ticket_impossibility_prove_bus_ticket_impossibility_l660_66031


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sin_value_l660_66084

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sin_value (a : ℕ → ℝ) :
  arithmetic_sequence a →
  a 2 + a 6 = 3 * Real.pi / 2 →
  Real.sin (2 * a 4 - Real.pi / 3) = -1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sin_value_l660_66084


namespace NUMINAMATH_CALUDE_softball_team_ratio_l660_66086

/-- Proves that for a co-ed softball team with 6 more women than men and 24 total players, 
    the ratio of men to women is 3:5 -/
theorem softball_team_ratio : 
  ∀ (men women : ℕ), 
  women = men + 6 →
  men + women = 24 →
  (men : ℚ) / women = 3 / 5 := by
sorry

end NUMINAMATH_CALUDE_softball_team_ratio_l660_66086


namespace NUMINAMATH_CALUDE_intersection_points_product_l660_66003

theorem intersection_points_product (m : ℝ) (x₁ x₂ : ℝ) (h₁ : 0 < x₁) (h₂ : 0 < x₂) :
  (∃ y₁ y₂ : ℝ, (Real.log x₁ - 1 / x₁ = m * x₁ ∧ Real.log x₂ - 1 / x₂ = m * x₂) ∧ x₁ ≠ x₂) →
  x₁ * x₂ > 2 * Real.exp 2 :=
by sorry

end NUMINAMATH_CALUDE_intersection_points_product_l660_66003


namespace NUMINAMATH_CALUDE_complex_number_location_l660_66071

theorem complex_number_location (z : ℂ) : 
  z = Complex.mk (Real.sin (2019 * π / 180)) (Real.cos (2019 * π / 180)) →
  Real.sin (2019 * π / 180) < 0 ∧ Real.cos (2019 * π / 180) < 0 :=
by
  sorry

#check complex_number_location

end NUMINAMATH_CALUDE_complex_number_location_l660_66071


namespace NUMINAMATH_CALUDE_product_max_value_l660_66043

theorem product_max_value (x y z u : ℝ) 
  (h_nonneg : x ≥ 0 ∧ y ≥ 0 ∧ z ≥ 0 ∧ u ≥ 0) 
  (h_constraint : 2*x + x*y + z + y*z*u = 1) : 
  x^2 * y^2 * z^2 * u ≤ 1/512 := by
  sorry

end NUMINAMATH_CALUDE_product_max_value_l660_66043


namespace NUMINAMATH_CALUDE_car_speed_problem_l660_66046

theorem car_speed_problem (distance_AB : ℝ) (speed_A : ℝ) (time : ℝ) (final_distance : ℝ) :
  distance_AB = 300 →
  speed_A = 40 →
  time = 2 →
  final_distance = 100 →
  (∃ speed_B : ℝ, speed_B = 140 ∨ speed_B = 60) :=
by sorry

end NUMINAMATH_CALUDE_car_speed_problem_l660_66046


namespace NUMINAMATH_CALUDE_max_value_product_l660_66052

theorem max_value_product (a b c x y z : ℝ) 
  (non_neg : 0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c ∧ 0 ≤ x ∧ 0 ≤ y ∧ 0 ≤ z)
  (sum_abc : a + b + c = 1)
  (sum_xyz : x + y + z = 1) :
  (a - x^2) * (b - y^2) * (c - z^2) ≤ 1/16 :=
by sorry

end NUMINAMATH_CALUDE_max_value_product_l660_66052


namespace NUMINAMATH_CALUDE_max_abs_z5_l660_66095

theorem max_abs_z5 (z₁ z₂ z₃ z₄ z₅ : ℂ)
  (h1 : Complex.abs z₁ ≤ 1)
  (h2 : Complex.abs z₂ ≤ 1)
  (h3 : Complex.abs (2 * z₃ - (z₁ + z₂)) ≤ Complex.abs (z₁ - z₂))
  (h4 : Complex.abs (2 * z₄ - (z₁ + z₂)) ≤ Complex.abs (z₁ - z₂))
  (h5 : Complex.abs (2 * z₅ - (z₃ + z₄)) ≤ Complex.abs (z₃ - z₄)) :
  Complex.abs z₅ ≤ Real.sqrt 3 ∧ ∃ z₁ z₂ z₃ z₄ z₅ : ℂ, 
    Complex.abs z₁ ≤ 1 ∧
    Complex.abs z₂ ≤ 1 ∧
    Complex.abs (2 * z₃ - (z₁ + z₂)) ≤ Complex.abs (z₁ - z₂) ∧
    Complex.abs (2 * z₄ - (z₁ + z₂)) ≤ Complex.abs (z₁ - z₂) ∧
    Complex.abs (2 * z₅ - (z₃ + z₄)) ≤ Complex.abs (z₃ - z₄) ∧
    Complex.abs z₅ = Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_max_abs_z5_l660_66095


namespace NUMINAMATH_CALUDE_juice_cost_calculation_l660_66070

theorem juice_cost_calculation (orange_cost apple_cost total_bottles orange_bottles : ℕ) 
  (h1 : orange_cost = 70)
  (h2 : apple_cost = 60)
  (h3 : total_bottles = 70)
  (h4 : orange_bottles = 42) :
  orange_cost * orange_bottles + apple_cost * (total_bottles - orange_bottles) = 4620 := by
  sorry

#check juice_cost_calculation

end NUMINAMATH_CALUDE_juice_cost_calculation_l660_66070


namespace NUMINAMATH_CALUDE_evies_age_l660_66010

theorem evies_age (x : ℕ) : x + 4 = 3 * (x - 2) → x + 1 = 6 := by
  sorry

end NUMINAMATH_CALUDE_evies_age_l660_66010


namespace NUMINAMATH_CALUDE_fermat_little_theorem_l660_66039

theorem fermat_little_theorem (p : ℕ) (a : ℤ) (h : Nat.Prime p) :
  ∃ k : ℤ, a^p - a = k * p :=
sorry

end NUMINAMATH_CALUDE_fermat_little_theorem_l660_66039


namespace NUMINAMATH_CALUDE_league_games_count_l660_66007

/-- Calculates the number of games in a round-robin tournament. -/
def numGames (n : ℕ) (k : ℕ) : ℕ := n * (n - 1) / 2 * k

/-- Proves that in a league with 20 teams, where each team plays every other team 4 times, 
    the total number of games played in the season is 760. -/
theorem league_games_count : numGames 20 4 = 760 := by
  sorry

end NUMINAMATH_CALUDE_league_games_count_l660_66007


namespace NUMINAMATH_CALUDE_smallest_prime_8_less_than_perfect_square_l660_66088

/-- A number is a perfect square if it's the square of some integer. -/
def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m^2

/-- A number is prime if it's greater than 1 and its only divisors are 1 and itself. -/
def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

theorem smallest_prime_8_less_than_perfect_square :
  ∃ (n : ℕ), is_prime n ∧ (∃ (m : ℕ), is_perfect_square m ∧ n = m - 8) ∧
  (∀ (k : ℕ), k < n → ¬(is_prime k ∧ ∃ (m : ℕ), is_perfect_square m ∧ k = m - 8)) :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_smallest_prime_8_less_than_perfect_square_l660_66088


namespace NUMINAMATH_CALUDE_smallest_number_l660_66099

theorem smallest_number (a b c d : ℝ) 
  (ha : a = Real.sqrt 3) 
  (hb : b = -1/3) 
  (hc : c = -2) 
  (hd : d = 0) : 
  c ≤ a ∧ c ≤ b ∧ c ≤ d :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_l660_66099


namespace NUMINAMATH_CALUDE_point_in_first_quadrant_l660_66054

/-- Given a point M(2,a) on the graph of y = k/x where k > 0, prove that the coordinates of M are both positive. -/
theorem point_in_first_quadrant (k a : ℝ) (h1 : k > 0) (h2 : a = k / 2) : 2 > 0 ∧ a > 0 := by
  sorry

end NUMINAMATH_CALUDE_point_in_first_quadrant_l660_66054


namespace NUMINAMATH_CALUDE_walking_scenario_theorem_l660_66045

/-- Represents the walking scenario with Yolanda, Bob, and Jim -/
structure WalkingScenario where
  total_distance : ℝ
  yolanda_speed : ℝ
  bob_speed_difference : ℝ
  jim_speed : ℝ
  yolanda_head_start : ℝ

/-- Calculates the distance Bob walked when he met Yolanda -/
def bob_distance_walked (scenario : WalkingScenario) : ℝ :=
  sorry

/-- Calculates the point where Jim and Yolanda met, measured from point X -/
def jim_yolanda_meeting_point (scenario : WalkingScenario) : ℝ :=
  sorry

/-- Theorem stating the correct distances for Bob and Jim -/
theorem walking_scenario_theorem (scenario : WalkingScenario) 
  (h1 : scenario.total_distance = 80)
  (h2 : scenario.yolanda_speed = 4)
  (h3 : scenario.bob_speed_difference = 2)
  (h4 : scenario.jim_speed = 5)
  (h5 : scenario.yolanda_head_start = 1) :
  bob_distance_walked scenario = 45.6 ∧ 
  jim_yolanda_meeting_point scenario = 38 :=
by sorry

end NUMINAMATH_CALUDE_walking_scenario_theorem_l660_66045


namespace NUMINAMATH_CALUDE_divisible_by_27_l660_66022

theorem divisible_by_27 (n : ℕ) : ∃ k : ℤ, (10 : ℤ)^n + 18*n - 1 = 27*k := by
  sorry

end NUMINAMATH_CALUDE_divisible_by_27_l660_66022


namespace NUMINAMATH_CALUDE_complex_number_existence_l660_66067

theorem complex_number_existence : ∃ z : ℂ, 
  (∃ r : ℝ, z + 5 / z = r) ∧ 
  (Complex.re (z + 3) = 2 * Complex.im z) ∧ 
  (z = (1 : ℝ) + (2 : ℝ) * Complex.I ∨ z = (-11 : ℝ) / 5 - (2 : ℝ) / 5 * Complex.I) :=
by sorry

end NUMINAMATH_CALUDE_complex_number_existence_l660_66067


namespace NUMINAMATH_CALUDE_halloween_candy_count_l660_66026

/-- The number of candy pieces Robin scored on Halloween -/
def initial_candy : ℕ := 23

/-- The number of candy pieces Robin ate -/
def eaten_candy : ℕ := 7

/-- The number of candy pieces Robin's sister gave her -/
def sister_candy : ℕ := 21

/-- The number of candy pieces Robin has now -/
def current_candy : ℕ := 37

/-- Theorem stating that the initial candy count is correct -/
theorem halloween_candy_count : 
  initial_candy - eaten_candy + sister_candy = current_candy := by
  sorry

end NUMINAMATH_CALUDE_halloween_candy_count_l660_66026


namespace NUMINAMATH_CALUDE_rotten_apples_l660_66019

/-- Given a problem about apples in crates and boxes, prove the number of rotten apples. -/
theorem rotten_apples (apples_per_crate : ℕ) (num_crates : ℕ) (apples_per_box : ℕ) (num_boxes : ℕ)
  (h1 : apples_per_crate = 42)
  (h2 : num_crates = 12)
  (h3 : apples_per_box = 10)
  (h4 : num_boxes = 50) :
  apples_per_crate * num_crates - apples_per_box * num_boxes = 4 := by
  sorry

#check rotten_apples

end NUMINAMATH_CALUDE_rotten_apples_l660_66019


namespace NUMINAMATH_CALUDE_simplify_expression_l660_66068

theorem simplify_expression (x y : ℝ) (hx : x = 3) (hy : y = 4) :
  (9 * x^2 * y^3) / (12 * x * y^2) = 9 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l660_66068


namespace NUMINAMATH_CALUDE_circle_area_problem_l660_66011

/-- The area of the region outside a circle of radius 1.5 and inside two circles of radius 2
    that are internally tangent to the smaller circle at opposite ends of its diameter -/
theorem circle_area_problem : ∃ (area : ℝ),
  let r₁ : ℝ := 1.5 -- radius of smaller circle
  let r₂ : ℝ := 2   -- radius of larger circles
  area = (13/4 : ℝ) * Real.pi - 3 * Real.sqrt 1.75 ∧
  area = 2 * (
    -- Area of sector in larger circle
    (1/3 : ℝ) * Real.pi * r₂^2 -
    -- Area of triangle
    (1/2 : ℝ) * r₁ * Real.sqrt (r₂^2 - r₁^2) -
    -- Area of quarter of smaller circle
    (1/4 : ℝ) * Real.pi * r₁^2
  ) := by sorry


end NUMINAMATH_CALUDE_circle_area_problem_l660_66011


namespace NUMINAMATH_CALUDE_water_consumption_ratio_l660_66006

theorem water_consumption_ratio (initial_volume : ℝ) (first_drink_fraction : ℝ) (final_volume : ℝ) :
  initial_volume = 4 →
  first_drink_fraction = 1/4 →
  final_volume = 1 →
  let remaining_after_first := initial_volume - first_drink_fraction * initial_volume
  let second_drink := remaining_after_first - final_volume
  (second_drink / remaining_after_first) = 2/3 := by sorry

end NUMINAMATH_CALUDE_water_consumption_ratio_l660_66006
