import Mathlib

namespace NUMINAMATH_CALUDE_power_fraction_equality_l4122_412296

theorem power_fraction_equality : (2^8 : ℚ) / (8^2 : ℚ) = 4 := by
  sorry

end NUMINAMATH_CALUDE_power_fraction_equality_l4122_412296


namespace NUMINAMATH_CALUDE_number_of_convertibles_l4122_412228

-- Define the total number of cars
def total_cars : ℕ := 125

-- Define the percentage of regular cars
def regular_car_percentage : ℚ := 64 / 100

-- Define the percentage of trucks
def truck_percentage : ℚ := 8 / 100

-- Theorem statement
theorem number_of_convertibles :
  ∃ (regular_cars trucks convertibles : ℕ),
    regular_cars + trucks + convertibles = total_cars ∧
    regular_cars = (regular_car_percentage * total_cars).floor ∧
    trucks = (truck_percentage * total_cars).floor ∧
    convertibles = 35 := by
  sorry

end NUMINAMATH_CALUDE_number_of_convertibles_l4122_412228


namespace NUMINAMATH_CALUDE_sqrt_four_fourth_powers_l4122_412245

theorem sqrt_four_fourth_powers : Real.sqrt (4^4 + 4^4 + 4^4 + 4^4) = 32 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_four_fourth_powers_l4122_412245


namespace NUMINAMATH_CALUDE_cake_recipe_ratio_l4122_412278

/-- Given a recipe with 60 eggs and a total of 90 cups of flour and eggs,
    prove that the ratio of cups of flour to eggs is 1:2. -/
theorem cake_recipe_ratio : 
  ∀ (flour eggs : ℕ), 
    eggs = 60 →
    flour + eggs = 90 →
    (flour : ℚ) / (eggs : ℚ) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cake_recipe_ratio_l4122_412278


namespace NUMINAMATH_CALUDE_qin_jiushao_algorithm_correct_l4122_412223

def f (x : ℝ) : ℝ := 12 + 35*x - 8*x^2 + 79*x^3 + 6*x^4 + 5*x^5 + 3*x^6

def qin_jiushao_v (n : ℕ) (x : ℝ) : ℝ :=
  match n with
  | 0 => 3
  | 1 => qin_jiushao_v 0 x * x + 5
  | 2 => qin_jiushao_v 1 x * x + 6
  | 3 => qin_jiushao_v 2 x * x + 79
  | 4 => qin_jiushao_v 3 x * x - 8
  | _ => 0  -- For completeness, though we only need up to v₄

theorem qin_jiushao_algorithm_correct :
  qin_jiushao_v 4 (-4) = 220 :=
sorry

end NUMINAMATH_CALUDE_qin_jiushao_algorithm_correct_l4122_412223


namespace NUMINAMATH_CALUDE_product_in_fourth_quadrant_l4122_412219

-- Define complex numbers z1 and z2
def z1 : ℂ := 2 + Complex.I
def z2 : ℂ := 1 - Complex.I

-- Define the product z
def z : ℂ := z1 * z2

-- Theorem statement
theorem product_in_fourth_quadrant :
  z.re > 0 ∧ z.im < 0 :=
sorry

end NUMINAMATH_CALUDE_product_in_fourth_quadrant_l4122_412219


namespace NUMINAMATH_CALUDE_mailbox_probability_l4122_412260

-- Define the number of mailboxes
def num_mailboxes : ℕ := 2

-- Define the number of letters
def num_letters : ℕ := 3

-- Define the function to calculate the total number of ways to distribute letters
def total_ways : ℕ := 2^num_letters

-- Define the function to calculate the number of favorable ways
def favorable_ways : ℕ := (num_letters.choose (num_letters - 1)) * (num_mailboxes^(num_mailboxes - 1))

-- Define the probability
def probability : ℚ := favorable_ways / total_ways

-- Theorem statement
theorem mailbox_probability : probability = 3/4 := by sorry

end NUMINAMATH_CALUDE_mailbox_probability_l4122_412260


namespace NUMINAMATH_CALUDE_max_discount_rate_l4122_412251

/-- Proves the maximum discount rate for a given cost price, selling price, and minimum profit margin. -/
theorem max_discount_rate 
  (cost_price : ℝ) 
  (selling_price : ℝ) 
  (min_profit_margin : ℝ) 
  (h1 : cost_price = 4)
  (h2 : selling_price = 5)
  (h3 : min_profit_margin = 0.1)
  : ∃ (max_discount : ℝ), 
    max_discount = 12 ∧ 
    ∀ (discount : ℝ), 
      discount ≤ max_discount ↔ 
      selling_price * (1 - discount / 100) ≥ cost_price * (1 + min_profit_margin) :=
by sorry

end NUMINAMATH_CALUDE_max_discount_rate_l4122_412251


namespace NUMINAMATH_CALUDE_x_plus_y_values_l4122_412220

theorem x_plus_y_values (x y : ℝ) (h1 : |x| = 3) (h2 : y^2 = 4) (h3 : x * y < 0) :
  x + y = 1 ∨ x + y = -1 := by
  sorry

end NUMINAMATH_CALUDE_x_plus_y_values_l4122_412220


namespace NUMINAMATH_CALUDE_probability_two_in_same_group_l4122_412259

/-- The probability of two specific individuals being in the same group when dividing 4 individuals into two equal groups -/
def probability_same_group : ℚ := 1 / 3

/-- The number of ways to divide 4 individuals into two equal groups -/
def total_ways : ℕ := 3

/-- The number of ways to have two specific individuals in the same group when dividing 4 individuals into two equal groups -/
def favorable_ways : ℕ := 1

theorem probability_two_in_same_group :
  probability_same_group = favorable_ways / total_ways := by
  sorry

#eval probability_same_group

end NUMINAMATH_CALUDE_probability_two_in_same_group_l4122_412259


namespace NUMINAMATH_CALUDE_new_boarders_count_l4122_412273

/-- The number of new boarders that joined the school -/
def new_boarders : ℕ := 30

/-- The initial number of boarders -/
def initial_boarders : ℕ := 150

/-- The initial ratio of boarders to day students -/
def initial_ratio : ℚ := 5 / 12

/-- The final ratio of boarders to day students -/
def final_ratio : ℚ := 1 / 2

theorem new_boarders_count :
  ∃ (initial_day_students : ℕ),
    (initial_boarders : ℚ) / initial_day_students = initial_ratio ∧
    (initial_boarders + new_boarders : ℚ) / initial_day_students = final_ratio :=
by sorry

end NUMINAMATH_CALUDE_new_boarders_count_l4122_412273


namespace NUMINAMATH_CALUDE_cube_sum_over_product_l4122_412207

theorem cube_sum_over_product (x y z : ℂ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0)
  (h_sum : x + y + z = 10) (h_diff_sq : (x - y)^2 + (x - z)^2 + (y - z)^2 = 2*x*y*z) :
  (x^3 + y^3 + z^3) / (x*y*z) = 13 := by
sorry

end NUMINAMATH_CALUDE_cube_sum_over_product_l4122_412207


namespace NUMINAMATH_CALUDE_triangle_area_is_36_sqrt_21_l4122_412238

/-- Triangle with an incircle that trisects a median -/
structure TriangleWithTrisectingIncircle where
  /-- Side length QR -/
  qr : ℝ
  /-- Radius of the incircle -/
  r : ℝ
  /-- Length of the median PS -/
  ps : ℝ
  /-- The incircle evenly trisects the median PS -/
  trisects_median : ps = 3 * r
  /-- QR equals 30 -/
  qr_length : qr = 30

/-- The area of a triangle with a trisecting incircle -/
def triangle_area (t : TriangleWithTrisectingIncircle) : ℝ := sorry

/-- Theorem stating the area of the specific triangle -/
theorem triangle_area_is_36_sqrt_21 (t : TriangleWithTrisectingIncircle) :
  triangle_area t = 36 * Real.sqrt 21 := by sorry

end NUMINAMATH_CALUDE_triangle_area_is_36_sqrt_21_l4122_412238


namespace NUMINAMATH_CALUDE_polynomial_coefficient_sum_l4122_412210

theorem polynomial_coefficient_sum (b₁ b₂ b₃ c₁ c₂ : ℝ) :
  (∀ x : ℝ, x^7 - x^6 + x^5 - x^4 + x^3 - x^2 + x - 1 = 
    (x^2 + b₁*x + c₁) * (x^2 + b₂*x + c₂) * (x^2 + b₃*x + 1)) →
  b₁*c₁ + b₂*c₂ + b₃ = -1 := by
sorry

end NUMINAMATH_CALUDE_polynomial_coefficient_sum_l4122_412210


namespace NUMINAMATH_CALUDE_cards_distribution_l4122_412272

theorem cards_distribution (total_cards : Nat) (num_people : Nat) 
  (h1 : total_cards = 60) (h2 : num_people = 9) : 
  num_people - (total_cards % num_people) = 3 := by
  sorry

end NUMINAMATH_CALUDE_cards_distribution_l4122_412272


namespace NUMINAMATH_CALUDE_product_of_three_numbers_l4122_412269

/-- Given three numbers a, b, and c satisfying certain conditions, 
    prove that their product is equal to 369912000/4913 -/
theorem product_of_three_numbers (a b c m : ℚ) : 
  a + b + c = 180 ∧ 
  8 * a = m ∧ 
  b - 10 = m ∧ 
  c + 10 = m → 
  a * b * c = 369912000 / 4913 := by
  sorry

end NUMINAMATH_CALUDE_product_of_three_numbers_l4122_412269


namespace NUMINAMATH_CALUDE_range_of_a_l4122_412234

def p (x : ℝ) : Prop := 2 * x^2 - 3 * x + 1 ≤ 0

def q (x a : ℝ) : Prop := x^2 - (2 * a + 1) * x + a * (a + 1) ≤ 0

theorem range_of_a (a : ℝ) :
  (¬ ∀ x, (¬ p x ↔ ¬ q x a)) →
  (0 ≤ a ∧ a ≤ 1/2) :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l4122_412234


namespace NUMINAMATH_CALUDE_unique_prime_squared_plus_fourteen_prime_l4122_412235

theorem unique_prime_squared_plus_fourteen_prime :
  ∀ p : ℕ, Prime p → Prime (p^2 + 14) → p = 3 := by
  sorry

end NUMINAMATH_CALUDE_unique_prime_squared_plus_fourteen_prime_l4122_412235


namespace NUMINAMATH_CALUDE_complex_power_evaluation_l4122_412293

-- Define the imaginary unit i
noncomputable def i : ℂ := Complex.I

-- State the theorem
theorem complex_power_evaluation :
  3 * i ^ 44 - 2 * i ^ 333 = 3 - 2 * i :=
by sorry

end NUMINAMATH_CALUDE_complex_power_evaluation_l4122_412293


namespace NUMINAMATH_CALUDE_dihedral_angle_sum_l4122_412265

/-- A dihedral angle -/
structure DihedralAngle where
  /-- The linear angle of the dihedral angle -/
  linearAngle : ℝ
  /-- The angle between the external normals of the dihedral angle -/
  externalNormalAngle : ℝ
  /-- The linear angle is between 0 and π -/
  linearAngle_bounds : 0 < linearAngle ∧ linearAngle < π
  /-- The external normal angle is between 0 and π -/
  externalNormalAngle_bounds : 0 < externalNormalAngle ∧ externalNormalAngle < π

/-- The sum of the external normal angle and the linear angle of a dihedral angle is π -/
theorem dihedral_angle_sum (d : DihedralAngle) : 
  d.externalNormalAngle + d.linearAngle = π :=
sorry

end NUMINAMATH_CALUDE_dihedral_angle_sum_l4122_412265


namespace NUMINAMATH_CALUDE_seventh_power_equation_l4122_412212

theorem seventh_power_equation (x : ℝ) (hx : x ≠ 0) :
  (7 * x)^5 = (14 * x)^4 ↔ x = 16/7 := by sorry

end NUMINAMATH_CALUDE_seventh_power_equation_l4122_412212


namespace NUMINAMATH_CALUDE_inequality_proof_l4122_412204

theorem inequality_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : 1/a + 1/b = 1) :
  ∀ n : ℕ+, (a + b)^n.val - a^n.val - b^n.val ≥ 2^(2*n.val) - 2^(n.val+1) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l4122_412204


namespace NUMINAMATH_CALUDE_hyperbola_equation_l4122_412213

-- Define the hyperbola
structure Hyperbola where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define the standard equation of a hyperbola
def standard_equation (h : Hyperbola) (x y : ℝ) : Prop :=
  (17 * x^2) / 4 - (17 * y^2) / 64 = 1

-- Define the asymptote equation
def asymptote_equation (x y : ℝ) : Prop :=
  y = 4 * x

-- Define the focus of the parabola
def parabola_focus : ℝ × ℝ := (2, 0)

-- Theorem statement
theorem hyperbola_equation (h : Hyperbola) :
  (∃ x y : ℝ, asymptote_equation x y) ∧
  (h.c = parabola_focus.1) →
  ∀ x y : ℝ, standard_equation h x y :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l4122_412213


namespace NUMINAMATH_CALUDE_mary_remaining_stickers_l4122_412261

/-- Calculates the number of remaining stickers after Mary uses some on her journal. -/
def remaining_stickers (initial : ℕ) (front_page : ℕ) (other_pages : ℕ) (per_other_page : ℕ) : ℕ :=
  initial - (front_page + other_pages * per_other_page)

/-- Proves that Mary has 44 stickers remaining after using some on her journal. -/
theorem mary_remaining_stickers :
  remaining_stickers 89 3 6 7 = 44 := by
  sorry

end NUMINAMATH_CALUDE_mary_remaining_stickers_l4122_412261


namespace NUMINAMATH_CALUDE_logarithm_simplification_l4122_412253

theorem logarithm_simplification : 
  Real.log 4 / Real.log 10 + 2 * Real.log 5 / Real.log 10 + 4^(-1/2 : ℝ) = 5/2 := by
  sorry

end NUMINAMATH_CALUDE_logarithm_simplification_l4122_412253


namespace NUMINAMATH_CALUDE_eighth_triangular_number_l4122_412275

/-- Triangular number function -/
def triangular_number (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The 8th triangular number is 36 -/
theorem eighth_triangular_number : triangular_number 8 = 36 := by
  sorry

end NUMINAMATH_CALUDE_eighth_triangular_number_l4122_412275


namespace NUMINAMATH_CALUDE_exists_non_prime_power_plus_a_l4122_412226

theorem exists_non_prime_power_plus_a (a : ℕ) (ha : a > 1) :
  ∃ n : ℕ, ¬ Nat.Prime (2^(2^n) + a) := by
  sorry

end NUMINAMATH_CALUDE_exists_non_prime_power_plus_a_l4122_412226


namespace NUMINAMATH_CALUDE_goods_train_passing_time_l4122_412227

/-- The time taken for a goods train to pass a man in an opposite moving train -/
theorem goods_train_passing_time
  (man_train_speed : ℝ)
  (goods_train_speed : ℝ)
  (goods_train_length : ℝ)
  (h1 : man_train_speed = 55)
  (h2 : goods_train_speed = 60.2)
  (h3 : goods_train_length = 320) :
  (goods_train_length / ((man_train_speed + goods_train_speed) * (1000 / 3600))) = 10 := by
  sorry


end NUMINAMATH_CALUDE_goods_train_passing_time_l4122_412227


namespace NUMINAMATH_CALUDE_faster_train_speed_l4122_412247

theorem faster_train_speed 
  (distance : ℝ) 
  (time : ℝ) 
  (slower_speed : ℝ) 
  (h1 : distance = 536) 
  (h2 : time = 4) 
  (h3 : slower_speed = 60) :
  ∃ faster_speed : ℝ, 
    faster_speed = distance / time - slower_speed ∧ 
    faster_speed = 74 :=
by sorry

end NUMINAMATH_CALUDE_faster_train_speed_l4122_412247


namespace NUMINAMATH_CALUDE_girls_in_class_l4122_412248

theorem girls_in_class (total_students : ℕ) (girl_ratio boy_ratio : ℕ) : 
  total_students = 36 → 
  girl_ratio = 4 → 
  boy_ratio = 5 → 
  (girl_ratio + boy_ratio : ℚ) * (total_students / (girl_ratio + boy_ratio : ℕ)) = girl_ratio * (total_students / (girl_ratio + boy_ratio : ℕ)) →
  girl_ratio * (total_students / (girl_ratio + boy_ratio : ℕ)) = 16 := by
sorry

end NUMINAMATH_CALUDE_girls_in_class_l4122_412248


namespace NUMINAMATH_CALUDE_average_age_combined_group_l4122_412291

theorem average_age_combined_group (num_students : Nat) (num_parents : Nat)
  (avg_age_students : ℚ) (avg_age_parents : ℚ) :
  num_students = 40 →
  num_parents = 50 →
  avg_age_students = 13 →
  avg_age_parents = 40 →
  (num_students * avg_age_students + num_parents * avg_age_parents) / (num_students + num_parents : ℚ) = 28 := by
  sorry

end NUMINAMATH_CALUDE_average_age_combined_group_l4122_412291


namespace NUMINAMATH_CALUDE_quadratic_equations_solutions_l4122_412274

theorem quadratic_equations_solutions :
  (∃ x : ℝ, x^2 - 4*x + 1 = 0 ↔ x = 2 + Real.sqrt 3 ∨ x = 2 - Real.sqrt 3) ∧
  (∃ x : ℝ, 2*x^2 - 7*x + 5 = 0 ↔ x = 5/2 ∨ x = 1) ∧
  (∃ x : ℝ, (x + 3)^2 - 2*(x + 3) = 0 ↔ x = -3 ∨ x = -1) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equations_solutions_l4122_412274


namespace NUMINAMATH_CALUDE_f_two_zeros_iff_a_in_range_l4122_412270

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * Real.log x + x^2 - (a + 2) * x

def has_exactly_two_zeros (a : ℝ) : Prop :=
  ∃ x y : ℝ, x > 0 ∧ y > 0 ∧ x ≠ y ∧ f a x = 0 ∧ f a y = 0 ∧
  ∀ z : ℝ, z > 0 → f a z = 0 → (z = x ∨ z = y)

theorem f_two_zeros_iff_a_in_range :
  ∀ a : ℝ, has_exactly_two_zeros a ↔ -1 < a ∧ a < 0 :=
sorry

end NUMINAMATH_CALUDE_f_two_zeros_iff_a_in_range_l4122_412270


namespace NUMINAMATH_CALUDE_remaining_tickets_l4122_412232

/-- Represents the number of tickets Tom won and spent at the arcade -/
def arcade_tickets (x y : ℕ) : Prop :=
  let whack_a_mole := 32
  let skee_ball := 25
  let space_invaders := x
  let hat := 7
  let keychain := 10
  let small_toy := 15
  y = (whack_a_mole + skee_ball + space_invaders) - (hat + keychain + small_toy)

/-- Theorem stating that the number of tickets Tom has left is 25 plus the number of tickets he won from 'space invaders' -/
theorem remaining_tickets (x y : ℕ) :
  arcade_tickets x y → y = 25 + x := by
  sorry

end NUMINAMATH_CALUDE_remaining_tickets_l4122_412232


namespace NUMINAMATH_CALUDE_inequality_equivalence_l4122_412202

theorem inequality_equivalence (x : ℝ) : x - 1 ≤ (1 + x) / 3 ↔ x ≤ 2 := by sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l4122_412202


namespace NUMINAMATH_CALUDE_treasure_probability_value_l4122_412239

/-- The probability of finding exactly 4 islands with treasure and no traps out of 8 islands -/
def treasure_probability : ℚ :=
  let n : ℕ := 8  -- Total number of islands
  let k : ℕ := 4  -- Number of islands with treasure
  let p_treasure : ℚ := 1/5  -- Probability of treasure and no traps
  let p_neither : ℚ := 7/10  -- Probability of neither treasure nor traps
  Nat.choose n k * p_treasure^k * p_neither^(n-k)

/-- The probability of finding exactly 4 islands with treasure and no traps out of 8 islands
    is equal to 673/25000 -/
theorem treasure_probability_value : treasure_probability = 673/25000 := by
  sorry

end NUMINAMATH_CALUDE_treasure_probability_value_l4122_412239


namespace NUMINAMATH_CALUDE_factorization_problem_1_l4122_412201

theorem factorization_problem_1 (x : ℝ) :
  6 * (x - 3)^2 - 24 = 6 * (x - 1) * (x - 5) := by sorry

end NUMINAMATH_CALUDE_factorization_problem_1_l4122_412201


namespace NUMINAMATH_CALUDE_volume_of_bounded_figure_l4122_412224

-- Define a cube with edge length 1
def cube : Set (Fin 3 → ℝ) := {v | ∀ i, 0 ≤ v i ∧ v i ≤ 1}

-- Define the planes through centers of adjacent sides
def planes : Set (Set (Fin 3 → ℝ)) :=
  {p | ∃ (i j k : Fin 3), i ≠ j ∧ j ≠ k ∧ i ≠ k ∧
    p = {v | v i + v j + v k = 3/2}}

-- Define the bounded figure
def bounded_figure : Set (Fin 3 → ℝ) :=
  {v ∈ cube | ∀ p ∈ planes, v ∈ p}

-- Theorem statement
theorem volume_of_bounded_figure :
  MeasureTheory.volume bounded_figure = 1/2 := by sorry

end NUMINAMATH_CALUDE_volume_of_bounded_figure_l4122_412224


namespace NUMINAMATH_CALUDE_distance_to_y_axis_l4122_412263

/-- 
Given a point P with coordinates (x, -8) where the distance from the x-axis to P 
is half the distance from the y-axis to P, prove that P is 16 units from the y-axis.
-/
theorem distance_to_y_axis (x : ℝ) :
  let p : ℝ × ℝ := (x, -8)
  let dist_to_x_axis := |p.2|
  let dist_to_y_axis := |p.1|
  dist_to_x_axis = (1/2) * dist_to_y_axis →
  dist_to_y_axis = 16 := by
  sorry

end NUMINAMATH_CALUDE_distance_to_y_axis_l4122_412263


namespace NUMINAMATH_CALUDE_symmetric_point_correct_l4122_412271

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Symmetry with respect to x-axis -/
def symmetricXAxis (p : Point3D) : Point3D :=
  ⟨p.x, -p.y, -p.z⟩

/-- The original point -/
def originalPoint : Point3D :=
  ⟨2, 3, 4⟩

/-- The symmetric point -/
def symmetricPoint : Point3D :=
  ⟨2, -3, -4⟩

theorem symmetric_point_correct : symmetricXAxis originalPoint = symmetricPoint := by
  sorry

end NUMINAMATH_CALUDE_symmetric_point_correct_l4122_412271


namespace NUMINAMATH_CALUDE_min_value_of_fraction_l4122_412225

theorem min_value_of_fraction :
  ∀ x : ℝ, (1/2 * x^2 + x + 1 ≠ 0) →
  ((3 * x^2 + 6 * x + 5) / (1/2 * x^2 + x + 1) ≥ 4) ∧
  (∃ y : ℝ, (1/2 * y^2 + y + 1 ≠ 0) ∧ ((3 * y^2 + 6 * y + 5) / (1/2 * y^2 + y + 1) = 4)) :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_fraction_l4122_412225


namespace NUMINAMATH_CALUDE_chenny_spoons_l4122_412280

/-- Given the following:
  * Chenny bought 9 plates at $2 each
  * Spoons cost $1.50 each
  * The total paid for plates and spoons is $24
  Prove that Chenny bought 4 spoons -/
theorem chenny_spoons (num_plates : ℕ) (price_plate : ℚ) (price_spoon : ℚ) (total_paid : ℚ) :
  num_plates = 9 →
  price_plate = 2 →
  price_spoon = 3/2 →
  total_paid = 24 →
  (total_paid - num_plates * price_plate) / price_spoon = 4 :=
by sorry

end NUMINAMATH_CALUDE_chenny_spoons_l4122_412280


namespace NUMINAMATH_CALUDE_quadratic_equation_properties_l4122_412206

theorem quadratic_equation_properties (k : ℝ) (a b : ℝ) :
  (∀ x, x^2 + 2*x - k = 0 ↔ x = a ∨ x = b) →
  a ≠ b →
  (k > -1) ∧ (a / (a + 1) - 1 / (b + 1) = 1) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_properties_l4122_412206


namespace NUMINAMATH_CALUDE_max_walk_distance_l4122_412276

/-- Represents the board system with a person walking on it. -/
structure BoardSystem where
  l : ℝ  -- Length of the board
  m : ℝ  -- Mass of the board
  x : ℝ  -- Distance the person walks from the stone

/-- The conditions for the board system to be in equilibrium. -/
def is_equilibrium (bs : BoardSystem) : Prop :=
  bs.l = 20 ∧  -- Board length is 20 meters
  bs.x ≤ bs.l ∧  -- Person cannot walk beyond the board length
  2 * bs.m * (bs.l / 4) = bs.m * (3 * bs.l / 8) + (bs.m / 2) * (bs.x - bs.l / 4)

/-- The theorem stating the maximum distance a person can walk. -/
theorem max_walk_distance (bs : BoardSystem) :
  is_equilibrium bs → bs.x = bs.l / 2 := by
  sorry

#check max_walk_distance

end NUMINAMATH_CALUDE_max_walk_distance_l4122_412276


namespace NUMINAMATH_CALUDE_river_speed_is_three_l4122_412286

/-- Represents a ship with its upstream speed -/
structure Ship where
  speed : ℝ

/-- Represents the rescue scenario -/
structure RescueScenario where
  ships : List Ship
  timeToTurn : ℝ
  distanceToRescue : ℝ
  riverSpeed : ℝ

/-- Theorem: Given the conditions, the river speed is 3 km/h -/
theorem river_speed_is_three (scenario : RescueScenario) :
  scenario.ships = [Ship.mk 4, Ship.mk 6, Ship.mk 10] →
  scenario.timeToTurn = 1 →
  scenario.distanceToRescue = 6 →
  scenario.riverSpeed = 3 := by
  sorry


end NUMINAMATH_CALUDE_river_speed_is_three_l4122_412286


namespace NUMINAMATH_CALUDE_minimum_cost_theorem_l4122_412264

/-- Represents the cost and survival properties of flower types -/
structure FlowerType where
  cost : ℝ
  survivalRate : ℝ

/-- Represents the planting scenario -/
structure PlantingScenario where
  typeA : FlowerType
  typeB : FlowerType
  totalPots : ℕ
  maxReplacement : ℕ

def minimumCost (scenario : PlantingScenario) : ℝ :=
  let m := scenario.totalPots / 2
  m * scenario.typeA.cost + (scenario.totalPots - m) * scenario.typeB.cost

theorem minimum_cost_theorem (scenario : PlantingScenario) :
  scenario.typeA.cost = 30 ∧
  scenario.typeB.cost = 60 ∧
  scenario.totalPots = 400 ∧
  scenario.typeA.survivalRate = 0.7 ∧
  scenario.typeB.survivalRate = 0.9 ∧
  scenario.maxReplacement = 80 ∧
  3 * scenario.typeA.cost + 4 * scenario.typeB.cost = 330 ∧
  4 * scenario.typeA.cost + 3 * scenario.typeB.cost = 300 →
  minimumCost scenario = 18000 :=
sorry

#check minimum_cost_theorem

end NUMINAMATH_CALUDE_minimum_cost_theorem_l4122_412264


namespace NUMINAMATH_CALUDE_size_relationship_l4122_412288

theorem size_relationship (a b : ℚ) 
  (ha : a > 0) 
  (hb : b < 0) 
  (hab : |a| > |b|) : 
  -a < -b ∧ -b < b ∧ b < a := by
  sorry

end NUMINAMATH_CALUDE_size_relationship_l4122_412288


namespace NUMINAMATH_CALUDE_cash_me_problem_l4122_412217

def is_digit (n : ℕ) : Prop := n ≥ 0 ∧ n ≤ 9

def to_number (a b c d : ℕ) : ℕ := a * 1000 + b * 100 + c * 10 + d

theorem cash_me_problem :
  ¬∃ (C A S H M E O I D : ℕ),
    is_digit C ∧ is_digit A ∧ is_digit S ∧ is_digit H ∧
    is_digit M ∧ is_digit E ∧ is_digit O ∧ is_digit I ∧ is_digit D ∧
    C ≠ 0 ∧ M ≠ 0 ∧ O ≠ 0 ∧
    to_number C A S H + to_number M E 0 0 = to_number O S I D ∧
    to_number O S I D ≥ 1000 ∧ to_number O S I D < 10000 :=
by sorry

end NUMINAMATH_CALUDE_cash_me_problem_l4122_412217


namespace NUMINAMATH_CALUDE_expansion_coefficient_l4122_412256

/-- The coefficient of x^5y^2 in the expansion of (x^2 + x + y)^5 -/
def coefficient_x5y2 : ℕ :=
  -- We don't define the actual calculation here, just the type
  30

/-- The binomial coefficient -/
def binomial (n k : ℕ) : ℕ := sorry

theorem expansion_coefficient :
  coefficient_x5y2 = binomial 5 2 * binomial 3 1 := by sorry

end NUMINAMATH_CALUDE_expansion_coefficient_l4122_412256


namespace NUMINAMATH_CALUDE_alpha_value_l4122_412292

theorem alpha_value (β α : Real) 
  (h1 : 0 < β ∧ β < π/2)
  (h2 : 0 < α ∧ α < π/2)
  (h3 : Real.tan β = 1/2)
  (h4 : Real.tan (α - β) = 1/3) :
  α = π/4 := by
sorry

end NUMINAMATH_CALUDE_alpha_value_l4122_412292


namespace NUMINAMATH_CALUDE_worker_wage_problem_l4122_412242

theorem worker_wage_problem (ordinary_rate : ℝ) (overtime_rate : ℝ) (total_hours : ℕ) 
  (overtime_hours : ℕ) (total_earnings : ℝ) :
  overtime_rate = 0.90 →
  total_hours = 50 →
  overtime_hours = 8 →
  total_earnings = 32.40 →
  ordinary_rate * (total_hours - overtime_hours : ℝ) + overtime_rate * overtime_hours = total_earnings →
  ordinary_rate = 0.60 := by
sorry

end NUMINAMATH_CALUDE_worker_wage_problem_l4122_412242


namespace NUMINAMATH_CALUDE_largest_m_for_factorization_l4122_412299

def is_valid_factorization (m A B : ℤ) : Prop :=
  A * B = 90 ∧ 5 * B + A = m

theorem largest_m_for_factorization :
  (∃ (m : ℤ), ∀ (A B : ℤ), is_valid_factorization m A B →
    ∀ (m' : ℤ), (∃ (A' B' : ℤ), is_valid_factorization m' A' B') → m' ≤ m) ∧
  (∃ (A B : ℤ), is_valid_factorization 451 A B) :=
sorry

end NUMINAMATH_CALUDE_largest_m_for_factorization_l4122_412299


namespace NUMINAMATH_CALUDE_election_votes_theorem_l4122_412298

/-- Represents an election between two candidates -/
structure Election where
  total_votes : ℕ
  winner_votes : ℕ
  loser_votes : ℕ

/-- The conditions of the election problem -/
def election_conditions (e : Election) : Prop :=
  e.winner_votes + e.loser_votes = e.total_votes ∧
  e.winner_votes - e.loser_votes = (e.total_votes : ℚ) * (1 / 10) ∧
  (e.winner_votes - 1500) - (e.loser_votes + 1500) = -(e.total_votes : ℚ) * (1 / 10)

/-- The theorem stating that under the given conditions, the total votes is 15000 -/
theorem election_votes_theorem (e : Election) :
  election_conditions e → e.total_votes = 15000 := by
  sorry

end NUMINAMATH_CALUDE_election_votes_theorem_l4122_412298


namespace NUMINAMATH_CALUDE_intersection_complement_equality_l4122_412246

-- Define the sets
def U : Set ℝ := Set.univ
def A : Set ℝ := {x | ∃ y, y = Real.sqrt x}
def B : Set ℝ := {y | ∃ x, y = -x^2}

-- State the theorem
theorem intersection_complement_equality : A ∩ (U \ B) = {x | x > 0} := by sorry

end NUMINAMATH_CALUDE_intersection_complement_equality_l4122_412246


namespace NUMINAMATH_CALUDE_simplify_fraction_l4122_412222

theorem simplify_fraction (x y : ℚ) (hx : x = 3) (hy : y = 2) :
  12 * x * y^3 / (9 * x^2 * y^2) = 8 / 9 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l4122_412222


namespace NUMINAMATH_CALUDE_obtuse_triangle_proof_l4122_412209

theorem obtuse_triangle_proof (α : Real) (h1 : 0 < α ∧ α < π) 
  (h2 : Real.sin α + Real.cos α = 2/3) : π/2 < α := by
  sorry

end NUMINAMATH_CALUDE_obtuse_triangle_proof_l4122_412209


namespace NUMINAMATH_CALUDE_effective_distance_is_seven_l4122_412240

/-- Calculates the effective distance walked given a constant walking rate, wind resistance reduction, and walking duration. -/
def effective_distance_walked (rate : ℝ) (wind_resistance : ℝ) (duration : ℝ) : ℝ :=
  (rate - wind_resistance) * duration

/-- Proves that given the specified conditions, the effective distance walked is 7 miles. -/
theorem effective_distance_is_seven :
  let rate : ℝ := 4
  let wind_resistance : ℝ := 0.5
  let duration : ℝ := 2
  effective_distance_walked rate wind_resistance duration = 7 := by
sorry

end NUMINAMATH_CALUDE_effective_distance_is_seven_l4122_412240


namespace NUMINAMATH_CALUDE_cuboid_diagonal_range_l4122_412230

theorem cuboid_diagonal_range (d1 d2 x : ℝ) :
  d1 = 5 →
  d2 = 4 →
  (∃ a b c : ℝ, a > 0 ∧ b > 0 ∧ c > 0 ∧
    a^2 + b^2 = d1^2 ∧
    a^2 + c^2 = d2^2 ∧
    b^2 + c^2 = x^2) →
  3 < x ∧ x < Real.sqrt 41 := by
sorry

end NUMINAMATH_CALUDE_cuboid_diagonal_range_l4122_412230


namespace NUMINAMATH_CALUDE_triangle_area_from_altitudes_l4122_412277

/-- The area of a triangle given its three altitudes -/
theorem triangle_area_from_altitudes (h₁ h₂ h₃ : ℝ) (h₁_pos : h₁ > 0) (h₂_pos : h₂ > 0) (h₃_pos : h₃ > 0) :
  ∃ S : ℝ, S > 0 ∧ S = Real.sqrt ((1/h₁ + 1/h₂ + 1/h₃) * (-1/h₁ + 1/h₂ + 1/h₃) * (1/h₁ - 1/h₂ + 1/h₃) * (1/h₁ + 1/h₂ - 1/h₃)) :=
by sorry

end NUMINAMATH_CALUDE_triangle_area_from_altitudes_l4122_412277


namespace NUMINAMATH_CALUDE_decimal_to_binary_119_l4122_412229

/-- Converts a natural number to its binary representation as a list of bits -/
def toBinary (n : ℕ) : List Bool :=
  if n = 0 then [false] else
  let rec toBinaryAux (m : ℕ) : List Bool :=
    if m = 0 then [] else (m % 2 = 1) :: toBinaryAux (m / 2)
  toBinaryAux n

/-- The decimal number to be converted -/
def decimalNumber : ℕ := 119

/-- The expected binary representation -/
def expectedBinary : List Bool := [true, true, true, false, true, true, true]

/-- Theorem stating that the binary representation of 119 is [1,1,1,0,1,1,1] -/
theorem decimal_to_binary_119 : toBinary decimalNumber = expectedBinary := by
  sorry

end NUMINAMATH_CALUDE_decimal_to_binary_119_l4122_412229


namespace NUMINAMATH_CALUDE_vector_sum_l4122_412208

theorem vector_sum (x : ℝ) : 
  (⟨-3, 4, -2⟩ : ℝ × ℝ × ℝ) + (⟨5, -3, x⟩ : ℝ × ℝ × ℝ) = ⟨2, 1, x - 2⟩ := by
sorry

end NUMINAMATH_CALUDE_vector_sum_l4122_412208


namespace NUMINAMATH_CALUDE_a_gt_b_necessary_not_sufficient_l4122_412289

-- Define the curve C
def CurveC (a b x y : ℝ) : Prop := x^2 / a + y^2 / b = 1

-- Define what it means for C to be an ellipse with foci on the x-axis
def IsEllipseOnXAxis (a b : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ a > b ∧ ∀ x y, CurveC a b x y → x^2 + y^2 < a^2

-- Theorem stating that a > b is a necessary but not sufficient condition
theorem a_gt_b_necessary_not_sufficient :
  (∀ a b : ℝ, IsEllipseOnXAxis a b → a > b) ∧
  ¬(∀ a b : ℝ, a > b → IsEllipseOnXAxis a b) := by
  sorry

end NUMINAMATH_CALUDE_a_gt_b_necessary_not_sufficient_l4122_412289


namespace NUMINAMATH_CALUDE_collinear_points_triangle_inequality_l4122_412249

/-- Given five distinct collinear points A, B, C, D, E in order, with segment lengths AB = p, AC = q, AD = r, BE = s, DE = t,
    if AB and DE can be rotated about B and D respectively to form a triangle with positive area,
    then p < r/2 and s < t + p/2 must be true. -/
theorem collinear_points_triangle_inequality (p q r s t : ℝ) 
  (h_distinct : p > 0 ∧ q > p ∧ r > q ∧ s > 0 ∧ t > 0) 
  (h_triangle : p + s > r + t - s ∧ s + (r + t - s) > p ∧ p + (r + t - s) > s) :
  p < r / 2 ∧ s < t + p / 2 := by
  sorry

end NUMINAMATH_CALUDE_collinear_points_triangle_inequality_l4122_412249


namespace NUMINAMATH_CALUDE_light_travel_100_years_l4122_412233

/-- The distance light travels in one year in miles -/
def light_year_distance : ℝ := 5870000000000

/-- Theorem stating the distance light travels in 100 years -/
theorem light_travel_100_years :
  100 * light_year_distance = 587 * (10 ^ 12 : ℝ) :=
sorry

end NUMINAMATH_CALUDE_light_travel_100_years_l4122_412233


namespace NUMINAMATH_CALUDE_complex_equation_implies_xy_equals_one_l4122_412215

theorem complex_equation_implies_xy_equals_one (x y : ℝ) :
  (x + 1 : ℂ) + y * I = -I + 2 * x →
  x ^ y = 1 :=
by
  sorry

end NUMINAMATH_CALUDE_complex_equation_implies_xy_equals_one_l4122_412215


namespace NUMINAMATH_CALUDE_min_value_theorem_l4122_412257

theorem min_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) (h_sum : 2*a + b = 6) :
  (1/a + 2/b) ≥ 4/3 ∧ ∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ 2*a₀ + b₀ = 6 ∧ 1/a₀ + 2/b₀ = 4/3 := by
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l4122_412257


namespace NUMINAMATH_CALUDE_units_digit_of_sum_units_digit_of_42_4_plus_24_4_l4122_412214

theorem units_digit_of_sum (a b : ℕ) : (a^4 + b^4) % 10 = ((a^4 % 10) + (b^4 % 10)) % 10 := by sorry

theorem units_digit_of_42_4_plus_24_4 : (42^4 + 24^4) % 10 = 2 := by
  have h1 : 42^4 % 10 = 6 := by sorry
  have h2 : 24^4 % 10 = 6 := by sorry
  have h3 : (6 + 6) % 10 = 2 := by sorry
  
  calc
    (42^4 + 24^4) % 10 = ((42^4 % 10) + (24^4 % 10)) % 10 := by apply units_digit_of_sum
    _ = (6 + 6) % 10 := by rw [h1, h2]
    _ = 2 := by rw [h3]

end NUMINAMATH_CALUDE_units_digit_of_sum_units_digit_of_42_4_plus_24_4_l4122_412214


namespace NUMINAMATH_CALUDE_logarithm_system_solution_l4122_412216

theorem logarithm_system_solution :
  ∃ (x y z : ℝ),
    (x > 0 ∧ y > 0 ∧ z > 0) ∧
    (Real.log z / Real.log (2 * x) = 3) ∧
    (Real.log z / Real.log (5 * y) = 6) ∧
    (Real.log z / Real.log (x * y) = 2/3) ∧
    (x = 1 / (2 * Real.rpow 10 (1/3))) ∧
    (y = 1 / (5 * Real.rpow 10 (1/6))) ∧
    (z = 1/10) := by
  sorry

end NUMINAMATH_CALUDE_logarithm_system_solution_l4122_412216


namespace NUMINAMATH_CALUDE_centroid_dot_product_l4122_412241

/-- Triangle ABC with centroid G -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  G : ℝ × ℝ

/-- Vector from point P to point Q -/
def vector (P Q : ℝ × ℝ) : ℝ × ℝ := (Q.1 - P.1, Q.2 - P.2)

/-- Dot product of two 2D vectors -/
def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

/-- Squared distance between two points -/
def distance_squared (P Q : ℝ × ℝ) : ℝ := (Q.1 - P.1)^2 + (Q.2 - P.2)^2

theorem centroid_dot_product (t : Triangle) : 
  (distance_squared t.A t.B = 1) →
  (distance_squared t.B t.C = 2) →
  (distance_squared t.A t.C = 3) →
  (t.G = ((t.A.1 + t.B.1 + t.C.1) / 3, (t.A.2 + t.B.2 + t.C.2) / 3)) →
  (dot_product (vector t.A t.G) (vector t.A t.C) = 4/3) := by
  sorry

end NUMINAMATH_CALUDE_centroid_dot_product_l4122_412241


namespace NUMINAMATH_CALUDE_eraser_cost_proof_l4122_412244

/-- The cost of an eraser in dollars -/
def eraser_cost : ℚ := 2

/-- The cost of a pencil in dollars -/
def pencil_cost : ℚ := 4

/-- The number of pencils sold -/
def pencils_sold : ℕ := 20

/-- The total revenue in dollars -/
def total_revenue : ℚ := 80

/-- The ratio of erasers to pencils sold -/
def eraser_pencil_ratio : ℕ := 2

theorem eraser_cost_proof :
  eraser_cost = 2 ∧
  eraser_cost = pencil_cost / 2 ∧
  pencils_sold * pencil_cost = total_revenue ∧
  pencils_sold * eraser_pencil_ratio * eraser_cost = total_revenue / 2 :=
by sorry

end NUMINAMATH_CALUDE_eraser_cost_proof_l4122_412244


namespace NUMINAMATH_CALUDE_china_first_negative_numbers_l4122_412221

-- Define an enumeration for the countries
inductive Country
  | France
  | China
  | England
  | UnitedStates

-- Define a function that represents the property of being the first country to recognize and use negative numbers
def firstToUseNegativeNumbers : Country → Prop :=
  fun c => c = Country.China

-- Theorem statement
theorem china_first_negative_numbers :
  ∃ c : Country, firstToUseNegativeNumbers c ∧
  (c = Country.France ∨ c = Country.China ∨ c = Country.England ∨ c = Country.UnitedStates) :=
by
  sorry


end NUMINAMATH_CALUDE_china_first_negative_numbers_l4122_412221


namespace NUMINAMATH_CALUDE_natalia_cycling_distance_l4122_412266

/-- Represents the total distance cycled over four days given specific conditions --/
def total_distance (monday tuesday : ℕ) : ℕ :=
  let wednesday := tuesday / 2
  let thursday := monday + wednesday
  monday + tuesday + wednesday + thursday

/-- Theorem stating that given the specific conditions in the problem, 
    the total distance cycled is 180 km --/
theorem natalia_cycling_distance : total_distance 40 50 = 180 := by
  sorry

end NUMINAMATH_CALUDE_natalia_cycling_distance_l4122_412266


namespace NUMINAMATH_CALUDE_total_money_sum_l4122_412284

theorem total_money_sum (J : ℕ) : 
  (3 * J = 60) → 
  (J + 3 * J + (2 * J - 7) = 113) := by
  sorry

end NUMINAMATH_CALUDE_total_money_sum_l4122_412284


namespace NUMINAMATH_CALUDE_pepper_plants_died_l4122_412283

/-- Represents the garden with its plants and vegetables --/
structure Garden where
  tomato_plants : ℕ
  eggplant_plants : ℕ
  pepper_plants : ℕ
  dead_tomato_plants : ℕ
  dead_pepper_plants : ℕ
  vegetables_per_plant : ℕ
  total_vegetables : ℕ

/-- Theorem representing the problem and its solution --/
theorem pepper_plants_died (g : Garden) : g.dead_pepper_plants = 1 :=
  by
  have h1 : g.tomato_plants = 6 := by sorry
  have h2 : g.eggplant_plants = 2 := by sorry
  have h3 : g.pepper_plants = 4 := by sorry
  have h4 : g.dead_tomato_plants = g.tomato_plants / 2 := by sorry
  have h5 : g.vegetables_per_plant = 7 := by sorry
  have h6 : g.total_vegetables = 56 := by sorry
  
  sorry

end NUMINAMATH_CALUDE_pepper_plants_died_l4122_412283


namespace NUMINAMATH_CALUDE_triangle_BC_proof_l4122_412236

def triangle_BC (A B C : ℝ) (tanA : ℝ) (AB : ℝ) : Prop :=
  let angleB := Real.pi / 2
  let BC := ((AB ^ 2) + (tanA * AB) ^ 2).sqrt
  angleB = Real.pi / 2 ∧ 
  tanA = 3 / 7 ∧ 
  AB = 14 → 
  BC = 2 * Real.sqrt 58

theorem triangle_BC_proof : triangle_BC Real.pi Real.pi Real.pi (3/7) 14 := by
  sorry

end NUMINAMATH_CALUDE_triangle_BC_proof_l4122_412236


namespace NUMINAMATH_CALUDE_binomial_15_choose_3_l4122_412281

theorem binomial_15_choose_3 : Nat.choose 15 3 = 455 := by
  sorry

end NUMINAMATH_CALUDE_binomial_15_choose_3_l4122_412281


namespace NUMINAMATH_CALUDE_distribute_five_to_three_l4122_412200

/-- The number of ways to distribute n students to k universities, 
    with each university admitting at least one student -/
def distribute_students (n : ℕ) (k : ℕ) : ℕ :=
  sorry

/-- Theorem: Distributing 5 students to 3 universities results in 150 different methods -/
theorem distribute_five_to_three : distribute_students 5 3 = 150 := by
  sorry

end NUMINAMATH_CALUDE_distribute_five_to_three_l4122_412200


namespace NUMINAMATH_CALUDE_difference_of_squares_l4122_412268

theorem difference_of_squares (a b : ℝ) : a^2 - b^2 = (a + b) * (a - b) := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_l4122_412268


namespace NUMINAMATH_CALUDE_cost_of_dozen_pens_l4122_412250

/-- Given the cost of 3 pens and 5 pencils is Rs. 100, and the cost ratio of one pen to one pencil
    is 5:1, prove that the cost of one dozen pens is Rs. 300. -/
theorem cost_of_dozen_pens (pen_cost pencil_cost : ℕ) : 
  (3 * pen_cost + 5 * pencil_cost = 100) →
  (pen_cost = 5 * pencil_cost) →
  (12 * pen_cost = 300) :=
by sorry

end NUMINAMATH_CALUDE_cost_of_dozen_pens_l4122_412250


namespace NUMINAMATH_CALUDE_car_tire_rotation_theorem_l4122_412254

/-- Calculates the number of miles each tire is used given the total number of tires, 
    tires used simultaneously, and total miles traveled. -/
def miles_per_tire (total_tires : ℕ) (tires_used : ℕ) (total_miles : ℕ) : ℕ :=
  (total_miles * tires_used) / total_tires

/-- Proves that for a car with 5 tires, where 4 are used simultaneously over 30,000 miles,
    each tire is used for 24,000 miles. -/
theorem car_tire_rotation_theorem :
  miles_per_tire 5 4 30000 = 24000 := by
  sorry

#eval miles_per_tire 5 4 30000

end NUMINAMATH_CALUDE_car_tire_rotation_theorem_l4122_412254


namespace NUMINAMATH_CALUDE_expression_one_proof_l4122_412294

theorem expression_one_proof : 1 + (-2) + |(-2) - 3| - 5 = -1 := by
  sorry

end NUMINAMATH_CALUDE_expression_one_proof_l4122_412294


namespace NUMINAMATH_CALUDE_arithmetic_square_root_of_one_fourth_l4122_412218

theorem arithmetic_square_root_of_one_fourth (x : ℝ) : x = Real.sqrt (1/4) → x = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_square_root_of_one_fourth_l4122_412218


namespace NUMINAMATH_CALUDE_inverse_exists_iff_a_eq_zero_l4122_412285

-- Define the function f(x) = (x - a)|x|
def f (a : ℝ) (x : ℝ) : ℝ := (x - a) * abs x

-- State the theorem
theorem inverse_exists_iff_a_eq_zero (a : ℝ) :
  Function.Injective (f a) ↔ a = 0 := by sorry

end NUMINAMATH_CALUDE_inverse_exists_iff_a_eq_zero_l4122_412285


namespace NUMINAMATH_CALUDE_talia_father_age_l4122_412205

/-- Represents the ages of Talia, her mother, and her father -/
structure FamilyAges where
  talia : ℕ
  mother : ℕ
  father : ℕ

/-- Conditions for the family ages problem -/
def FamilyAgeProblem (ages : FamilyAges) : Prop :=
  (ages.talia + 7 = 20) ∧
  (ages.mother = 3 * ages.talia) ∧
  (ages.father + 3 = ages.mother)

/-- Theorem stating that given the conditions, Talia's father is 36 years old -/
theorem talia_father_age (ages : FamilyAges) :
  FamilyAgeProblem ages → ages.father = 36 := by
  sorry

end NUMINAMATH_CALUDE_talia_father_age_l4122_412205


namespace NUMINAMATH_CALUDE_max_value_complex_l4122_412255

theorem max_value_complex (z : ℂ) (h : Complex.abs z = 2) :
  Complex.abs ((z - 2)^2 * (z + 2)) ≤ 16 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_max_value_complex_l4122_412255


namespace NUMINAMATH_CALUDE_cds_on_shelf_l4122_412297

/-- The number of CDs that a single rack can hold -/
def cds_per_rack : ℕ := 8

/-- The number of racks that can fit on a shelf -/
def racks_per_shelf : ℕ := 4

/-- Theorem stating the total number of CDs that can fit on a shelf -/
theorem cds_on_shelf : cds_per_rack * racks_per_shelf = 32 := by
  sorry

end NUMINAMATH_CALUDE_cds_on_shelf_l4122_412297


namespace NUMINAMATH_CALUDE_circle_intersection_l4122_412295

def circle1 (x y : ℝ) : Prop := (x - 2)^2 + (y - 10)^2 = 50

def circle2 (x y : ℝ) : Prop := x^2 + y^2 + 2*(x - y) - 18 = 0

theorem circle_intersection :
  ∃ (x1 y1 x2 y2 : ℝ),
    (circle1 x1 y1 ∧ circle2 x1 y1) ∧
    (circle1 x2 y2 ∧ circle2 x2 y2) ∧
    (x1 = 3 ∧ y1 = 3) ∧
    (x2 = -3 ∧ y2 = 5) :=
  sorry

end NUMINAMATH_CALUDE_circle_intersection_l4122_412295


namespace NUMINAMATH_CALUDE_faye_crayons_count_l4122_412267

/-- Given that Faye arranges her crayons in 16 rows with 6 crayons per row,
    prove that she has 96 crayons in total. -/
theorem faye_crayons_count : 
  let rows : ℕ := 16
  let crayons_per_row : ℕ := 6
  rows * crayons_per_row = 96 := by
sorry

end NUMINAMATH_CALUDE_faye_crayons_count_l4122_412267


namespace NUMINAMATH_CALUDE_absolute_difference_of_product_and_sum_l4122_412243

theorem absolute_difference_of_product_and_sum (p q : ℝ) 
  (h1 : p * q = 6) 
  (h2 : p + q = 7) : 
  |p - q| = 5 := by
sorry

end NUMINAMATH_CALUDE_absolute_difference_of_product_and_sum_l4122_412243


namespace NUMINAMATH_CALUDE_specific_plot_fencing_cost_l4122_412258

/-- A rectangular plot with specific dimensions and fencing cost. -/
structure RectangularPlot where
  width : ℝ
  length : ℝ
  fencing_cost_per_meter : ℝ
  perimeter : ℝ

/-- The total cost of fencing for a rectangular plot. -/
def total_fencing_cost (plot : RectangularPlot) : ℝ :=
  plot.perimeter * plot.fencing_cost_per_meter

/-- Theorem stating the total fencing cost for a specific rectangular plot. -/
theorem specific_plot_fencing_cost :
  ∃ (plot : RectangularPlot),
    plot.length = plot.width + 10 ∧
    plot.perimeter = 220 ∧
    plot.fencing_cost_per_meter = 6.5 ∧
    total_fencing_cost plot = 1430 := by
  sorry

end NUMINAMATH_CALUDE_specific_plot_fencing_cost_l4122_412258


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l4122_412252

/-- An arithmetic sequence with its sum function -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  S : ℕ → ℝ  -- The sum function
  sum_formula : ∀ n, S n = n * (a 1 + a n) / 2
  is_arithmetic : ∀ n, a (n + 1) - a n = a 2 - a 1

/-- Theorem: For an arithmetic sequence, if S_9 = 54 and S_8 - S_5 = 30, then S_11 = 88 -/
theorem arithmetic_sequence_sum (seq : ArithmeticSequence)
    (h1 : seq.S 9 = 54)
    (h2 : seq.S 8 - seq.S 5 = 30) :
    seq.S 11 = 88 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l4122_412252


namespace NUMINAMATH_CALUDE_s_scale_indeterminate_l4122_412211

/-- Represents a linear relationship between two measurement scales -/
structure ScaleRelation where
  /-- Slope of the linear relationship -/
  a : ℝ
  /-- Y-intercept of the linear relationship -/
  b : ℝ

/-- Converts a p-scale measurement to an s-scale measurement -/
def toSScale (relation : ScaleRelation) (p : ℝ) : ℝ :=
  relation.a * p + relation.b

/-- Theorem stating that the s-scale measurement for p=24 cannot be uniquely determined -/
theorem s_scale_indeterminate (known_p : ℝ) (known_s : ℝ) (target_p : ℝ) 
    (h1 : known_p = 6) (h2 : known_s = 30) (h3 : target_p = 24) :
    ∃ (r1 r2 : ScaleRelation), r1 ≠ r2 ∧ 
    toSScale r1 known_p = known_s ∧
    toSScale r2 known_p = known_s ∧
    toSScale r1 target_p ≠ toSScale r2 target_p :=
  sorry

end NUMINAMATH_CALUDE_s_scale_indeterminate_l4122_412211


namespace NUMINAMATH_CALUDE_smallest_prime_with_composite_reverse_l4122_412262

/-- A function that reverses the digits of a two-digit number -/
def reverseDigits (n : ℕ) : ℕ :=
  (n % 10) * 10 + (n / 10)

/-- A function that checks if a number is prime -/
def isPrime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, 1 < m → m < n → ¬(n % m = 0)

/-- A function that checks if a number is composite -/
def isComposite (n : ℕ) : Prop :=
  n > 1 ∧ ∃ m : ℕ, 1 < m ∧ m < n ∧ n % m = 0

/-- The main theorem -/
theorem smallest_prime_with_composite_reverse :
  ∃ (p : ℕ),
    isPrime p ∧
    p ≥ 10 ∧
    p < 100 ∧
    p / 10 = 3 ∧
    isComposite (reverseDigits p) ∧
    (∀ q : ℕ, isPrime q → q ≥ 10 → q < 100 → q / 10 = 3 →
      isComposite (reverseDigits q) → p ≤ q) ∧
    p = 23 :=
  sorry

end NUMINAMATH_CALUDE_smallest_prime_with_composite_reverse_l4122_412262


namespace NUMINAMATH_CALUDE_combined_net_earnings_proof_l4122_412287

def connor_hourly_rate : ℝ := 7.20
def connor_hours : ℝ := 8
def emily_hourly_rate : ℝ := 2 * connor_hourly_rate
def emily_hours : ℝ := 10
def sarah_hourly_rate : ℝ := 5 * connor_hourly_rate + connor_hourly_rate
def sarah_hours : ℝ := connor_hours

def connor_deduction_rate : ℝ := 0.05
def emily_deduction_rate : ℝ := 0.08
def sarah_deduction_rate : ℝ := 0.10

def connor_gross_earnings : ℝ := connor_hourly_rate * connor_hours
def emily_gross_earnings : ℝ := emily_hourly_rate * emily_hours
def sarah_gross_earnings : ℝ := sarah_hourly_rate * sarah_hours

def connor_net_earnings : ℝ := connor_gross_earnings * (1 - connor_deduction_rate)
def emily_net_earnings : ℝ := emily_gross_earnings * (1 - emily_deduction_rate)
def sarah_net_earnings : ℝ := sarah_gross_earnings * (1 - sarah_deduction_rate)

def combined_net_earnings : ℝ := connor_net_earnings + emily_net_earnings + sarah_net_earnings

theorem combined_net_earnings_proof : combined_net_earnings = 498.24 := by
  sorry

end NUMINAMATH_CALUDE_combined_net_earnings_proof_l4122_412287


namespace NUMINAMATH_CALUDE_symmetry_properties_l4122_412279

/-- Two rational numbers are symmetric about a point with a given symmetric radius. -/
def symmetric (m n p r : ℚ) : Prop :=
  m ≠ n ∧ m ≠ p ∧ n ≠ p ∧ |m - p| = r ∧ |n - p| = r

theorem symmetry_properties :
  (∃ x r : ℚ, symmetric 3 x 1 r ∧ x = -1 ∧ r = 2) ∧
  (∃ a b r : ℚ, symmetric a b 2 r ∧ |a| = 2 * |b| ∧ (r = 2/3 ∨ r = 6)) :=
by sorry

end NUMINAMATH_CALUDE_symmetry_properties_l4122_412279


namespace NUMINAMATH_CALUDE_green_marble_fraction_l4122_412282

theorem green_marble_fraction (total : ℝ) (h1 : total > 0) : 
  let initial_green := (1/4) * total
  let initial_yellow := total - initial_green
  let new_green := 3 * initial_green
  let new_total := new_green + initial_yellow
  new_green / new_total = 1/2 := by sorry

end NUMINAMATH_CALUDE_green_marble_fraction_l4122_412282


namespace NUMINAMATH_CALUDE_polynomial_remainder_l4122_412237

theorem polynomial_remainder (x : ℝ) : 
  (x^4 - 4*x^2 + 7*x - 8) % (x - 3) = 58 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_remainder_l4122_412237


namespace NUMINAMATH_CALUDE_ball_count_l4122_412231

theorem ball_count (white green yellow red purple : ℕ)
  (h1 : white = 22)
  (h2 : green = 18)
  (h3 : yellow = 2)
  (h4 : red = 15)
  (h5 : purple = 3)
  (h6 : (white + green + yellow : ℚ) / (white + green + yellow + red + purple) = 7/10) :
  white + green + yellow + red + purple = 60 := by
  sorry

end NUMINAMATH_CALUDE_ball_count_l4122_412231


namespace NUMINAMATH_CALUDE_probability_multiple_6_or_8_l4122_412203

def is_multiple_of_6_or_8 (n : ℕ) : Bool :=
  n % 6 = 0 ∨ n % 8 = 0

def count_multiples (n : ℕ) : ℕ :=
  (List.range n).filter is_multiple_of_6_or_8 |>.length

theorem probability_multiple_6_or_8 :
  (count_multiples 60 : ℚ) / 60 = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_probability_multiple_6_or_8_l4122_412203


namespace NUMINAMATH_CALUDE_trains_meet_at_1108_l4122_412290

/-- Represents a train with its departure time and speed -/
structure Train where
  departureTime : Nat  -- minutes since midnight
  speed : Nat          -- km/h

/-- Represents a station with its distance from Station A -/
structure Station where
  distanceFromA : Nat  -- km

def stationA : Station := { distanceFromA := 0 }
def stationB : Station := { distanceFromA := 300 }
def stationC : Station := { distanceFromA := 150 }

def trainA : Train := { departureTime := 9 * 60 + 45, speed := 60 }
def trainB : Train := { departureTime := 10 * 60, speed := 80 }

def stopTime : Nat := 10  -- minutes

/-- Calculates the meeting time of two trains given the conditions -/
def calculateMeetingTime (trainA trainB : Train) (stationA stationB stationC : Station) (stopTime : Nat) : Nat :=
  sorry  -- Proof to be implemented

theorem trains_meet_at_1108 :
  calculateMeetingTime trainA trainB stationA stationB stationC stopTime = 11 * 60 + 8 := by
  sorry  -- Proof to be implemented

end NUMINAMATH_CALUDE_trains_meet_at_1108_l4122_412290
