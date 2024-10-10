import Mathlib

namespace total_legs_count_l1127_112786

theorem total_legs_count (total_tables : ℕ) (four_leg_tables : ℕ) : 
  total_tables = 36 → four_leg_tables = 16 → 
  (∃ (three_leg_tables : ℕ), 
    three_leg_tables + four_leg_tables = total_tables ∧
    3 * three_leg_tables + 4 * four_leg_tables = 124) := by
  sorry

end total_legs_count_l1127_112786


namespace viktor_tally_theorem_l1127_112713

/-- Represents Viktor's tally system -/
structure TallySystem where
  x_value : ℕ  -- number of rallies represented by an X
  o_value : ℕ  -- number of rallies represented by an O

/-- Represents the final tally -/
structure FinalTally where
  o_count : ℕ  -- number of O's in the tally
  x_count : ℕ  -- number of X's in the tally

/-- Calculates the range of possible rallies given a tally system and final tally -/
def rally_range (system : TallySystem) (tally : FinalTally) : 
  {min_rallies : ℕ // ∃ max_rallies : ℕ, min_rallies ≤ max_rallies ∧ max_rallies ≤ min_rallies + system.x_value - 1} :=
sorry

theorem viktor_tally_theorem (system : TallySystem) (tally : FinalTally) :
  system.x_value = 10 ∧ 
  system.o_value = 100 ∧ 
  tally.o_count = 3 ∧ 
  tally.x_count = 7 →
  ∃ (range : {min_rallies : ℕ // ∃ max_rallies : ℕ, min_rallies ≤ max_rallies ∧ max_rallies ≤ min_rallies + system.x_value - 1}),
    range = rally_range system tally ∧
    range.val = 370 ∧
    (∃ max_rallies : ℕ, range.property.choose = 379) :=
sorry

end viktor_tally_theorem_l1127_112713


namespace abc_inequality_l1127_112706

theorem abc_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
  abc ≤ 1/9 ∧ a/(b+c) + b/(a+c) + c/(a+b) ≤ 1/(2 * Real.sqrt (abc)) := by
  sorry

end abc_inequality_l1127_112706


namespace complex_sum_reciprocal_magnitude_l1127_112726

theorem complex_sum_reciprocal_magnitude (z w : ℂ) :
  Complex.abs z = 2 →
  Complex.abs w = 4 →
  Complex.abs (z + w) = 3 →
  ∃ θ : ℝ, θ = Real.pi / 3 ∧ z * Complex.exp (Complex.I * θ) = w →
  Complex.abs (1 / z + 1 / w) = 3 / 8 := by
sorry

end complex_sum_reciprocal_magnitude_l1127_112726


namespace sequence_ratio_implies_half_l1127_112765

/-- Represents a positive rational number less than 1 -/
structure PositiveRationalLessThanOne where
  val : ℚ
  pos : 0 < val
  lt_one : val < 1

/-- Given conditions for the sequences and their relationship -/
structure SequenceConditions where
  d : ℚ
  d_nonzero : d ≠ 0
  q : PositiveRationalLessThanOne
  a : ℕ → ℚ
  b : ℕ → ℚ
  a_def : ∀ n, a n = d * n
  b_def : ∀ n, b n = d^2 * q.val^(n-1)
  sum_ratio_integer : ∃ k : ℕ+, (a 1^2 + a 2^2 + a 3^2) / (b 1 + b 2 + b 3) = k

/-- The main theorem stating that under the given conditions, q must equal 1/2 -/
theorem sequence_ratio_implies_half (cond : SequenceConditions) : cond.q.val = 1/2 := by
  sorry

end sequence_ratio_implies_half_l1127_112765


namespace two_possible_values_for_D_l1127_112788

def is_digit (n : ℕ) : Prop := n ≥ 0 ∧ n ≤ 9

def distinct_digits (A B C D E : ℕ) : Prop :=
  is_digit A ∧ is_digit B ∧ is_digit C ∧ is_digit D ∧ is_digit E ∧
  A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ A ≠ E ∧
  B ≠ C ∧ B ≠ D ∧ B ≠ E ∧
  C ≠ D ∧ C ≠ E ∧
  D ≠ E

def addition_equation (A B C D E : ℕ) : Prop :=
  (A * 10000 + B * 1000 + C * 100 + D * 10 + B) +
  (B * 10000 + C * 1000 + A * 100 + D * 10 + E) =
  (E * 10000 + D * 1000 + D * 100 + E * 10 + E)

theorem two_possible_values_for_D :
  ∃ (D₁ D₂ : ℕ), D₁ ≠ D₂ ∧
  (∀ (A B C D E : ℕ), distinct_digits A B C D E → addition_equation A B C D E →
    D = D₁ ∨ D = D₂) ∧
  (∀ (A B C E : ℕ), ∃ (D : ℕ), distinct_digits A B C D E ∧ addition_equation A B C D E) :=
by sorry

end two_possible_values_for_D_l1127_112788


namespace combined_average_correct_l1127_112741

-- Define the percentages for each city
def springfield : Fin 4 → ℚ
  | 0 => 12
  | 1 => 18
  | 2 => 25
  | 3 => 40

def shelbyville : Fin 4 → ℚ
  | 0 => 10
  | 1 => 15
  | 2 => 23
  | 3 => 35

-- Define the years
def years : Fin 4 → ℕ
  | 0 => 1990
  | 1 => 2000
  | 2 => 2010
  | 3 => 2020

-- Define the combined average function
def combinedAverage (i : Fin 4) : ℚ :=
  (springfield i + shelbyville i) / 2

-- Theorem statement
theorem combined_average_correct :
  (combinedAverage 0 = 11) ∧
  (combinedAverage 1 = 33/2) ∧
  (combinedAverage 2 = 24) ∧
  (combinedAverage 3 = 75/2) := by
  sorry

end combined_average_correct_l1127_112741


namespace pet_store_hamsters_l1127_112779

theorem pet_store_hamsters (rabbit_count : ℕ) (rabbit_ratio : ℕ) (hamster_ratio : ℕ) : 
  rabbit_count = 18 → 
  rabbit_ratio = 3 → 
  hamster_ratio = 4 → 
  (rabbit_count / rabbit_ratio) * hamster_ratio = 24 := by
sorry

end pet_store_hamsters_l1127_112779


namespace x_squared_plus_reciprocal_l1127_112718

theorem x_squared_plus_reciprocal (x : ℝ) (h : x ≠ 0) :
  x^4 + 1/x^4 = 47 → x^2 + 1/x^2 = 7 := by
sorry

end x_squared_plus_reciprocal_l1127_112718


namespace negation_of_existence_negation_of_squared_plus_one_less_than_zero_l1127_112716

theorem negation_of_existence (P : ℝ → Prop) :
  (¬ ∃ x : ℝ, P x) ↔ (∀ x : ℝ, ¬ P x) :=
by
  sorry

theorem negation_of_squared_plus_one_less_than_zero :
  (¬ ∃ x : ℝ, x^2 + 1 < 0) ↔ (∀ x : ℝ, x^2 + 1 ≥ 0) :=
by
  sorry

end negation_of_existence_negation_of_squared_plus_one_less_than_zero_l1127_112716


namespace work_completion_time_l1127_112753

/-- The time it takes for A to complete the entire work -/
def a_complete_time : ℝ := 21

/-- The time it takes for B to complete the entire work -/
def b_complete_time : ℝ := 15

/-- The number of days B worked before leaving -/
def b_worked_days : ℝ := 10

/-- The time it takes for A to complete the remaining work after B leaves -/
def a_remaining_time : ℝ := 7

theorem work_completion_time :
  a_complete_time = 21 →
  b_complete_time = 15 →
  b_worked_days = 10 →
  a_remaining_time = 7 := by
  sorry

end work_completion_time_l1127_112753


namespace square_land_side_length_l1127_112752

theorem square_land_side_length (area : ℝ) (side : ℝ) :
  area = 400 →
  side * side = area →
  side = 20 := by
sorry

end square_land_side_length_l1127_112752


namespace inheritance_calculation_l1127_112705

theorem inheritance_calculation (x : ℝ) : 
  (0.25 * x + 0.15 * (x - 0.25 * x) = 20000) → x = 55172 := by
  sorry

end inheritance_calculation_l1127_112705


namespace sphere_only_circular_all_views_l1127_112739

-- Define the geometric shapes
inductive Shape
| Cuboid
| Cylinder
| Cone
| Sphere

-- Define the views
inductive View
| Front
| Left
| Top

-- Function to determine if a view of a shape is circular
def isCircularView (s : Shape) (v : View) : Prop :=
  match s, v with
  | Shape.Sphere, _ => True
  | Shape.Cylinder, View.Top => True
  | Shape.Cone, View.Top => True
  | _, _ => False

-- Theorem stating that only the Sphere has circular views in all three perspectives
theorem sphere_only_circular_all_views :
  ∀ s : Shape, (∀ v : View, isCircularView s v) ↔ s = Shape.Sphere := by
  sorry

end sphere_only_circular_all_views_l1127_112739


namespace B_equals_l1127_112773

-- Define the universal set U
def U : Set Nat := {1, 2, 3, 4, 5, 6, 7}

-- Define sets A and B
variable (A B : Set Nat)

-- State the conditions
axiom union_eq : A ∪ B = U
axiom intersection_eq : A ∩ (U \ B) = {2, 4, 6}

-- Theorem to prove
theorem B_equals : B = {1, 3, 5, 7} := by sorry

end B_equals_l1127_112773


namespace class_vision_median_l1127_112743

/-- Represents the vision data for a class of students -/
structure VisionData where
  values : List Float
  frequencies : List Nat
  total_students : Nat

/-- Calculates the median of the vision data -/
def median (data : VisionData) : Float :=
  sorry

/-- The specific vision data for the class -/
def class_vision_data : VisionData :=
  { values := [4.0, 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8, 4.9, 5.0],
    frequencies := [1, 2, 6, 3, 3, 4, 1, 2, 5, 7, 5],
    total_students := 39 }

/-- Theorem stating that the median of the class vision data is 4.6 -/
theorem class_vision_median :
  median class_vision_data = 4.6 := by
  sorry

end class_vision_median_l1127_112743


namespace condition_for_proposition_l1127_112707

def A : Set ℝ := {x | 1 ≤ x ∧ x ≤ 2}

theorem condition_for_proposition (a : ℝ) :
  (∀ x ∈ A, x^2 - a ≤ 0) ↔ a ≥ 4 := by sorry

end condition_for_proposition_l1127_112707


namespace cos_pi_third_plus_alpha_l1127_112780

theorem cos_pi_third_plus_alpha (α : ℝ) 
  (h : Real.sin (π / 6 - α) = 1 / 3) : 
  Real.cos (π / 3 + α) = 1 / 3 := by
  sorry

end cos_pi_third_plus_alpha_l1127_112780


namespace trajectory_and_circle_properties_l1127_112781

-- Define the vectors a and b
def a (m x y : ℝ) : ℝ × ℝ := (m * x, y + 1)
def b (x y : ℝ) : ℝ × ℝ := (x, y - 1)

-- Define the dot product of two 2D vectors
def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

-- Define the perpendicularity condition
def perpendicular (m x y : ℝ) : Prop := dot_product (a m x y) (b x y) = 0

-- Define the equation of trajectory E
def trajectory_equation (m x y : ℝ) : Prop := m * x^2 + y^2 = 1

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop := x^2 + y^2 = 4/5

-- Define a tangent line to the circle
def tangent_line (k t x y : ℝ) : Prop := y = k * x + t

-- Define the perpendicularity condition for OA and OB
def OA_perp_OB (x1 y1 x2 y2 : ℝ) : Prop := x1 * x2 + y1 * y2 = 0

-- Main theorem
theorem trajectory_and_circle_properties (m : ℝ) :
  (∀ x y : ℝ, perpendicular m x y → trajectory_equation m x y) ∧
  (m = 1/4 →
    ∃ k t x1 y1 x2 y2 : ℝ,
      tangent_line k t x1 y1 ∧
      tangent_line k t x2 y2 ∧
      trajectory_equation m x1 y1 ∧
      trajectory_equation m x2 y2 ∧
      circle_equation x1 y1 ∧
      circle_equation x2 y2 ∧
      OA_perp_OB x1 y1 x2 y2) :=
sorry

end trajectory_and_circle_properties_l1127_112781


namespace scientific_notation_of_10374_billion_l1127_112754

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  is_valid : 1 ≤ coefficient ∧ coefficient < 10

/-- Converts a real number to scientific notation with a specified number of significant figures -/
def toScientificNotation (x : ℝ) (sigFigs : ℕ) : ScientificNotation :=
  sorry

/-- The value to be converted (10,374 billion yuan) -/
def originalValue : ℝ := 10374 * 1000000000

/-- The number of significant figures to retain -/
def sigFigures : ℕ := 3

theorem scientific_notation_of_10374_billion :
  toScientificNotation originalValue sigFigures =
    ScientificNotation.mk 1.037 13 (by norm_num) :=
  sorry

end scientific_notation_of_10374_billion_l1127_112754


namespace a_zero_necessary_not_sufficient_l1127_112762

def is_pure_imaginary (z : ℂ) : Prop := z.re = 0

theorem a_zero_necessary_not_sufficient :
  ∃ (a b : ℝ), (is_pure_imaginary (a + b * I) → a = 0) ∧
  ¬(a = 0 → is_pure_imaginary (a + b * I)) :=
sorry

end a_zero_necessary_not_sufficient_l1127_112762


namespace arithmetic_square_root_of_16_l1127_112704

theorem arithmetic_square_root_of_16 : 
  ∃ (x : ℝ), x ≥ 0 ∧ x ^ 2 = 16 ∧ x = 4 := by
  sorry

end arithmetic_square_root_of_16_l1127_112704


namespace cola_cost_l1127_112772

/-- Proves that the cost of each cola bottle is $2 given the conditions of Wilson's purchase. -/
theorem cola_cost (hamburger_price : ℚ) (num_hamburgers : ℕ) (num_cola : ℕ) (discount : ℚ) (total_paid : ℚ) :
  hamburger_price = 5 →
  num_hamburgers = 2 →
  num_cola = 3 →
  discount = 4 →
  total_paid = 12 →
  (total_paid + discount - num_hamburgers * hamburger_price) / num_cola = 2 := by
  sorry

end cola_cost_l1127_112772


namespace y_divisibility_l1127_112766

theorem y_divisibility : ∃ k : ℕ, 
  (96 + 144 + 200 + 300 + 600 + 720 + 4800 : ℕ) = 4 * k ∧ 
  (∃ m : ℕ, (96 + 144 + 200 + 300 + 600 + 720 + 4800 : ℕ) ≠ 8 * m ∨ 
            (96 + 144 + 200 + 300 + 600 + 720 + 4800 : ℕ) ≠ 16 * m ∨ 
            (96 + 144 + 200 + 300 + 600 + 720 + 4800 : ℕ) ≠ 32 * m) :=
by sorry

end y_divisibility_l1127_112766


namespace triangle_inequality_l1127_112756

/-- Theorem: In a triangle with two sides of lengths 3 and 8, the third side is between 5 and 11 -/
theorem triangle_inequality (a b c : ℝ) : a = 3 ∧ b = 8 → 5 < c ∧ c < 11 := by
  sorry

end triangle_inequality_l1127_112756


namespace notebook_payment_l1127_112708

/-- The value of a nickel in cents -/
def nickel_value : ℕ := 5

/-- The cost of the notebook in cents -/
def notebook_cost : ℕ := 130

/-- The number of nickels needed to pay for the notebook -/
def nickels_needed : ℕ := notebook_cost / nickel_value

theorem notebook_payment :
  nickels_needed = 26 := by sorry

end notebook_payment_l1127_112708


namespace winter_sales_calculation_l1127_112723

/-- Represents the sales of pizzas in millions for each season -/
structure SeasonalSales where
  spring : ℝ
  summer : ℝ
  fall : ℝ
  winter : ℝ

/-- Calculates the total annual sales given the seasonal sales -/
def totalAnnualSales (sales : SeasonalSales) : ℝ :=
  sales.spring + sales.summer + sales.fall + sales.winter

/-- Theorem: Given the conditions, the number of pizzas sold in winter is 6.6 million -/
theorem winter_sales_calculation (sales : SeasonalSales)
    (h1 : sales.fall = 0.2 * totalAnnualSales sales)
    (h2 : sales.winter = 1.1 * sales.summer)
    (h3 : sales.spring = 5)
    (h4 : sales.summer = 6) :
    sales.winter = 6.6 := by
  sorry

#check winter_sales_calculation

end winter_sales_calculation_l1127_112723


namespace exists_factorial_starting_with_2005_l1127_112745

theorem exists_factorial_starting_with_2005 : 
  ∃ (n : ℕ+), ∃ (k : ℕ), 2005 * 10^k ≤ n.val.factorial ∧ n.val.factorial < 2006 * 10^k :=
sorry

end exists_factorial_starting_with_2005_l1127_112745


namespace largest_common_measure_l1127_112731

theorem largest_common_measure (segment1 segment2 : ℕ) 
  (h1 : segment1 = 15) (h2 : segment2 = 12) : 
  ∃ (m : ℕ), m > 0 ∧ m ∣ segment1 ∧ m ∣ segment2 ∧ 
  ∀ (n : ℕ), n > m → (n ∣ segment1 ∧ n ∣ segment2) → False :=
by sorry

end largest_common_measure_l1127_112731


namespace investment_solution_l1127_112749

def investment_problem (x : ℝ) : Prop :=
  let total_investment : ℝ := 1500
  let rate1 : ℝ := 1.04  -- 4% annual compound interest
  let rate2 : ℝ := 1.06  -- 6% annual compound interest
  let total_after_year : ℝ := 1590
  (x * rate1 + (total_investment - x) * rate2 = total_after_year) ∧
  (0 ≤ x) ∧ (x ≤ total_investment)

theorem investment_solution :
  ∃! x : ℝ, investment_problem x ∧ x = 0 :=
sorry

end investment_solution_l1127_112749


namespace max_intersections_count_l1127_112748

/-- The number of points on the x-axis segment -/
def n : ℕ := 15

/-- The number of points on the y-axis segment -/
def m : ℕ := 10

/-- The maximum number of intersection points -/
def max_intersections : ℕ := n.choose 2 * m.choose 2

/-- Theorem stating the maximum number of intersection points -/
theorem max_intersections_count :
  max_intersections = 4725 :=
sorry

end max_intersections_count_l1127_112748


namespace problem_l1127_112714

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then 2^x + 1 else x^2 + a*x

theorem problem (a : ℝ) : f a (f a 0) = 4*a → a = 2 := by
  sorry

end problem_l1127_112714


namespace books_in_boxes_l1127_112725

/-- The number of ways to place n different objects into k different boxes -/
def arrangements (n k : ℕ) : ℕ := k^n

/-- There are 6 different books -/
def num_books : ℕ := 6

/-- There are 5 different boxes -/
def num_boxes : ℕ := 5

/-- Theorem: The number of ways to place 6 different books into 5 different boxes is 5^6 -/
theorem books_in_boxes : arrangements num_books num_boxes = 5^6 := by
  sorry

end books_in_boxes_l1127_112725


namespace triangle_value_l1127_112755

theorem triangle_value (p : ℚ) (triangle : ℚ) 
  (eq1 : triangle * p + p = 72)
  (eq2 : (triangle * p + p) + p = 111) :
  triangle = 11 / 13 := by
  sorry

end triangle_value_l1127_112755


namespace line_through_three_points_l1127_112794

/-- Given a line passing through points (4, 10), (-3, m), and (-12, 5), prove that m = 125/16 -/
theorem line_through_three_points (m : ℚ) : 
  (let slope1 := (m - 10) / (-7 : ℚ)
   let slope2 := (5 - m) / (-9 : ℚ)
   slope1 = slope2) →
  m = 125 / 16 := by
sorry

end line_through_three_points_l1127_112794


namespace quilt_shaded_fraction_l1127_112738

/-- Represents a square quilt block -/
structure QuiltBlock where
  size : ℕ
  diagonal_shaded : Bool

/-- Calculate the shaded fraction of a quilt block -/
def shaded_fraction (q : QuiltBlock) : ℚ :=
  if q.diagonal_shaded && q.size = 3 then 1/6 else 0

/-- Theorem: The shaded fraction of a 3x3 quilt block with half-shaded diagonal squares is 1/6 -/
theorem quilt_shaded_fraction :
  ∀ (q : QuiltBlock), q.size = 3 → q.diagonal_shaded → shaded_fraction q = 1/6 := by
  sorry

end quilt_shaded_fraction_l1127_112738


namespace archie_marbles_l1127_112784

/-- Represents the number of marbles Archie has at various stages --/
structure MarbleCount where
  initial : ℕ
  afterStreet : ℕ
  afterSewer : ℕ
  afterBush : ℕ
  final : ℕ

/-- Represents the number of Glacier marbles Archie has at various stages --/
structure GlacierMarbleCount where
  initial : ℕ
  final : ℕ

/-- The main theorem about Archie's marbles --/
theorem archie_marbles (m : MarbleCount) (g : GlacierMarbleCount) : 
  (m.afterStreet = (m.initial * 2) / 5) →
  (m.afterSewer = m.afterStreet / 2) →
  (m.afterBush = (m.afterSewer * 3) / 4) →
  (m.final = m.afterBush + 5) →
  (m.final = 15) →
  (g.final = 4) →
  (g.initial = (m.initial * 3) / 10) →
  (m.initial = 67 ∧ g.initial - g.final = 16) := by
  sorry


end archie_marbles_l1127_112784


namespace complement_of_A_wrt_I_l1127_112761

def I : Finset ℕ := {1, 2, 3, 4, 5, 6}
def A : Finset ℕ := {1, 3, 4}

theorem complement_of_A_wrt_I :
  I \ A = {2, 5, 6} := by sorry

end complement_of_A_wrt_I_l1127_112761


namespace jump_frequency_proof_l1127_112720

def jump_data : List Nat := [50, 63, 77, 83, 87, 88, 89, 91, 93, 100, 102, 111, 117, 121, 130, 133, 146, 158, 177, 188]

def in_range (n : Nat) : Bool := 90 ≤ n ∧ n ≤ 110

def count_in_range (data : List Nat) : Nat :=
  data.filter in_range |>.length

theorem jump_frequency_proof :
  (count_in_range jump_data : Rat) / jump_data.length = 0.20 := by
  sorry

end jump_frequency_proof_l1127_112720


namespace fraction_subtraction_l1127_112789

theorem fraction_subtraction (x : ℝ) (h : x ≠ 1) :
  x / (x - 1) - 1 / (x - 1) = 1 := by
  sorry

end fraction_subtraction_l1127_112789


namespace least_value_of_d_l1127_112785

theorem least_value_of_d :
  let f : ℝ → ℝ := λ d => |((3 - 2*d) / 5) + 2|
  ∃ d_min : ℝ, d_min = -1 ∧
    (∀ d : ℝ, f d ≤ 3 → d ≥ d_min) ∧
    (f d_min ≤ 3) := by
  sorry

end least_value_of_d_l1127_112785


namespace point_in_second_quadrant_l1127_112790

def second_quadrant (x y : ℝ) : Prop := x < 0 ∧ y > 0

theorem point_in_second_quadrant (m : ℝ) :
  second_quadrant (-1) (m^2 + 1) := by
  sorry

end point_in_second_quadrant_l1127_112790


namespace work_completion_time_l1127_112722

theorem work_completion_time 
  (a_rate : ℚ) 
  (b_rate : ℚ) 
  (joint_work_days : ℕ) 
  (h1 : a_rate = 1 / 5) 
  (h2 : b_rate = 1 / 15) 
  (h3 : joint_work_days = 2) : 
  ℕ :=
by
  sorry

#check work_completion_time

end work_completion_time_l1127_112722


namespace range_of_f_l1127_112797

noncomputable def f (x : ℝ) : ℝ := 3 * (x - 4)

theorem range_of_f :
  ∀ y : ℝ, y ≠ -27 → ∃ x : ℝ, x ≠ -5 ∧ f x = y :=
sorry

end range_of_f_l1127_112797


namespace rook_placement_on_colored_board_l1127_112777

theorem rook_placement_on_colored_board :
  let n : ℕ := 8  -- number of rooks and rows/columns
  let m : ℕ := 32  -- number of colors
  let total_arrangements : ℕ := n.factorial
  let problematic_arrangements : ℕ := m * (n - 2).factorial
  total_arrangements > problematic_arrangements :=
by sorry

end rook_placement_on_colored_board_l1127_112777


namespace beam_cost_calculation_l1127_112736

/-- Represents the dimensions of a beam -/
structure BeamDimensions where
  thickness : ℕ
  width : ℕ
  length : ℕ

/-- Calculates the volume of a beam given its dimensions -/
def beamVolume (d : BeamDimensions) : ℕ :=
  d.thickness * d.width * d.length

/-- Calculates the total volume of multiple beams with the same dimensions -/
def totalVolume (count : ℕ) (d : BeamDimensions) : ℕ :=
  count * beamVolume d

/-- Theorem: Given the cost of 30 beams with dimensions 12x16x14,
    the cost of 14 beams with dimensions 8x12x10 is 16 2/3 coins -/
theorem beam_cost_calculation (cost_30_beams : ℚ) :
  let d1 : BeamDimensions := ⟨12, 16, 14⟩
  let d2 : BeamDimensions := ⟨8, 12, 10⟩
  cost_30_beams = 100 →
  (14 : ℚ) * cost_30_beams * (totalVolume 14 d2 : ℚ) / (totalVolume 30 d1 : ℚ) = 50 / 3 := by
  sorry

end beam_cost_calculation_l1127_112736


namespace number_triangle_problem_l1127_112751

theorem number_triangle_problem (x y : ℕ+) (h : x * y = 2022) : 
  (∃ (n : ℕ+), ∀ (m : ℕ+), (m * m ∣ x) ∧ (m * m ∣ y) → m ≤ n) ∧
  (∀ (n : ℕ+), (n * n ∣ x) ∧ (n * n ∣ y) → n = 1) :=
sorry

end number_triangle_problem_l1127_112751


namespace triangle_sides_from_heights_l1127_112763

/-- Given a triangle with heights d, e, and f corresponding to sides a, b, and c respectively,
    this theorem states the relationship between the sides and heights. -/
theorem triangle_sides_from_heights (d e f : ℝ) (hd : d > 0) (he : e > 0) (hf : f > 0) :
  ∃ (a b c : ℝ),
    let A := ((1/d + 1/e + 1/f) * (-1/d + 1/e + 1/f) * (1/d - 1/e + 1/f) * (1/d + 1/e - 1/f))
    a = 2 / (d * Real.sqrt A) ∧
    b = 2 / (e * Real.sqrt A) ∧
    c = 2 / (f * Real.sqrt A) :=
sorry

end triangle_sides_from_heights_l1127_112763


namespace quadratic_inequality_solution_l1127_112770

theorem quadratic_inequality_solution (a : ℝ) (m : ℝ) : 
  (∀ x : ℝ, ax^2 - 6*x + a^2 < 0 ↔ 1 < x ∧ x < m) → m = 2 := by
  sorry

end quadratic_inequality_solution_l1127_112770


namespace dogs_added_on_monday_l1127_112758

theorem dogs_added_on_monday
  (initial_dogs : ℕ)
  (sunday_dogs : ℕ)
  (total_dogs : ℕ)
  (h1 : initial_dogs = 2)
  (h2 : sunday_dogs = 5)
  (h3 : total_dogs = 10)
  : total_dogs - (initial_dogs + sunday_dogs) = 3 :=
by sorry

end dogs_added_on_monday_l1127_112758


namespace arithmetic_mean_set_not_full_segment_l1127_112799

open Set

/-- The set of points generated by repeatedly inserting 9 arithmetic means
    between consecutive points on the segment [a, a+1] -/
def arithmeticMeanSet (a : ℕ) : Set ℝ :=
  { x | ∃ (n : ℕ) (m : ℕ), x = a + m / (10 ^ n) ∧ m < 10 ^ n }

/-- The theorem stating that the set of points generated by repeatedly inserting
    9 arithmetic means is not equal to the entire segment [a, a+1] -/
theorem arithmetic_mean_set_not_full_segment (a : ℕ) :
  ∃ x, x ∈ Icc (a : ℝ) (a + 1) ∧ x ∉ arithmeticMeanSet a :=
by
  sorry

end arithmetic_mean_set_not_full_segment_l1127_112799


namespace star_comm_star_assoc_star_disprove_l1127_112711

-- Define the custom operation *
def star (a b : ℝ) : ℝ := a * b + a + b

-- Theorem 1: Commutativity of *
theorem star_comm (a b : ℝ) : star a b = star b a := by sorry

-- Theorem 2: Associativity of *
theorem star_assoc (a b c : ℝ) : star (star a b) c = star a (star b c) := by sorry

-- Theorem 3: Disprove the given property
theorem star_disprove : ¬(∀ (a b : ℝ), star (a + 1) b = star a b + star 1 b) := by sorry

end star_comm_star_assoc_star_disprove_l1127_112711


namespace range_of_a_l1127_112700

def p (a : ℝ) : Prop := a * (1 - a) > 0

def q (a : ℝ) : Prop := ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
  x₁^2 + (2*a - 3)*x₁ + 1 = 0 ∧ 
  x₂^2 + (2*a - 3)*x₂ + 1 = 0

def S : Set ℝ := {a | a ≤ 0 ∨ (1/2 ≤ a ∧ a < 1) ∨ a > 5/2}

theorem range_of_a : {a : ℝ | (p a ∨ q a) ∧ ¬(p a ∧ q a)} = S := by sorry

end range_of_a_l1127_112700


namespace problem_solution_l1127_112730

theorem problem_solution (x : ℝ) 
  (h : x + Real.sqrt (x^2 - 4) + 1 / (x - Real.sqrt (x^2 - 4)) = 30) :
  x^2 + Real.sqrt (x^4 - 16) + 1 / (x^2 + Real.sqrt (x^4 - 16)) = 52441/900 := by
sorry

end problem_solution_l1127_112730


namespace linear_function_characterization_l1127_112778

/-- A function satisfying the given property for a fixed α -/
def SatisfiesProperty (α : ℝ) (f : ℕ+ → ℝ) : Prop :=
  ∀ (k m : ℕ+), α * m.val ≤ k.val ∧ k.val ≤ (α + 1) * m.val → f (k + m) = f k + f m

/-- The main theorem stating that any function satisfying the property is linear -/
theorem linear_function_characterization (α : ℝ) (hα : α > 0) (f : ℕ+ → ℝ) 
  (hf : SatisfiesProperty α f) : 
  ∃ (D : ℝ), ∀ (n : ℕ+), f n = D * n.val := by
  sorry

end linear_function_characterization_l1127_112778


namespace boat_upstream_speed_l1127_112792

/-- Calculates the upstream speed of a boat given its still water speed and downstream speed -/
def upstream_speed (still_water_speed downstream_speed : ℝ) : ℝ :=
  2 * still_water_speed - downstream_speed

/-- Theorem: Given a boat with a speed of 8.5 km/hr in still water and a downstream speed of 13 km/hr, its upstream speed is 4 km/hr -/
theorem boat_upstream_speed :
  upstream_speed 8.5 13 = 4 := by
  sorry

end boat_upstream_speed_l1127_112792


namespace frank_money_problem_l1127_112721

theorem frank_money_problem (initial_money : ℝ) : 
  initial_money > 0 →
  let remaining_after_groceries := initial_money - (1/5 * initial_money)
  let remaining_after_magazine := remaining_after_groceries - (1/4 * remaining_after_groceries)
  remaining_after_magazine = 360 →
  initial_money = 600 := by
sorry


end frank_money_problem_l1127_112721


namespace chi_square_greater_than_critical_l1127_112715

/-- Represents the contingency table data --/
structure ContingencyTable where
  total_sample : ℕ
  disease_probability : ℚ
  blue_collar_with_disease : ℕ
  white_collar_without_disease : ℕ

/-- Calculates the chi-square value for the given contingency table --/
def calculate_chi_square (table : ContingencyTable) : ℚ :=
  let white_collar_with_disease := table.total_sample * table.disease_probability - table.blue_collar_with_disease
  let blue_collar_without_disease := table.total_sample * (1 - table.disease_probability) - table.white_collar_without_disease
  let n := table.total_sample
  let a := white_collar_with_disease
  let b := table.white_collar_without_disease
  let c := table.blue_collar_with_disease
  let d := blue_collar_without_disease
  n * (a * d - b * c)^2 / ((a + b) * (c + d) * (a + c) * (b + d))

/-- The critical value for α = 0.005 --/
def critical_value : ℚ := 7879 / 1000

/-- Theorem stating that the calculated chi-square value is greater than the critical value --/
theorem chi_square_greater_than_critical (table : ContingencyTable) 
  (h1 : table.total_sample = 50)
  (h2 : table.disease_probability = 3/5)
  (h3 : table.blue_collar_with_disease = 10)
  (h4 : table.white_collar_without_disease = 5) :
  calculate_chi_square table > critical_value :=
sorry

end chi_square_greater_than_critical_l1127_112715


namespace pen_ratio_l1127_112735

/-- Represents the number of pens bought by each person -/
structure PenPurchase where
  dorothy : ℕ
  julia : ℕ
  robert : ℕ

/-- The cost of one pen in cents -/
def pen_cost : ℕ := 150

/-- The total amount spent by the three friends in cents -/
def total_spent : ℕ := 3300

/-- Conditions of the pen purchase -/
def pen_purchase_conditions (p : PenPurchase) : Prop :=
  p.julia = 3 * p.robert ∧
  p.robert = 4 ∧
  p.dorothy + p.julia + p.robert = total_spent / pen_cost

theorem pen_ratio (p : PenPurchase) 
  (h : pen_purchase_conditions p) : 
  p.dorothy * 2 = p.julia := by
  sorry

end pen_ratio_l1127_112735


namespace horner_method_difference_l1127_112796

def f (x : ℝ) : ℝ := 1 - 5*x - 8*x^2 + 10*x^3 + 6*x^4 + 12*x^5 + 3*x^6

def v₀ : ℝ := 3
def v₁ (x : ℝ) : ℝ := v₀ * x + 12
def v₂ (x : ℝ) : ℝ := v₁ x * x + 6
def v₃ (x : ℝ) : ℝ := v₂ x * x + 10
def v₄ (x : ℝ) : ℝ := v₃ x * x - 8

theorem horner_method_difference (x : ℝ) (hx : x = -4) :
  (max v₀ (max (v₁ x) (max (v₂ x) (max (v₃ x) (v₄ x))))) -
  (min v₀ (min (v₁ x) (min (v₂ x) (min (v₃ x) (v₄ x))))) = 62 := by
  sorry

end horner_method_difference_l1127_112796


namespace percentage_markup_proof_l1127_112740

def selling_price : ℚ := 8587
def cost_price : ℚ := 6925

theorem percentage_markup_proof :
  let markup := selling_price - cost_price
  let percentage_markup := (markup / cost_price) * 100
  ∃ ε > 0, abs (percentage_markup - 23.99) < ε := by
sorry

end percentage_markup_proof_l1127_112740


namespace cow_count_l1127_112757

theorem cow_count (total_legs : ℕ) (legs_per_cow : ℕ) (h1 : total_legs = 460) (h2 : legs_per_cow = 4) : 
  total_legs / legs_per_cow = 115 := by
sorry

end cow_count_l1127_112757


namespace inequality_proof_l1127_112764

theorem inequality_proof (a b c : ℝ) (n : ℕ) 
  (ha : a ≥ 0) (hb : b ≥ 0) (hc : c ≥ 0) (hn : n > 0) :
  a^n + b^n + c^n ≥ a*b^(n-1) + b*c^(n-1) + c*a^(n-1) := by
  sorry

end inequality_proof_l1127_112764


namespace problem_statement_l1127_112744

theorem problem_statement (a b : ℝ) 
  (h1 : a < b) (h2 : b < 0) (h3 : a^2 + b^2 = 4*a*b) : 
  (a + b) / (a - b) = Real.sqrt 3 := by
  sorry

end problem_statement_l1127_112744


namespace hairs_to_grow_back_l1127_112793

def hairs_lost_washing : ℕ := 32

def hairs_lost_brushing : ℕ := hairs_lost_washing / 2

def total_hairs_lost : ℕ := hairs_lost_washing + hairs_lost_brushing

theorem hairs_to_grow_back : total_hairs_lost + 1 = 49 := by sorry

end hairs_to_grow_back_l1127_112793


namespace complete_square_quadratic_l1127_112791

theorem complete_square_quadratic (x : ℝ) : 
  (∃ c d : ℝ, x^2 + 6*x - 3 = 0 ↔ (x + c)^2 = d) → 
  (∃ c : ℝ, (x + c)^2 = 12) :=
by sorry

end complete_square_quadratic_l1127_112791


namespace twenty_sheets_joined_length_l1127_112774

/-- The length of joined papers given the number of sheets, length per sheet, and overlap length -/
def joinedPapersLength (numSheets : ℕ) (sheetLength : ℝ) (overlapLength : ℝ) : ℝ :=
  numSheets * sheetLength - (numSheets - 1) * overlapLength

/-- Theorem stating that 20 sheets of 10 cm paper with 0.5 cm overlap results in 190.5 cm total length -/
theorem twenty_sheets_joined_length :
  joinedPapersLength 20 10 0.5 = 190.5 := by
  sorry

#eval joinedPapersLength 20 10 0.5

end twenty_sheets_joined_length_l1127_112774


namespace quadratic_discriminant_l1127_112787

/-- The discriminant of a quadratic equation ax^2 + bx + c = 0 -/
def discriminant (a b c : ℝ) : ℝ := b^2 - 4*a*c

/-- The coefficients of the quadratic equation x^2 - 7x + 4 = 0 -/
def a : ℝ := 1
def b : ℝ := -7
def c : ℝ := 4

theorem quadratic_discriminant : discriminant a b c = 33 := by
  sorry

end quadratic_discriminant_l1127_112787


namespace sqrt_difference_inequality_l1127_112783

theorem sqrt_difference_inequality (x : ℝ) (h : x ≥ 4) :
  Real.sqrt x - Real.sqrt (x - 1) ≥ 1 / x := by
  sorry

end sqrt_difference_inequality_l1127_112783


namespace min_reciprocal_sum_squares_min_reciprocal_sum_squares_achieved_l1127_112768

theorem min_reciprocal_sum_squares (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 20) :
  (1 / x^2 + 1 / y^2) ≥ 2 / 25 :=
by sorry

theorem min_reciprocal_sum_squares_achieved (ε : ℝ) (hε : ε > 0) :
  ∃ x y : ℝ, x > 0 ∧ y > 0 ∧ x + y = 20 ∧ 1 / x^2 + 1 / y^2 < 2 / 25 + ε :=
by sorry

end min_reciprocal_sum_squares_min_reciprocal_sum_squares_achieved_l1127_112768


namespace outer_circle_diameter_l1127_112782

/-- Proves that given an outer circle with diameter D and an inner circle with diameter 24,
    if 0.36 of the outer circle's surface is not covered by the inner circle,
    then the diameter of the outer circle is 30. -/
theorem outer_circle_diameter
  (D : ℝ) -- Diameter of the outer circle
  (h1 : D > 0) -- Diameter is positive
  (h2 : π * (D / 2)^2 - π * 12^2 = 0.36 * π * (D / 2)^2) -- Condition about uncovered area
  : D = 30 := by
  sorry


end outer_circle_diameter_l1127_112782


namespace equation_solution_l1127_112724

theorem equation_solution :
  ∀ x : ℝ, (x + 4)^2 = 5*(x + 4) ↔ x = -4 ∨ x = 1 := by
  sorry

end equation_solution_l1127_112724


namespace trihedral_angle_inequalities_l1127_112771

structure TrihedralAngle where
  SA : Real
  SB : Real
  SC : Real
  α : Real
  β : Real
  γ : Real
  ASB : Real
  BSC : Real
  CSA : Real

def is_acute_dihedral (t : TrihedralAngle) : Prop := sorry

theorem trihedral_angle_inequalities (t : TrihedralAngle) :
  t.α + t.β + t.γ ≤ t.ASB + t.BSC + t.CSA ∧
  (is_acute_dihedral t → t.α + t.β + t.γ ≥ (t.ASB + t.BSC + t.CSA) / 2) := by
  sorry

end trihedral_angle_inequalities_l1127_112771


namespace officer_selection_l1127_112717

theorem officer_selection (n m : ℕ) (hn : n = 5) (hm : m = 3) :
  (n.choose m) * m.factorial = 60 := by
  sorry

end officer_selection_l1127_112717


namespace geom_seq_sum_property_l1127_112737

/-- Represents a geometric sequence and its properties -/
structure GeometricSequence where
  a : ℕ → ℝ  -- The sequence
  q : ℝ      -- Common ratio
  S : ℕ → ℝ  -- Sum function
  sum_formula : ∀ n, S n = (a 1) * (1 - q^n) / (1 - q)
  geom_seq : ∀ n, a (n + 1) = q * a n

/-- 
Given a geometric sequence with S_4 = 1 and S_12 = 13,
prove that a_13 + a_14 + a_15 + a_16 = 27
-/
theorem geom_seq_sum_property (g : GeometricSequence) 
  (h1 : g.S 4 = 1) (h2 : g.S 12 = 13) :
  g.a 13 + g.a 14 + g.a 15 + g.a 16 = 27 := by
  sorry

end geom_seq_sum_property_l1127_112737


namespace olympic_medals_l1127_112747

/-- Olympic Medals Theorem -/
theorem olympic_medals (china_total russia_total us_total : ℕ)
  (china_gold china_silver china_bronze : ℕ)
  (russia_gold russia_silver russia_bronze : ℕ)
  (us_gold us_silver us_bronze : ℕ)
  (h1 : china_total = 100)
  (h2 : russia_total = 72)
  (h3 : us_total = 110)
  (h4 : china_silver + china_bronze = russia_silver + russia_bronze)
  (h5 : russia_gold + 28 = china_gold)
  (h6 : us_gold = russia_gold + 13)
  (h7 : us_gold = us_bronze)
  (h8 : us_silver = us_gold + 2)
  (h9 : china_bronze = china_silver + 7)
  (h10 : china_total = china_gold + china_silver + china_bronze)
  (h11 : russia_total = russia_gold + russia_silver + russia_bronze)
  (h12 : us_total = us_gold + us_silver + us_bronze) :
  china_gold = 51 ∧ us_silver = 38 ∧ russia_bronze = 28 := by
  sorry


end olympic_medals_l1127_112747


namespace min_value_trigonometric_expression_l1127_112746

open Real

theorem min_value_trigonometric_expression (θ : ℝ) (h : 0 < θ ∧ θ < π/2) :
  (4 * cos θ + 3 / sin θ + 2 * sqrt 2 * tan θ) ≥ 6 * sqrt 3 * (2 ^ (1/6)) :=
by sorry

end min_value_trigonometric_expression_l1127_112746


namespace andrew_remaining_vacation_days_l1127_112719

def vacation_days_earned (days_worked : ℕ) : ℕ :=
  days_worked / 10

def days_count_for_vacation (total_days_worked public_holidays sick_leave : ℕ) : ℕ :=
  total_days_worked - public_holidays - sick_leave

theorem andrew_remaining_vacation_days 
  (total_days_worked : ℕ) 
  (public_holidays : ℕ) 
  (sick_leave : ℕ) 
  (march_vacation : ℕ) 
  (h1 : total_days_worked = 290)
  (h2 : public_holidays = 10)
  (h3 : sick_leave = 5)
  (h4 : march_vacation = 5) :
  vacation_days_earned (days_count_for_vacation total_days_worked public_holidays sick_leave) - 
  (march_vacation + 2 * march_vacation) = 12 :=
by
  sorry

#eval vacation_days_earned (days_count_for_vacation 290 10 5) - (5 + 2 * 5)

end andrew_remaining_vacation_days_l1127_112719


namespace average_daily_high_temperature_l1127_112709

def daily_highs : List ℝ := [49, 62, 58, 57, 46]

theorem average_daily_high_temperature :
  (daily_highs.sum / daily_highs.length : ℝ) = 54.4 := by
  sorry

end average_daily_high_temperature_l1127_112709


namespace constant_ratio_problem_l1127_112727

theorem constant_ratio_problem (k : ℝ) (x₁ x₂ y₁ y₂ : ℝ) :
  (∀ x y, (4 * x + 3) / (2 * y - 5) = k) →
  x₁ = 1 →
  y₁ = 5 →
  y₂ = 10 →
  x₂ = 9 / 2 :=
by sorry

end constant_ratio_problem_l1127_112727


namespace village_population_l1127_112712

theorem village_population (final_population : ℕ) : 
  final_population = 4860 → 
  ∃ (original_population : ℕ), 
    (original_population : ℝ) * 0.9 * 0.75 = final_population ∧ 
    original_population = 7200 := by
  sorry

end village_population_l1127_112712


namespace one_real_solution_l1127_112775

/-- The number of distinct real solutions to the equation (x-5)(x^2 + 5x + 10) = 0 -/
def num_solutions : ℕ := 1

/-- The equation (x-5)(x^2 + 5x + 10) = 0 has exactly one real solution -/
theorem one_real_solution : num_solutions = 1 := by
  sorry

end one_real_solution_l1127_112775


namespace dave_lost_tickets_l1127_112702

/-- Prove that Dave lost 2 tickets at the arcade -/
theorem dave_lost_tickets (initial_tickets : ℕ) (spent_tickets : ℕ) (remaining_tickets : ℕ) 
  (h1 : initial_tickets = 14)
  (h2 : spent_tickets = 10)
  (h3 : remaining_tickets = 2) :
  initial_tickets - (spent_tickets + remaining_tickets) = 2 := by
  sorry

end dave_lost_tickets_l1127_112702


namespace second_quadrant_point_coordinates_l1127_112769

/-- A point in the second quadrant of a coordinate plane. -/
structure SecondQuadrantPoint where
  x : ℝ
  y : ℝ
  x_neg : x < 0
  y_pos : y > 0

/-- The theorem stating that a point in the second quadrant with given distances to the axes has specific coordinates. -/
theorem second_quadrant_point_coordinates (P : SecondQuadrantPoint) 
  (dist_x_axis : |P.y| = 4)
  (dist_y_axis : |P.x| = 5) :
  P.x = -5 ∧ P.y = 4 := by
  sorry

end second_quadrant_point_coordinates_l1127_112769


namespace min_value_of_w_l1127_112798

theorem min_value_of_w :
  ∀ (x y z w : ℝ),
    -2 ≤ x ∧ x ≤ 5 →
    -3 ≤ y ∧ y ≤ 7 →
    4 ≤ z ∧ z ≤ 8 →
    w = x * y - z →
    w ≥ -23 ∧ ∃ (x₀ y₀ z₀ : ℝ),
      -2 ≤ x₀ ∧ x₀ ≤ 5 ∧
      -3 ≤ y₀ ∧ y₀ ≤ 7 ∧
      4 ≤ z₀ ∧ z₀ ≤ 8 ∧
      x₀ * y₀ - z₀ = -23 :=
by sorry


end min_value_of_w_l1127_112798


namespace triangle_sine_inequality_l1127_112733

theorem triangle_sine_inequality (A B C : Real) (h : A + B + C = Real.pi) :
  Real.sin A ^ 2 + Real.sin B ^ 2 + Real.sin C ^ 2 ≤ 9 / 4 := by
  sorry

end triangle_sine_inequality_l1127_112733


namespace overlapping_rectangle_area_l1127_112734

theorem overlapping_rectangle_area (Y : ℝ) (X : ℝ) (h1 : Y > 0) (h2 : X > 0) 
  (h3 : X = (1/8) * (2*Y - X)) : X = (2/9) * Y := by
  sorry

end overlapping_rectangle_area_l1127_112734


namespace incorrect_number_value_l1127_112742

theorem incorrect_number_value (n : ℕ) (initial_avg correct_avg correct_value : ℚ) 
  (h1 : n = 10)
  (h2 : initial_avg = 20)
  (h3 : correct_avg = 26)
  (h4 : correct_value = 86) :
  ∃ x : ℚ, n * correct_avg - n * initial_avg = correct_value - x ∧ x = 26 := by
sorry

end incorrect_number_value_l1127_112742


namespace probability_log_base_2_equal_1_l1127_112703

def dice_face := Fin 6

def is_valid_roll (x y : dice_face) : Prop :=
  (y.val : ℝ) = 2 * (x.val : ℝ)

def total_outcomes : ℕ := 36

def favorable_outcomes : ℕ := 3

theorem probability_log_base_2_equal_1 :
  (favorable_outcomes : ℝ) / total_outcomes = 1 / 12 := by
  sorry

end probability_log_base_2_equal_1_l1127_112703


namespace polynomial_division_theorem_l1127_112701

theorem polynomial_division_theorem (x : ℝ) : 
  (x^5 + x^4 + x^3 + x^2 + x + 1) * (x - 1) + 9 = x^6 + 8 := by
  sorry

end polynomial_division_theorem_l1127_112701


namespace solution_set_for_m_eq_2_m_range_for_solution_set_R_l1127_112795

-- Define the function f
def f (m : ℝ) (x : ℝ) : ℝ := x^2 + (m-1)*x - m

-- Part 1
theorem solution_set_for_m_eq_2 :
  {x : ℝ | f 2 x < 0} = {x : ℝ | -2 < x ∧ x < 1} := by sorry

-- Part 2
theorem m_range_for_solution_set_R :
  (∀ x, f m x ≥ -1) → m ∈ Set.Icc (-3) 1 := by sorry

end solution_set_for_m_eq_2_m_range_for_solution_set_R_l1127_112795


namespace complex_number_in_fourth_quadrant_l1127_112750

theorem complex_number_in_fourth_quadrant :
  let i : ℂ := Complex.I
  let z : ℂ := (2 - i) / (1 + i)
  (z.re > 0 ∧ z.im < 0) := by sorry

end complex_number_in_fourth_quadrant_l1127_112750


namespace largest_n_for_unique_k_l1127_112728

theorem largest_n_for_unique_k : 
  ∀ n : ℕ, n > 112 → 
  ¬(∃! k : ℤ, (7 : ℚ)/16 < (n : ℚ)/(n + k) ∧ (n : ℚ)/(n + k) < 8/17) ∧ 
  (∃! k : ℤ, (7 : ℚ)/16 < (112 : ℚ)/(112 + k) ∧ (112 : ℚ)/(112 + k) < 8/17) :=
by sorry

end largest_n_for_unique_k_l1127_112728


namespace platform_length_l1127_112767

/-- Given a train of length 450 m, running at 108 kmph, crosses a platform in 25 seconds,
    prove that the length of the platform is 300 m. -/
theorem platform_length (train_length : ℝ) (train_speed_kmph : ℝ) (crossing_time : ℝ) :
  train_length = 450 ∧ 
  train_speed_kmph = 108 ∧ 
  crossing_time = 25 →
  (train_speed_kmph * 1000 / 3600 * crossing_time - train_length) = 300 := by
sorry

end platform_length_l1127_112767


namespace min_value_theorem_l1127_112760

theorem min_value_theorem (x y : ℝ) (h : 2 * x^2 + 3 * x * y + 2 * y^2 = 1) :
  ∃ (min_val : ℝ), min_val = -9/8 ∧ ∀ (a b : ℝ), 2 * a^2 + 3 * a * b + 2 * b^2 = 1 →
    x + y + x * y ≥ min_val ∧ a + b + a * b ≥ min_val :=
by sorry

end min_value_theorem_l1127_112760


namespace test_score_calculation_l1127_112729

theorem test_score_calculation (total_questions : ℕ) (correct_answers : ℕ) (incorrect_penalty : ℕ) (total_score : ℕ) :
  total_questions = 30 →
  correct_answers = 19 →
  incorrect_penalty = 5 →
  total_score = 325 →
  ∃ (points_per_correct : ℕ),
    points_per_correct * correct_answers - incorrect_penalty * (total_questions - correct_answers) = total_score ∧
    points_per_correct = 20 :=
by sorry

end test_score_calculation_l1127_112729


namespace pure_imaginary_solution_l1127_112732

-- Define the complex number i
def i : ℂ := Complex.I

-- Define the complex-valued function z
def z (m : ℝ) : ℂ := (1 + i) * m^2 - (4 + i) * m + 3

-- Theorem statement
theorem pure_imaginary_solution (m : ℝ) :
  (z m).re = 0 ∧ (z m).im ≠ 0 → m = 3 :=
sorry

end pure_imaginary_solution_l1127_112732


namespace second_discount_percentage_l1127_112710

theorem second_discount_percentage (original_price : ℝ) (first_discount : ℝ) (third_discount : ℝ) (final_price : ℝ) :
  original_price = 9795.3216374269 →
  first_discount = 20 →
  third_discount = 5 →
  final_price = 6700 →
  ∃ (second_discount : ℝ), 
    (original_price * (1 - first_discount / 100) * (1 - second_discount / 100) * (1 - third_discount / 100) = final_price) ∧
    (abs (second_discount - 10) < 0.0000000001) := by
  sorry

end second_discount_percentage_l1127_112710


namespace parabola_focus_l1127_112759

/-- The parabola equation -/
def parabola_equation (x y : ℝ) : Prop := x = (1/4) * y^2

/-- The focus of a parabola -/
def focus : ℝ × ℝ := (1, 0)

/-- Theorem: The focus of the parabola x = (1/4)y^2 is at (1, 0) -/
theorem parabola_focus :
  ∀ (x y : ℝ), parabola_equation x y → 
  (∃ (p : ℝ × ℝ), p = focus ∧ 
   (x - p.1)^2 + (y - p.2)^2 = (x + p.1)^2) :=
by sorry

end parabola_focus_l1127_112759


namespace odd_terms_in_binomial_expansion_l1127_112776

/-- 
Given odd integers a and b, the number of odd terms 
in the expansion of (a+b)^8 is equal to 2.
-/
theorem odd_terms_in_binomial_expansion (a b : ℤ) 
  (ha : Odd a) (hb : Odd b) : 
  (Finset.filter (fun i => Odd (Nat.choose 8 i * a^(8-i) * b^i)) 
    (Finset.range 9)).card = 2 := by
  sorry

end odd_terms_in_binomial_expansion_l1127_112776
