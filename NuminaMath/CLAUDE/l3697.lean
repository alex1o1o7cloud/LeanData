import Mathlib

namespace NUMINAMATH_CALUDE_work_completion_time_l3697_369797

-- Define the efficiency of worker B
def B_efficiency : ℚ := 1 / 24

-- Define the efficiency of worker A (twice as efficient as B)
def A_efficiency : ℚ := 2 * B_efficiency

-- Define the combined efficiency of A and B
def combined_efficiency : ℚ := A_efficiency + B_efficiency

-- Theorem: A and B together can complete the work in 8 days
theorem work_completion_time : (1 : ℚ) / combined_efficiency = 8 := by sorry

end NUMINAMATH_CALUDE_work_completion_time_l3697_369797


namespace NUMINAMATH_CALUDE_mode_of_sample_data_l3697_369716

def sample_data : List Int := [-2, 0, 6, 3, 6]

def mode (data : List Int) : Int :=
  data.foldl (fun acc x => if data.count x > data.count acc then x else acc) 0

theorem mode_of_sample_data :
  mode sample_data = 6 := by sorry

end NUMINAMATH_CALUDE_mode_of_sample_data_l3697_369716


namespace NUMINAMATH_CALUDE_squares_in_figure_100_l3697_369751

-- Define the sequence function
def f (n : ℕ) : ℕ := 2 * n^3 + 2 * n^2 + 4 * n + 1

-- State the theorem
theorem squares_in_figure_100 :
  f 0 = 1 ∧ f 1 = 9 ∧ f 2 = 29 ∧ f 3 = 65 → f 100 = 2020401 :=
by
  sorry


end NUMINAMATH_CALUDE_squares_in_figure_100_l3697_369751


namespace NUMINAMATH_CALUDE_unique_base_representation_l3697_369739

theorem unique_base_representation : ∃! n : ℕ+, 
  ∃ A B : ℕ, 
    (0 ≤ A ∧ A < 7) ∧ 
    (0 ≤ B ∧ B < 7) ∧
    (0 ≤ A ∧ A < 5) ∧ 
    (0 ≤ B ∧ B < 5) ∧
    (n : ℕ) = 7 * A + B ∧
    (n : ℕ) = 5 * B + A ∧
    (n : ℕ) = 17 := by
  sorry

end NUMINAMATH_CALUDE_unique_base_representation_l3697_369739


namespace NUMINAMATH_CALUDE_no_valid_solution_l3697_369781

/-- Represents the conditions of the age problem -/
structure AgeProblem where
  jane_current_age : ℕ
  dick_current_age : ℕ
  n : ℕ
  jane_future_age : ℕ
  dick_future_age : ℕ

/-- Checks if the given ages satisfy the problem conditions -/
def satisfies_conditions (problem : AgeProblem) : Prop :=
  problem.jane_current_age = 30 ∧
  problem.dick_current_age = problem.jane_current_age + 5 ∧
  problem.n > 0 ∧
  problem.jane_future_age = problem.jane_current_age + problem.n ∧
  problem.dick_future_age = problem.dick_current_age + problem.n ∧
  10 ≤ problem.jane_future_age ∧ problem.jane_future_age ≤ 99 ∧
  10 ≤ problem.dick_future_age ∧ problem.dick_future_age ≤ 99 ∧
  (problem.jane_future_age / 10 = problem.dick_future_age % 10) ∧
  (problem.jane_future_age % 10 = problem.dick_future_age / 10)

/-- The main theorem stating that no valid solution exists -/
theorem no_valid_solution : ¬∃ (problem : AgeProblem), satisfies_conditions problem := by
  sorry

end NUMINAMATH_CALUDE_no_valid_solution_l3697_369781


namespace NUMINAMATH_CALUDE_no_real_solutions_l3697_369718

theorem no_real_solutions : ¬∃ (x y : ℝ), x^2 + 3*y^2 - 4*x - 6*y + 10 = 0 := by
  sorry

end NUMINAMATH_CALUDE_no_real_solutions_l3697_369718


namespace NUMINAMATH_CALUDE_fourth_root_is_negative_seven_l3697_369774

/-- Represents a polynomial of degree 4 with rational coefficients -/
structure QuarticPolynomial where
  d : ℚ
  e : ℚ
  f : ℚ

/-- Checks if a given number is a root of the polynomial -/
def isRoot (p : QuarticPolynomial) (x : ℝ) : Prop :=
  x^4 + p.d * x^2 + p.e * x + p.f = 0

theorem fourth_root_is_negative_seven
  (p : QuarticPolynomial)
  (h1 : isRoot p (3 - Real.sqrt 5))
  (h2 : ∃ (a b : ℤ), isRoot p a ∧ isRoot p b) :
  isRoot p (-7) :=
sorry

end NUMINAMATH_CALUDE_fourth_root_is_negative_seven_l3697_369774


namespace NUMINAMATH_CALUDE_tree_support_uses_triangle_stability_l3697_369790

/-- A triangle formed by two supporting sticks and a tree -/
structure TreeSupport where
  stickOne : ℝ × ℝ  -- Coordinates of the first stick's base
  stickTwo : ℝ × ℝ  -- Coordinates of the second stick's base
  treeTop : ℝ × ℝ   -- Coordinates of the tree's top

/-- The property of a triangle that provides support -/
def triangleProperty : String := "stability"

/-- 
  Theorem: The property of triangles applied when using two wooden sticks 
  to support a falling tree is stability.
-/
theorem tree_support_uses_triangle_stability (support : TreeSupport) : 
  triangleProperty = "stability" := by
  sorry

end NUMINAMATH_CALUDE_tree_support_uses_triangle_stability_l3697_369790


namespace NUMINAMATH_CALUDE_bill_toilet_paper_usage_l3697_369700

/-- Calculates the number of toilet paper squares used per bathroom visit -/
def toilet_paper_usage (bathroom_visits_per_day : ℕ) (total_rolls : ℕ) (squares_per_roll : ℕ) (total_days : ℕ) : ℕ :=
  (total_rolls * squares_per_roll) / (total_days * bathroom_visits_per_day)

/-- Proves that Bill uses 5 squares of toilet paper per bathroom visit -/
theorem bill_toilet_paper_usage :
  toilet_paper_usage 3 1000 300 20000 = 5 := by
  sorry

end NUMINAMATH_CALUDE_bill_toilet_paper_usage_l3697_369700


namespace NUMINAMATH_CALUDE_smallest_gcd_multiple_l3697_369728

theorem smallest_gcd_multiple (p q : ℕ+) (h : Nat.gcd p q = 9) :
  (∀ p q : ℕ+, Nat.gcd p q = 9 → Nat.gcd (8 * p) (18 * q) ≥ 18) ∧
  (∃ p q : ℕ+, Nat.gcd p q = 9 ∧ Nat.gcd (8 * p) (18 * q) = 18) :=
by sorry

end NUMINAMATH_CALUDE_smallest_gcd_multiple_l3697_369728


namespace NUMINAMATH_CALUDE_minimum_value_range_l3697_369787

noncomputable def f (x : ℝ) := x^3 - 3*x

def has_minimum_on_interval (f : ℝ → ℝ) (a b : ℝ) :=
  ∃ (c : ℝ), a < c ∧ c < b ∧ ∀ (x : ℝ), a < x ∧ x < b → f c ≤ f x

theorem minimum_value_range (a : ℝ) :
  has_minimum_on_interval f a (10 + 2*a^2) ↔ -2 ≤ a ∧ a < 1 :=
sorry

end NUMINAMATH_CALUDE_minimum_value_range_l3697_369787


namespace NUMINAMATH_CALUDE_first_purchase_amount_l3697_369772

/-- Represents the student-entrepreneur's mask selling scenario -/
structure MaskSelling where
  /-- Cost price of each package of masks (in rubles) -/
  cost_price : ℝ
  /-- Selling price of each package of masks (in rubles) -/
  selling_price : ℝ
  /-- Number of packages bought in the first purchase -/
  initial_quantity : ℝ
  /-- Profit from the first sale (in rubles) -/
  first_profit : ℝ
  /-- Profit from the second sale (in rubles) -/
  second_profit : ℝ

/-- Theorem stating the amount spent on the first purchase -/
theorem first_purchase_amount (m : MaskSelling)
  (h1 : m.first_profit = 1000)
  (h2 : m.second_profit = 1500)
  (h3 : m.selling_price > m.cost_price)
  (h4 : m.initial_quantity * m.selling_price = 
        (m.initial_quantity * m.selling_price / m.cost_price) * m.cost_price) :
  m.initial_quantity * m.cost_price = 2000 := by
  sorry


end NUMINAMATH_CALUDE_first_purchase_amount_l3697_369772


namespace NUMINAMATH_CALUDE_power_of_power_l3697_369720

theorem power_of_power (a : ℝ) : (a^3)^2 = a^6 := by
  sorry

end NUMINAMATH_CALUDE_power_of_power_l3697_369720


namespace NUMINAMATH_CALUDE_paper_cranes_count_l3697_369741

theorem paper_cranes_count (T : ℕ) : 
  (T / 2 : ℚ) - (T / 2 : ℚ) / 5 = 400 → T = 1000 := by
  sorry

end NUMINAMATH_CALUDE_paper_cranes_count_l3697_369741


namespace NUMINAMATH_CALUDE_circle_tangent_to_line_and_center_at_parabola_focus_l3697_369708

theorem circle_tangent_to_line_and_center_at_parabola_focus :
  ∀ (x y : ℝ),
  (∃ (h : ℝ), y^2 = 8*x → (2, 0) = (h, 0)) →
  (∃ (r : ℝ), r = Real.sqrt 2) →
  (x - 2)^2 + y^2 = 2 →
  (∃ (t : ℝ), t = x ∧ t = y) →
  (∃ (d : ℝ), d = |x - y| / Real.sqrt 2 ∧ d = Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_circle_tangent_to_line_and_center_at_parabola_focus_l3697_369708


namespace NUMINAMATH_CALUDE_simplify_expression_l3697_369747

theorem simplify_expression (x : ℝ) : (3*x + 25) + (150*x - 5) + x^2 = x^2 + 153*x + 20 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3697_369747


namespace NUMINAMATH_CALUDE_sixth_term_is_three_l3697_369782

/-- An arithmetic progression with specific properties -/
structure ArithmeticProgression where
  a : ℕ → ℝ  -- The sequence
  sum_first_three : (a 1) + (a 2) + (a 3) = 168
  diff_two_five : (a 2) - (a 5) = 42

/-- The 6th term of the arithmetic progression is 3 -/
theorem sixth_term_is_three (ap : ArithmeticProgression) : ap.a 6 = 3 := by
  sorry

end NUMINAMATH_CALUDE_sixth_term_is_three_l3697_369782


namespace NUMINAMATH_CALUDE_biased_die_probability_l3697_369773

theorem biased_die_probability (p : ℝ) (h1 : 0 ≤ p ∧ p ≤ 1) :
  (Nat.choose 8 6 : ℝ) * p^6 * (1-p)^2 = 125/256 → p^6 * (1-p)^2 = 125/7168 := by
  sorry

end NUMINAMATH_CALUDE_biased_die_probability_l3697_369773


namespace NUMINAMATH_CALUDE_complex_number_quadrant_l3697_369703

theorem complex_number_quadrant : 
  let z : ℂ := (2 - Complex.I) / (1 + Complex.I)
  (z.re > 0 ∧ z.im < 0) := by sorry

end NUMINAMATH_CALUDE_complex_number_quadrant_l3697_369703


namespace NUMINAMATH_CALUDE_necessary_but_not_sufficient_l3697_369789

theorem necessary_but_not_sufficient : 
  (∀ x : ℝ, x > 3 → x > 1) ∧ 
  (∃ x : ℝ, x > 1 ∧ ¬(x > 3)) := by
  sorry

end NUMINAMATH_CALUDE_necessary_but_not_sufficient_l3697_369789


namespace NUMINAMATH_CALUDE_sum_of_multiples_of_6_and_9_is_multiple_of_3_l3697_369793

theorem sum_of_multiples_of_6_and_9_is_multiple_of_3 (x y : ℤ) 
  (hx : ∃ m : ℤ, x = 6 * m) 
  (hy : ∃ n : ℤ, y = 9 * n) : 
  ∃ k : ℤ, x + y = 3 * k := by
sorry

end NUMINAMATH_CALUDE_sum_of_multiples_of_6_and_9_is_multiple_of_3_l3697_369793


namespace NUMINAMATH_CALUDE_given_number_eq_scientific_form_l3697_369709

/-- Scientific notation representation of a real number -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  norm : 1 ≤ |coefficient| ∧ |coefficient| < 10

/-- The given number to be expressed in scientific notation -/
def givenNumber : ℝ := 0.00076

/-- The scientific notation representation of the given number -/
def scientificForm : ScientificNotation :=
  { coefficient := 7.6
    exponent := -4
    norm := by sorry }

/-- Theorem stating that the given number is equal to its scientific notation representation -/
theorem given_number_eq_scientific_form : 
  givenNumber = scientificForm.coefficient * (10 : ℝ) ^ scientificForm.exponent := by
  sorry

end NUMINAMATH_CALUDE_given_number_eq_scientific_form_l3697_369709


namespace NUMINAMATH_CALUDE_port_vessel_count_port_vessel_count_proof_l3697_369750

theorem port_vessel_count : ℕ → ℕ → ℕ → ℕ → Prop :=
  fun cruise_ships cargo_ships sailboats fishing_boats =>
    cruise_ships = 4 ∧
    cargo_ships = 2 * cruise_ships ∧
    sailboats = cargo_ships + 6 ∧
    sailboats = 7 * fishing_boats →
    cruise_ships + cargo_ships + sailboats + fishing_boats = 28

/-- Proof of the theorem -/
theorem port_vessel_count_proof : port_vessel_count 4 8 14 2 := by
  sorry

end NUMINAMATH_CALUDE_port_vessel_count_port_vessel_count_proof_l3697_369750


namespace NUMINAMATH_CALUDE_rectangle_point_distances_l3697_369764

-- Define the rectangle and point P
def Rectangle (A B C D : ℝ × ℝ) : Prop :=
  -- Add conditions for a rectangle here
  True

def InsideRectangle (P : ℝ × ℝ) (A B C D : ℝ × ℝ) : Prop :=
  -- Add conditions for P being inside the rectangle here
  True

-- Define the distance function
def distance (P Q : ℝ × ℝ) : ℝ :=
  -- Add definition for Euclidean distance here
  0

-- Theorem statement
theorem rectangle_point_distances 
  (A B C D P : ℝ × ℝ) 
  (h_rect : Rectangle A B C D) 
  (h_inside : InsideRectangle P A B C D) 
  (h_PA : distance P A = 5)
  (h_PD : distance P D = 12)
  (h_PC : distance P C = 13) :
  distance P B = 5 * Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_rectangle_point_distances_l3697_369764


namespace NUMINAMATH_CALUDE_cone_axial_angle_when_max_section_twice_axial_l3697_369765

/-- Represents a right circular cone -/
structure RightCircularCone where
  vertex : Point
  axialAngle : ℝ

/-- Represents a cross-section of a cone -/
structure ConeSection where
  cone : RightCircularCone
  angle : ℝ

/-- The area of a cone section -/
def sectionArea (s : ConeSection) : ℝ := sorry

/-- The maximum area cross-section of a cone -/
def maxSectionArea (c : RightCircularCone) : ℝ := sorry

/-- The axial cross-section of a cone -/
def axialSection (c : RightCircularCone) : ConeSection := sorry

theorem cone_axial_angle_when_max_section_twice_axial 
  (c : RightCircularCone) :
  maxSectionArea c = 2 * sectionArea (axialSection c) →
  c.axialAngle = 120 * π / 180 := by sorry

end NUMINAMATH_CALUDE_cone_axial_angle_when_max_section_twice_axial_l3697_369765


namespace NUMINAMATH_CALUDE_unique_solution_iff_a_eq_half_l3697_369724

/-- The equation has a unique solution if and only if a = 1/2 -/
theorem unique_solution_iff_a_eq_half (a : ℝ) (ha : a > 0) :
  (∃! x : ℝ, 2 * a * x = x^2 - 2 * a * Real.log x) ↔ a = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_iff_a_eq_half_l3697_369724


namespace NUMINAMATH_CALUDE_mod_equivalence_l3697_369723

theorem mod_equivalence (m : ℕ) : 
  198 * 935 ≡ m [ZMOD 50] → 0 ≤ m → m < 50 → m = 30 := by
  sorry

end NUMINAMATH_CALUDE_mod_equivalence_l3697_369723


namespace NUMINAMATH_CALUDE_tangent_points_distance_l3697_369770

theorem tangent_points_distance (r : ℝ) (d : ℝ) (h1 : r = 7) (h2 : d = 25) :
  let tangent_length := Real.sqrt (d^2 - r^2)
  2 * tangent_length = 48 :=
sorry

end NUMINAMATH_CALUDE_tangent_points_distance_l3697_369770


namespace NUMINAMATH_CALUDE_M_intersect_N_eq_open_zero_one_l3697_369775

-- Define set M
def M : Set ℝ := {x | x^2 + x - 2 < 0}

-- Define set N
def N : Set ℝ := {x | 0 < x ∧ x ≤ 2}

-- Theorem statement
theorem M_intersect_N_eq_open_zero_one : M ∩ N = Set.Ioo 0 1 := by sorry

end NUMINAMATH_CALUDE_M_intersect_N_eq_open_zero_one_l3697_369775


namespace NUMINAMATH_CALUDE_grandmas_will_l3697_369702

theorem grandmas_will (total : ℕ) (shelby_share : ℕ) (other_grandchildren : ℕ) (one_share : ℕ) :
  total = 124600 ∧
  shelby_share = total / 2 ∧
  other_grandchildren = 10 ∧
  one_share = 6230 ∧
  (total - shelby_share) / other_grandchildren = one_share →
  total = 124600 :=
by sorry

end NUMINAMATH_CALUDE_grandmas_will_l3697_369702


namespace NUMINAMATH_CALUDE_least_addition_for_divisibility_l3697_369735

theorem least_addition_for_divisibility (n m : ℕ) (h : n = 29989 ∧ m = 73) :
  ∃ x : ℕ, x = 21 ∧ 
    (∀ y : ℕ, (n + y) % m = 0 → y ≥ x) ∧
    (n + x) % m = 0 :=
  sorry

end NUMINAMATH_CALUDE_least_addition_for_divisibility_l3697_369735


namespace NUMINAMATH_CALUDE_binomial_18_choose_6_l3697_369743

theorem binomial_18_choose_6 : Nat.choose 18 6 = 13260 := by
  sorry

end NUMINAMATH_CALUDE_binomial_18_choose_6_l3697_369743


namespace NUMINAMATH_CALUDE_parents_selection_count_l3697_369778

def number_of_students : ℕ := 6
def number_of_parents : ℕ := 12
def parents_to_choose : ℕ := 4

theorem parents_selection_count : 
  (number_of_students.choose 1) * ((number_of_parents - 2).choose 1) * ((number_of_parents - 4).choose 1) = 480 :=
by sorry

end NUMINAMATH_CALUDE_parents_selection_count_l3697_369778


namespace NUMINAMATH_CALUDE_four_digit_sum_with_reverse_l3697_369742

/-- Represents a four-digit number -/
def FourDigitNumber := { n : ℕ // 1000 ≤ n ∧ n < 10000 }

/-- Returns the reversed digits of a four-digit number -/
def reverseDigits (x : FourDigitNumber) : FourDigitNumber :=
  sorry

/-- The sum of a number and its reverse -/
def sumWithReverse (x : FourDigitNumber) : ℕ :=
  x.val + (reverseDigits x).val

theorem four_digit_sum_with_reverse (x : FourDigitNumber) :
  x.val % 10 ≠ 0 →
  (sumWithReverse x) % 100 = 0 →
  sumWithReverse x = 11000 := by
  sorry

end NUMINAMATH_CALUDE_four_digit_sum_with_reverse_l3697_369742


namespace NUMINAMATH_CALUDE_dave_derek_money_difference_l3697_369717

theorem dave_derek_money_difference :
  let derek_initial : ℕ := 40
  let derek_lunch1 : ℕ := 14
  let derek_dad_lunch : ℕ := 11
  let derek_lunch2 : ℕ := 5
  let dave_initial : ℕ := 50
  let dave_mom_lunch : ℕ := 7
  let derek_remaining : ℕ := derek_initial - derek_lunch1 - derek_dad_lunch - derek_lunch2
  let dave_remaining : ℕ := dave_initial - dave_mom_lunch
  dave_remaining - derek_remaining = 33 :=
by sorry

end NUMINAMATH_CALUDE_dave_derek_money_difference_l3697_369717


namespace NUMINAMATH_CALUDE_inequality_solution_l3697_369707

theorem inequality_solution (a : ℝ) :
  (∀ x : ℝ, x > 0 → (Real.log x / Real.log a) + |a + (Real.log x / Real.log a)| * (Real.log a / Real.log (Real.sqrt x)) ≥ a * (Real.log a / Real.log x)) ↔
  -1/3 ≤ a ∧ a ≤ 1 :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l3697_369707


namespace NUMINAMATH_CALUDE_grocery_shop_sales_l3697_369779

theorem grocery_shop_sales (sales1 sales2 sales3 sales4 sales6 average_sale : ℕ)
  (h1 : sales1 = 6735)
  (h2 : sales2 = 6927)
  (h3 : sales3 = 6855)
  (h4 : sales4 = 7230)
  (h5 : sales6 = 4691)
  (h6 : average_sale = 6500) :
  ∃ sales5 : ℕ, sales5 = 6562 ∧
  (sales1 + sales2 + sales3 + sales4 + sales5 + sales6) / 6 = average_sale := by
  sorry

end NUMINAMATH_CALUDE_grocery_shop_sales_l3697_369779


namespace NUMINAMATH_CALUDE_marks_age_multiple_l3697_369785

theorem marks_age_multiple (mark_current_age : ℕ) (aaron_current_age : ℕ) : 
  mark_current_age = 28 →
  mark_current_age - 3 = 3 * (aaron_current_age - 3) + 1 →
  ∃ x : ℕ, mark_current_age + 4 = x * (aaron_current_age + 4) + 2 →
  x = 2 := by
sorry

end NUMINAMATH_CALUDE_marks_age_multiple_l3697_369785


namespace NUMINAMATH_CALUDE_number_percentage_equality_l3697_369757

theorem number_percentage_equality (x : ℚ) : 
  (35 / 100) * x = (15 / 100) * 40 → x = 17 + 1 / 7 := by
  sorry

end NUMINAMATH_CALUDE_number_percentage_equality_l3697_369757


namespace NUMINAMATH_CALUDE_unique_mythical_with_most_divisors_l3697_369776

def is_prime (p : ℕ) : Prop := p > 1 ∧ ∀ m : ℕ, m ∣ p → m = 1 ∨ m = p

def is_mythical (n : ℕ) : Prop :=
  n > 0 ∧ ∀ d : ℕ, d ∣ n → ∃ p : ℕ, is_prime p ∧ d = p - 2

def number_of_divisors (n : ℕ) : ℕ :=
  (Finset.filter (· ∣ n) (Finset.range (n + 1))).card

theorem unique_mythical_with_most_divisors :
  is_mythical 135 ∧
  ∀ n : ℕ, is_mythical n → number_of_divisors n ≤ number_of_divisors 135 ∧
  (number_of_divisors n = number_of_divisors 135 → n = 135) :=
sorry

end NUMINAMATH_CALUDE_unique_mythical_with_most_divisors_l3697_369776


namespace NUMINAMATH_CALUDE_final_price_percentage_l3697_369733

-- Define the discounts and tax rate
def discount1 : ℝ := 0.5
def discount2 : ℝ := 0.1
def discount3 : ℝ := 0.2
def taxRate : ℝ := 0.08

-- Define the function to calculate the final price
def finalPrice (originalPrice : ℝ) : ℝ :=
  let price1 := originalPrice * (1 - discount1)
  let price2 := price1 * (1 - discount2)
  let price3 := price2 * (1 - discount3)
  price3 * (1 + taxRate)

-- Theorem statement
theorem final_price_percentage (originalPrice : ℝ) (originalPrice_pos : originalPrice > 0) :
  finalPrice originalPrice / originalPrice = 0.3888 := by
  sorry

end NUMINAMATH_CALUDE_final_price_percentage_l3697_369733


namespace NUMINAMATH_CALUDE_triangle_angle_calculation_l3697_369794

theorem triangle_angle_calculation (A B C : Real) (a b c : Real) :
  -- Triangle ABC with sides a, b, c opposite to angles A, B, C
  (A + B + C = π) →
  (a * Real.cos B - b * Real.cos A = b) →
  (C = π / 5) →
  (B = 4 * π / 15) :=
by sorry

end NUMINAMATH_CALUDE_triangle_angle_calculation_l3697_369794


namespace NUMINAMATH_CALUDE_work_completion_time_l3697_369798

theorem work_completion_time (W : ℝ) (W_p W_q W_r : ℝ) :
  W_p = W_q + W_r →                -- p can do the work in the same time as q and r together
  W_p + W_q = W / 10 →             -- p and q together can complete the work in 10 days
  W_r = W / 35 →                   -- r alone needs 35 days to complete the work
  W_q = W / 28                     -- q alone needs 28 days to complete the work
  := by sorry

end NUMINAMATH_CALUDE_work_completion_time_l3697_369798


namespace NUMINAMATH_CALUDE_pump_water_in_35_minutes_l3697_369713

theorem pump_water_in_35_minutes : 
  let pump_rate : ℚ := 300  -- gallons per hour
  let time : ℚ := 35 / 60   -- 35 minutes converted to hours
  pump_rate * time = 175
  := by sorry

end NUMINAMATH_CALUDE_pump_water_in_35_minutes_l3697_369713


namespace NUMINAMATH_CALUDE_one_dime_in_collection_l3697_369714

/-- Represents the types of coins --/
inductive CoinType
  | Penny
  | Nickel
  | Dime
  | Quarter

/-- The value of each coin type in cents --/
def coinValue : CoinType → Nat
  | CoinType.Penny => 1
  | CoinType.Nickel => 5
  | CoinType.Dime => 10
  | CoinType.Quarter => 25

/-- A collection of coins --/
structure CoinCollection where
  pennies : Nat
  nickels : Nat
  dimes : Nat
  quarters : Nat

/-- Calculate the total value of a coin collection in cents --/
def totalValue (coins : CoinCollection) : Nat :=
  coins.pennies * coinValue CoinType.Penny +
  coins.nickels * coinValue CoinType.Nickel +
  coins.dimes * coinValue CoinType.Dime +
  coins.quarters * coinValue CoinType.Quarter

/-- The main theorem --/
theorem one_dime_in_collection :
  ∀ (coins : CoinCollection),
    totalValue coins = 102 ∧
    coins.pennies + coins.nickels + coins.dimes + coins.quarters = 9 ∧
    coins.pennies ≥ 1 ∧ coins.nickels ≥ 1 ∧ coins.dimes ≥ 1 ∧ coins.quarters ≥ 1
    → coins.dimes = 1 := by
  sorry

end NUMINAMATH_CALUDE_one_dime_in_collection_l3697_369714


namespace NUMINAMATH_CALUDE_wooden_easel_cost_l3697_369758

theorem wooden_easel_cost (paintbrush_cost paint_cost albert_has additional_needed : ℚ)
  (h1 : paintbrush_cost = 1.5)
  (h2 : paint_cost = 4.35)
  (h3 : albert_has = 6.5)
  (h4 : additional_needed = 12) :
  let total_cost := albert_has + additional_needed
  let other_items_cost := paintbrush_cost + paint_cost
  let easel_cost := total_cost - other_items_cost
  easel_cost = 12.65 := by sorry

end NUMINAMATH_CALUDE_wooden_easel_cost_l3697_369758


namespace NUMINAMATH_CALUDE_triangle_area_from_parametric_lines_l3697_369769

/-- The area of a triangle formed by two points on given lines and the origin -/
theorem triangle_area_from_parametric_lines (t s : ℝ) : 
  let l : ℝ × ℝ → Prop := λ p => ∃ t, p.1 = 3 + 5*t ∧ p.2 = 2 + 4*t
  let m : ℝ × ℝ → Prop := λ p => ∃ s, p.1 = 2 + 5*s ∧ p.2 = 3 + 4*s
  let C : ℝ × ℝ := (3 + 5*t, 2 + 4*t)
  let D : ℝ × ℝ := (2 + 5*s, 3 + 4*s)
  let O : ℝ × ℝ := (0, 0)
  l C → m D → 
  (1/2 : ℝ) * |5 + 2*s + 7*t| = 
  (1/2 : ℝ) * |C.1 * D.2 - C.2 * D.1| :=
by sorry

end NUMINAMATH_CALUDE_triangle_area_from_parametric_lines_l3697_369769


namespace NUMINAMATH_CALUDE_min_value_and_inequality_l3697_369748

-- Define the function f(x)
def f (x : ℝ) : ℝ := |x + 3| + |x - 1|

-- Theorem statement
theorem min_value_and_inequality :
  (∃ m : ℝ, ∀ x : ℝ, f x ≥ m ∧ ∃ x₀ : ℝ, f x₀ = m) ∧
  (∀ a b : ℝ, a > 0 → b > 0 → a + b = 4 → 1/a + 4/b ≥ 9/4) := by
  sorry

end NUMINAMATH_CALUDE_min_value_and_inequality_l3697_369748


namespace NUMINAMATH_CALUDE_largest_fraction_l3697_369762

theorem largest_fraction : 
  (26 : ℚ) / 51 > 101 / 203 ∧ 
  (26 : ℚ) / 51 > 47 / 93 ∧ 
  (26 : ℚ) / 51 > 5 / 11 ∧ 
  (26 : ℚ) / 51 > 199 / 401 := by
  sorry

end NUMINAMATH_CALUDE_largest_fraction_l3697_369762


namespace NUMINAMATH_CALUDE_min_value_x_plus_y_l3697_369754

theorem min_value_x_plus_y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : (x - 1) * (y + 1) = 16) :
  x + y ≥ 8 ∧ ∃ x₀ y₀ : ℝ, x₀ > 0 ∧ y₀ > 0 ∧ (x₀ - 1) * (y₀ + 1) = 16 ∧ x₀ + y₀ = 8 := by
  sorry

end NUMINAMATH_CALUDE_min_value_x_plus_y_l3697_369754


namespace NUMINAMATH_CALUDE_sphere_volume_ratio_l3697_369721

theorem sphere_volume_ratio (r₁ r₂ : ℝ) (h : 4 * Real.pi * r₁^2 / (4 * Real.pi * r₂^2) = 1 / 9) :
  (4 / 3) * Real.pi * r₁^3 / ((4 / 3) * Real.pi * r₂^3) = 1 / 27 := by
sorry

end NUMINAMATH_CALUDE_sphere_volume_ratio_l3697_369721


namespace NUMINAMATH_CALUDE_equation_solution_l3697_369704

theorem equation_solution : 
  {x : ℝ | (12 - 3*x)^2 = x^2} = {3, 6} := by sorry

end NUMINAMATH_CALUDE_equation_solution_l3697_369704


namespace NUMINAMATH_CALUDE_month_days_l3697_369761

theorem month_days (days_took_capsules days_forgot_capsules : ℕ) 
  (h1 : days_took_capsules = 29)
  (h2 : days_forgot_capsules = 2) : 
  days_took_capsules + days_forgot_capsules = 31 := by
sorry

end NUMINAMATH_CALUDE_month_days_l3697_369761


namespace NUMINAMATH_CALUDE_power_equation_solutions_l3697_369731

theorem power_equation_solutions :
  ∀ a b c : ℕ, 2^a * 3^b = 7^c - 1 ↔ (a = 1 ∧ b = 1 ∧ c = 1) ∨ (a = 4 ∧ b = 1 ∧ c = 2) :=
by sorry

end NUMINAMATH_CALUDE_power_equation_solutions_l3697_369731


namespace NUMINAMATH_CALUDE_systematic_sampling_smallest_number_l3697_369777

/-- Systematic sampling theorem for a specific case -/
theorem systematic_sampling_smallest_number
  (total_items : ℕ)
  (sample_size : ℕ)
  (highest_drawn : ℕ)
  (h1 : total_items = 32)
  (h2 : sample_size = 8)
  (h3 : highest_drawn = 31)
  (h4 : highest_drawn ≤ total_items)
  : ∃ (smallest_drawn : ℕ),
    smallest_drawn = 3 ∧
    smallest_drawn > 0 ∧
    smallest_drawn ≤ highest_drawn ∧
    (highest_drawn - smallest_drawn) % (total_items / sample_size) = 0 :=
by
  sorry


end NUMINAMATH_CALUDE_systematic_sampling_smallest_number_l3697_369777


namespace NUMINAMATH_CALUDE_solve_system_for_b_l3697_369749

theorem solve_system_for_b :
  ∀ (x y b : ℝ),
  (4 * x + 2 * y = b) →
  (3 * x + 4 * y = 3 * b) →
  (x = 3) →
  (b = -15) := by
sorry

end NUMINAMATH_CALUDE_solve_system_for_b_l3697_369749


namespace NUMINAMATH_CALUDE_min_draws_to_ensure_target_l3697_369711

/-- The number of distinct labels -/
def n : ℕ := 50

/-- The total number of records -/
def total_records : ℕ := n * (n + 1) / 2

/-- The number of records we want to ensure for a single label -/
def target : ℕ := 10

/-- The function that calculates the minimum number of records to draw -/
def min_draws : ℕ := 
  (target - 1) * (n - (target - 1)) + (target - 1) * target / 2

/-- Theorem stating the minimum number of records to draw -/
theorem min_draws_to_ensure_target : 
  min_draws = 415 :=
sorry

end NUMINAMATH_CALUDE_min_draws_to_ensure_target_l3697_369711


namespace NUMINAMATH_CALUDE_factorial_inequality_l3697_369795

theorem factorial_inequality (n : ℕ) (h : n ≥ 2) :
  2 * Real.log (Nat.factorial n) > (n^2 - 2*n + 1) / n := by
  sorry

end NUMINAMATH_CALUDE_factorial_inequality_l3697_369795


namespace NUMINAMATH_CALUDE_min_value_theorem_l3697_369715

theorem min_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∀ x y : ℝ, a * x + b * y - 2 = 0 → x^2 + y^2 - 6*x - 4*y - 12 = 0) →
  (∃ x y : ℝ, a * x + b * y - 2 = 0 ∧ x^2 + y^2 - 6*x - 4*y - 12 = 0) →
  (∀ a' b' : ℝ, a' > 0 → b' > 0 → 
    (∀ x y : ℝ, a' * x + b' * y - 2 = 0 → x^2 + y^2 - 6*x - 4*y - 12 = 0) →
    (∃ x y : ℝ, a' * x + b' * y - 2 = 0 ∧ x^2 + y^2 - 6*x - 4*y - 12 = 0) →
    3/a + 2/b ≤ 3/a' + 2/b') →
  3/a + 2/b = 25/2 := by
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l3697_369715


namespace NUMINAMATH_CALUDE_unique_point_equal_angles_l3697_369706

-- Define the ellipse
def is_on_ellipse (x y : ℝ) : Prop := x^2 / 2 + y^2 = 1

-- Define the focus
def F : ℝ × ℝ := (1, 0)

-- Define the point P
def P : ℝ × ℝ := (2, 0)

-- Define a chord passing through F
def is_chord_through_F (A B : ℝ × ℝ) : Prop :=
  is_on_ellipse A.1 A.2 ∧ is_on_ellipse B.1 B.2 ∧ 
  ∃ t : ℝ, (1 - t) • A + t • B = F

-- Define equality of angles APF and BPF
def angles_equal (A B : ℝ × ℝ) : Prop :=
  (A.2 / (A.1 - P.1)) + (B.2 / (B.1 - P.1)) = 0

-- Theorem statement
theorem unique_point_equal_angles :
  ∀ A B : ℝ × ℝ, is_chord_through_F A B → angles_equal A B ∧
  ∀ p : ℝ, p > 0 ∧ p ≠ 2 → ∃ A' B' : ℝ × ℝ, is_chord_through_F A' B' ∧ ¬angles_equal A' B' :=
sorry

end NUMINAMATH_CALUDE_unique_point_equal_angles_l3697_369706


namespace NUMINAMATH_CALUDE_smallest_number_is_three_l3697_369705

/-- Represents the systematic sampling of classes -/
structure ClassSampling where
  total_classes : Nat
  selected_classes : Nat
  sum_of_selected : Nat

/-- Calculates the smallest number in the systematic sample -/
def smallest_number (sampling : ClassSampling) : Nat :=
  let interval := sampling.total_classes / sampling.selected_classes
  (sampling.sum_of_selected - (interval * (sampling.selected_classes - 1) * sampling.selected_classes / 2)) / sampling.selected_classes

/-- Theorem: The smallest number in the given systematic sample is 3 -/
theorem smallest_number_is_three (sampling : ClassSampling) 
  (h1 : sampling.total_classes = 30)
  (h2 : sampling.selected_classes = 5)
  (h3 : sampling.sum_of_selected = 75) :
  smallest_number sampling = 3 := by
  sorry

#eval smallest_number { total_classes := 30, selected_classes := 5, sum_of_selected := 75 }

end NUMINAMATH_CALUDE_smallest_number_is_three_l3697_369705


namespace NUMINAMATH_CALUDE_valid_bases_for_346_l3697_369799

def is_valid_base (b : ℕ) : Prop :=
  b > 1 ∧ b^3 ≤ 346 ∧ 346 < b^4 ∧
  ∃ (d₃ d₂ d₁ d₀ : ℕ), 
    d₃ * b^3 + d₂ * b^2 + d₁ * b^1 + d₀ * b^0 = 346 ∧
    d₃ ≠ 0 ∧ d₀ % 2 = 0

theorem valid_bases_for_346 :
  ∀ b : ℕ, is_valid_base b ↔ (b = 6 ∨ b = 7) :=
sorry

end NUMINAMATH_CALUDE_valid_bases_for_346_l3697_369799


namespace NUMINAMATH_CALUDE_problem_solution_l3697_369771

/-- Given f(x) = ax^2 + bx where a ≠ 0 and f(2) = 0 -/
def f (a b x : ℝ) : ℝ := a * x^2 + b * x

theorem problem_solution (a b : ℝ) (ha : a ≠ 0) :
  (f a b 2 = 0) →
  /- Part I -/
  (∃! x, f a b x - x = 0) →
  (∀ x, f a b x = -1/2 * x^2 + x) ∧
  /- Part II -/
  (a = 1 →
    (∀ x ∈ Set.Icc (-1) 2, f 1 b x ≤ 3) ∧
    (∀ x ∈ Set.Icc (-1) 2, f 1 b x ≥ -1) ∧
    (∃ x ∈ Set.Icc (-1) 2, f 1 b x = 3) ∧
    (∃ x ∈ Set.Icc (-1) 2, f 1 b x = -1)) ∧
  /- Part III -/
  ((∀ x ≥ 2, f a b x ≥ 2 - a) → a ≥ 2) := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3697_369771


namespace NUMINAMATH_CALUDE_car_speeds_problem_l3697_369722

/-- Proves that given the problem conditions, the speeds of the two cars are 60 km/h and 90 km/h -/
theorem car_speeds_problem (total_distance : ℝ) (meeting_distance : ℝ) (speed_difference : ℝ)
  (h1 : total_distance = 200)
  (h2 : meeting_distance = 80)
  (h3 : speed_difference = 30)
  (h4 : meeting_distance / speed_a = (total_distance - meeting_distance) / (speed_a + speed_difference))
  (speed_a : ℝ)
  (speed_b : ℝ)
  (h5 : speed_b = speed_a + speed_difference) :
  speed_a = 60 ∧ speed_b = 90 :=
by
  sorry

end NUMINAMATH_CALUDE_car_speeds_problem_l3697_369722


namespace NUMINAMATH_CALUDE_correct_factorization_l3697_369759

theorem correct_factorization (a b : ℝ) : a * (a - b) - b * (b - a) = (a - b) * (a + b) := by
  sorry

end NUMINAMATH_CALUDE_correct_factorization_l3697_369759


namespace NUMINAMATH_CALUDE_factor_z4_minus_81_l3697_369752

theorem factor_z4_minus_81 (z : ℂ) : 
  z^4 - 81 = (z - 3) * (z + 3) * (z^2 + 9) := by sorry

end NUMINAMATH_CALUDE_factor_z4_minus_81_l3697_369752


namespace NUMINAMATH_CALUDE_manufacturer_profit_percentage_l3697_369729

-- Define the given values
def customer_price : ℝ := 30.09
def retailer_profit_percent : ℝ := 25
def wholesaler_profit_percent : ℝ := 20
def manufacturer_cost : ℝ := 17

-- Define the theorem
theorem manufacturer_profit_percentage :
  let retailer_cost := customer_price / (1 + retailer_profit_percent / 100)
  let wholesaler_price := retailer_cost
  let wholesaler_cost := wholesaler_price / (1 + wholesaler_profit_percent / 100)
  let manufacturer_price := wholesaler_cost
  let manufacturer_profit := manufacturer_price - manufacturer_cost
  manufacturer_profit / manufacturer_cost * 100 = 18 := by
sorry

end NUMINAMATH_CALUDE_manufacturer_profit_percentage_l3697_369729


namespace NUMINAMATH_CALUDE_smallest_block_size_l3697_369792

/-- Represents the dimensions of a rectangular block. -/
structure BlockDimensions where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Calculates the total number of cubes in a block given its dimensions. -/
def totalCubes (d : BlockDimensions) : ℕ :=
  d.length * d.width * d.height

/-- Calculates the number of invisible cubes when three faces are visible. -/
def invisibleCubes (d : BlockDimensions) : ℕ :=
  (d.length - 1) * (d.width - 1) * (d.height - 1)

/-- Checks if the given dimensions satisfy the problem conditions. -/
def isValidBlock (d : BlockDimensions) : Prop :=
  invisibleCubes d = 300 ∧ d.length > 1 ∧ d.width > 1 ∧ d.height > 1

/-- Theorem stating that the smallest possible number of cubes is 462. -/
theorem smallest_block_size :
  ∃ (d : BlockDimensions), isValidBlock d ∧ totalCubes d = 462 ∧
  (∀ (d' : BlockDimensions), isValidBlock d' → totalCubes d' ≥ 462) :=
sorry

end NUMINAMATH_CALUDE_smallest_block_size_l3697_369792


namespace NUMINAMATH_CALUDE_x_cubed_plus_inverse_l3697_369710

theorem x_cubed_plus_inverse (x : ℝ) (h : 47 = x^6 + 1/x^6) : x^3 + 1/x^3 = 7 := by
  sorry

end NUMINAMATH_CALUDE_x_cubed_plus_inverse_l3697_369710


namespace NUMINAMATH_CALUDE_complex_square_root_l3697_369784

theorem complex_square_root (p q : ℕ+) (h : (p + q * Complex.I) ^ 2 = 7 + 24 * Complex.I) :
  p + q * Complex.I = 4 + 3 * Complex.I :=
by sorry

end NUMINAMATH_CALUDE_complex_square_root_l3697_369784


namespace NUMINAMATH_CALUDE_second_vessel_capacity_l3697_369786

/-- Proves that the capacity of the second vessel is 6 liters given the problem conditions -/
theorem second_vessel_capacity : 
  ∀ (vessel2_capacity : ℝ),
    -- Given conditions
    let vessel1_capacity : ℝ := 2
    let vessel1_concentration : ℝ := 0.25
    let vessel2_concentration : ℝ := 0.40
    let total_liquid : ℝ := 8
    let final_vessel_capacity : ℝ := 10
    let final_concentration : ℝ := 0.29000000000000004

    -- Total liquid equation
    vessel1_capacity + vessel2_capacity = total_liquid →
    
    -- Alcohol balance equation
    (vessel1_capacity * vessel1_concentration + 
     vessel2_capacity * vessel2_concentration) / final_vessel_capacity = final_concentration →
    
    -- Conclusion
    vessel2_capacity = 6 := by
  sorry

end NUMINAMATH_CALUDE_second_vessel_capacity_l3697_369786


namespace NUMINAMATH_CALUDE_triangle_angle_measure_l3697_369726

theorem triangle_angle_measure (A B C : ℝ) (h1 : A = 3 * Real.pi / 4) (h2 : C > 0) (h3 : C < Real.pi / 4) (h4 : Real.sin C = 1 / 2) : C = Real.pi / 6 := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_measure_l3697_369726


namespace NUMINAMATH_CALUDE_candy_bar_weight_reduction_l3697_369768

theorem candy_bar_weight_reduction 
  (original_weight : ℝ) 
  (original_price : ℝ) 
  (new_weight : ℝ) 
  (h1 : original_weight > 0) 
  (h2 : original_price > 0) 
  (h3 : new_weight > 0) 
  (h4 : new_weight < original_weight) 
  (h5 : original_price / new_weight = (1 + 1/3) * (original_price / original_weight)) :
  (original_weight - new_weight) / original_weight = 1/4 := by
sorry

end NUMINAMATH_CALUDE_candy_bar_weight_reduction_l3697_369768


namespace NUMINAMATH_CALUDE_dime_difference_l3697_369753

/-- Represents the number of each type of coin in the piggy bank -/
structure CoinCount where
  nickels : ℕ
  dimes : ℕ
  quarters : ℕ
  half_dollars : ℕ

/-- The total number of coins in the piggy bank -/
def total_coins : ℕ := 120

/-- The total value of coins in cents -/
def total_value : ℕ := 1240

/-- Calculates the total number of coins for a given CoinCount -/
def count_coins (c : CoinCount) : ℕ :=
  c.nickels + c.dimes + c.quarters + c.half_dollars

/-- Calculates the total value in cents for a given CoinCount -/
def calculate_value (c : CoinCount) : ℕ :=
  5 * c.nickels + 10 * c.dimes + 25 * c.quarters + 50 * c.half_dollars

/-- Defines a valid CoinCount that satisfies the problem conditions -/
def is_valid_count (c : CoinCount) : Prop :=
  count_coins c = total_coins ∧ calculate_value c = total_value

/-- Finds the maximum number of dimes possible -/
def max_dimes : ℕ := 128

/-- Finds the minimum number of dimes possible -/
def min_dimes : ℕ := 2

theorem dime_difference :
  ∃ (max min : CoinCount),
    is_valid_count max ∧
    is_valid_count min ∧
    max.dimes = max_dimes ∧
    min.dimes = min_dimes ∧
    max_dimes - min_dimes = 126 := by
  sorry

end NUMINAMATH_CALUDE_dime_difference_l3697_369753


namespace NUMINAMATH_CALUDE_cistern_emptying_time_l3697_369760

/-- Given a cistern with normal fill time and leak-affected fill time, 
    calculate the time to empty through the leak. -/
theorem cistern_emptying_time 
  (normal_fill_time : ℝ) 
  (leak_fill_time : ℝ) 
  (h1 : normal_fill_time = 2) 
  (h2 : leak_fill_time = 4) : 
  (1 / (1 / normal_fill_time - 1 / leak_fill_time)) = 4 := by
  sorry

#check cistern_emptying_time

end NUMINAMATH_CALUDE_cistern_emptying_time_l3697_369760


namespace NUMINAMATH_CALUDE_race_length_is_90_l3697_369767

/-- The race between Nicky and Cristina -/
structure Race where
  head_start : ℝ
  cristina_speed : ℝ
  nicky_speed : ℝ
  catch_up_time : ℝ

/-- Calculate the length of the race -/
def race_length (r : Race) : ℝ :=
  r.nicky_speed * r.catch_up_time

/-- Theorem stating that the race length is 90 meters -/
theorem race_length_is_90 (r : Race)
  (h1 : r.head_start = 12)
  (h2 : r.cristina_speed = 5)
  (h3 : r.nicky_speed = 3)
  (h4 : r.catch_up_time = 30) :
  race_length r = 90 := by
  sorry

#check race_length_is_90

end NUMINAMATH_CALUDE_race_length_is_90_l3697_369767


namespace NUMINAMATH_CALUDE_m_range_l3697_369740

-- Define the function f
def f (m : ℝ) (x : ℝ) : ℝ := -x^2 + (m - 1) * x

-- State the theorem
theorem m_range :
  (∀ (x : ℝ), x^2 + 4*x - m ≥ 0) ∧
  (∀ (x y : ℝ), x < y → x ≤ -3 → y ≤ -3 → f m x ≤ f m y) →
  m ∈ Set.Icc (-5 : ℝ) (-4 : ℝ) :=
sorry

end NUMINAMATH_CALUDE_m_range_l3697_369740


namespace NUMINAMATH_CALUDE_min_value_theorem_l3697_369736

theorem min_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 1) (hab : a + b = 2) :
  (∀ x y : ℝ, x > 0 → y > 1 → x + y = 2 → 2/x + 1/(y-1) ≥ 2/a + 1/(b-1)) →
  2/a + 1/(b-1) = 3 + 2 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l3697_369736


namespace NUMINAMATH_CALUDE_divisibility_by_nine_l3697_369791

theorem divisibility_by_nine (D E : Nat) : 
  D ≤ 9 → E ≤ 9 → (D * 100000 + 864000 + E * 100 + 72) % 9 = 0 →
  (D + E = 0 ∨ D + E = 9 ∨ D + E = 18) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_by_nine_l3697_369791


namespace NUMINAMATH_CALUDE_valid_distributions_count_l3697_369701

/-- Represents a triangular array of 8 rows -/
def TriangularArray := Fin 8 → Fin 8 → ℕ

/-- The bottom row of the triangular array -/
def BottomRow := Fin 8 → Fin 2

/-- Checks if a number is a multiple of 5 -/
def IsMultipleOf5 (n : ℕ) : Prop := ∃ k, n = 5 * k

/-- Calculates the value of a square based on the two squares below it -/
def CalculateSquareValue (arr : TriangularArray) (row : Fin 8) (col : Fin 8) : ℕ :=
  if row = 0 then arr 0 col
  else arr (row - 1) col + arr (row - 1) (col + 1)

/-- Builds the triangular array from the bottom row -/
def BuildArray (bottom : BottomRow) : TriangularArray :=
  sorry

/-- Counts the number of valid bottom row distributions -/
def CountValidDistributions : ℕ :=
  sorry

/-- The main theorem stating that the count of valid distributions is 32 -/
theorem valid_distributions_count :
  CountValidDistributions = 32 :=
sorry

end NUMINAMATH_CALUDE_valid_distributions_count_l3697_369701


namespace NUMINAMATH_CALUDE_ball_color_distribution_l3697_369756

theorem ball_color_distribution (x y z : ℕ) : 
  x + y + z = 20 →
  x > 0 ∧ y > 0 ∧ z > 0 →
  (z : ℚ) / 20 - (2 * x : ℚ) / (2 * x + y + z) = 1 / 5 →
  x = 5 ∧ y = 3 ∧ z = 12 := by
sorry

end NUMINAMATH_CALUDE_ball_color_distribution_l3697_369756


namespace NUMINAMATH_CALUDE_tiles_needed_is_108_l3697_369730

/-- Represents the dimensions of a rectangular object -/
structure Dimensions where
  length : ℕ
  width : ℕ

/-- Calculates the area of a rectangular object given its dimensions -/
def area (d : Dimensions) : ℕ := d.length * d.width

/-- Converts feet to inches -/
def feet_to_inches (feet : ℕ) : ℕ := feet * 12

/-- The dimensions of a tile in inches -/
def tile : Dimensions := ⟨4, 6⟩

/-- The dimensions of the floor in feet -/
def floor : Dimensions := ⟨3, 6⟩

/-- The number of tiles needed to cover the floor -/
def tiles_needed : ℕ :=
  (area ⟨feet_to_inches floor.length, feet_to_inches floor.width⟩) / (area tile)

theorem tiles_needed_is_108 : tiles_needed = 108 := by
  sorry

#eval tiles_needed

end NUMINAMATH_CALUDE_tiles_needed_is_108_l3697_369730


namespace NUMINAMATH_CALUDE_johns_weekly_consumption_l3697_369727

/-- Represents John's daily beverage consumption --/
structure DailyConsumption where
  water : ℝ  -- in gallons
  milk : ℝ   -- in pints
  juice : ℝ  -- in fluid ounces

/-- Conversion factors --/
def gallon_to_quart : ℝ := 4
def pint_to_quart : ℝ := 0.5
def floz_to_quart : ℝ := 0.03125

/-- John's daily consumption --/
def johns_consumption : DailyConsumption := {
  water := 1.5,
  milk := 3,
  juice := 20
}

/-- Number of days in a week --/
def days_in_week : ℕ := 7

/-- Theorem stating John's weekly beverage consumption in quarts --/
theorem johns_weekly_consumption :
  (johns_consumption.water * gallon_to_quart +
   johns_consumption.milk * pint_to_quart +
   johns_consumption.juice * floz_to_quart) * days_in_week = 56.875 := by
  sorry

end NUMINAMATH_CALUDE_johns_weekly_consumption_l3697_369727


namespace NUMINAMATH_CALUDE_student_score_l3697_369783

theorem student_score (total_questions : Nat) (correct_responses : Nat) : 
  total_questions = 100 →
  correct_responses = 88 →
  let incorrect_responses := total_questions - correct_responses
  let score := correct_responses - 2 * incorrect_responses
  score = 64 := by
sorry

end NUMINAMATH_CALUDE_student_score_l3697_369783


namespace NUMINAMATH_CALUDE_library_reorganization_l3697_369725

theorem library_reorganization (initial_boxes : Nat) (books_per_initial_box : Nat) (books_per_new_box : Nat) : 
  initial_boxes = 2025 →
  books_per_initial_box = 25 →
  books_per_new_box = 28 →
  (initial_boxes * books_per_initial_box) % books_per_new_box = 21 := by
sorry

end NUMINAMATH_CALUDE_library_reorganization_l3697_369725


namespace NUMINAMATH_CALUDE_quadratic_expression_minimum_l3697_369712

theorem quadratic_expression_minimum :
  ∀ x y : ℝ, x^2 + 4*x*y + 5*y^2 - 8*x - 6*y ≥ -22 ∧
  ∃ x y : ℝ, x^2 + 4*x*y + 5*y^2 - 8*x - 6*y = -22 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_expression_minimum_l3697_369712


namespace NUMINAMATH_CALUDE_field_trip_girls_fraction_l3697_369766

theorem field_trip_girls_fraction (g : ℚ) (h1 : g > 0) : 
  let b := 2 * g
  let girls_on_trip := (4 / 5) * g
  let boys_on_trip := (3 / 4) * b
  let total_on_trip := girls_on_trip + boys_on_trip
  girls_on_trip / total_on_trip = 8 / 23 := by
sorry

end NUMINAMATH_CALUDE_field_trip_girls_fraction_l3697_369766


namespace NUMINAMATH_CALUDE_right_triangle_existence_l3697_369763

theorem right_triangle_existence (a b c d : ℕ+) 
  (h1 : a * b = c * d) 
  (h2 : a + b = c - d) : 
  ∃ x y z : ℕ+, x^2 + y^2 = z^2 ∧ (1/2 : ℚ) * x * y = a * b := by
sorry

end NUMINAMATH_CALUDE_right_triangle_existence_l3697_369763


namespace NUMINAMATH_CALUDE_triangle_angle_measure_l3697_369732

theorem triangle_angle_measure (a b c : ℝ) (A B C : ℝ) :
  a > 0 ∧ b > 0 ∧ c > 0 →
  A > 0 ∧ B > 0 ∧ C > 0 →
  A + B + C = π →
  a * c = b^2 - a^2 →
  A = π / 6 →
  B = π / 3 := by
sorry

end NUMINAMATH_CALUDE_triangle_angle_measure_l3697_369732


namespace NUMINAMATH_CALUDE_chocolate_milk_remaining_l3697_369755

/-- The amount of chocolate milk remaining after drinking some on two consecutive days. -/
theorem chocolate_milk_remaining (initial : ℝ) (day1 : ℝ) (day2 : ℝ) (h1 : initial = 1.6) (h2 : day1 = 0.8) (h3 : day2 = 0.3) :
  initial - day1 - day2 = 0.5 := by
  sorry

end NUMINAMATH_CALUDE_chocolate_milk_remaining_l3697_369755


namespace NUMINAMATH_CALUDE_first_nonzero_digit_of_1_over_127_l3697_369734

theorem first_nonzero_digit_of_1_over_127 :
  ∃ (n : ℕ), n > 0 ∧ (1000 : ℚ) / 127 = 7 + n / 127 ∧ n < 127 :=
by sorry

end NUMINAMATH_CALUDE_first_nonzero_digit_of_1_over_127_l3697_369734


namespace NUMINAMATH_CALUDE_cos_2018pi_over_3_l3697_369719

theorem cos_2018pi_over_3 : Real.cos (2018 * Real.pi / 3) = -(1 / 2) := by sorry

end NUMINAMATH_CALUDE_cos_2018pi_over_3_l3697_369719


namespace NUMINAMATH_CALUDE_quadratic_root_value_l3697_369788

theorem quadratic_root_value (n : ℝ) : n^2 - 5*n + 4 = 0 → n^2 - 5*n = -4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_value_l3697_369788


namespace NUMINAMATH_CALUDE_part_one_part_two_l3697_369746

-- Define the propositions r(x) and s(x)
def r (m : ℝ) (x : ℝ) : Prop := Real.sin x + Real.cos x > m
def s (m : ℝ) (x : ℝ) : Prop := x^2 + m*x + 1 > 0

-- Part 1
theorem part_one (m : ℝ) : 
  (∀ x ∈ Set.Ioo (1/2 : ℝ) 2, s m x) → m > -2 :=
sorry

-- Part 2
theorem part_two (m : ℝ) :
  (∀ x : ℝ, (r m x ∧ ¬s m x) ∨ (¬r m x ∧ s m x)) →
  m ∈ Set.Iic (-2) ∪ Set.Ioc (-Real.sqrt 2) 2 :=
sorry

end NUMINAMATH_CALUDE_part_one_part_two_l3697_369746


namespace NUMINAMATH_CALUDE_inequality_solution_l3697_369780

noncomputable def solution_set (a : ℝ) : Set ℝ :=
  if a < -1 ∨ (0 < a ∧ a < 1) then
    {x | a < x ∧ x < 1/a}
  else if a = 1 ∨ a = -1 then
    ∅
  else if a > 1 ∨ (-1 < a ∧ a < 0) then
    {x | 1/a < x ∧ x < a}
  else
    ∅

theorem inequality_solution (a : ℝ) (h : a ≠ 0) :
  {x : ℝ | x^2 - (a + 1/a)*x + 1 < 0} = solution_set a :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l3697_369780


namespace NUMINAMATH_CALUDE_watermelon_juice_percentage_l3697_369744

def total_volume : ℝ := 120
def orange_juice_percentage : ℝ := 15
def grape_juice_volume : ℝ := 30

theorem watermelon_juice_percentage :
  let orange_juice_volume := total_volume * (orange_juice_percentage / 100)
  let watermelon_juice_volume := total_volume - orange_juice_volume - grape_juice_volume
  (watermelon_juice_volume / total_volume) * 100 = 60 := by
sorry

end NUMINAMATH_CALUDE_watermelon_juice_percentage_l3697_369744


namespace NUMINAMATH_CALUDE_magnitude_comparison_l3697_369796

theorem magnitude_comparison : 7^(0.3 : ℝ) > (0.3 : ℝ)^7 ∧ (0.3 : ℝ)^7 > Real.log 0.3 := by
  sorry

end NUMINAMATH_CALUDE_magnitude_comparison_l3697_369796


namespace NUMINAMATH_CALUDE_wire_cutting_l3697_369738

theorem wire_cutting (total_length : ℝ) (ratio : ℝ) (shorter_piece : ℝ) : 
  total_length = 140 ∧ 
  ratio = 2 / 5 ∧ 
  shorter_piece + (1 + ratio) * shorter_piece = total_length →
  shorter_piece = 40 := by
  sorry

end NUMINAMATH_CALUDE_wire_cutting_l3697_369738


namespace NUMINAMATH_CALUDE_a_less_equal_two_l3697_369745

-- Define the sets A and B
def A : Set ℝ := {x | x ≥ 3}
def B (a : ℝ) : Set ℝ := {x | 2 * a - x > 1}

-- State the theorem
theorem a_less_equal_two (a : ℝ) : A ∩ (Set.univ \ B a) = A → a ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_a_less_equal_two_l3697_369745


namespace NUMINAMATH_CALUDE_area_of_triangle_pqs_l3697_369737

/-- Represents a trapezoid PQRS -/
structure Trapezoid where
  pq : ℝ
  rs : ℝ
  area : ℝ

/-- Theorem: Given a trapezoid PQRS with an area of 20, where RS is three times the length of PQ,
    the area of triangle PQS is 5. -/
theorem area_of_triangle_pqs (t : Trapezoid) 
    (h1 : t.area = 20)
    (h2 : t.rs = 3 * t.pq) : 
    t.area / 4 = 5 := by
  sorry

end NUMINAMATH_CALUDE_area_of_triangle_pqs_l3697_369737
