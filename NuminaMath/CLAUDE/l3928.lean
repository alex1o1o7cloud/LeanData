import Mathlib

namespace NUMINAMATH_CALUDE_eleven_integer_chords_l3928_392844

/-- Represents a circle with a point P inside it -/
structure CircleWithPoint where
  radius : ℝ
  distance_to_p : ℝ

/-- Counts the number of integer-length chords containing P -/
def count_integer_chords (c : CircleWithPoint) : ℕ :=
  sorry

/-- The main theorem -/
theorem eleven_integer_chords :
  let c := CircleWithPoint.mk 17 12
  count_integer_chords c = 11 := by sorry

end NUMINAMATH_CALUDE_eleven_integer_chords_l3928_392844


namespace NUMINAMATH_CALUDE_red_bank_amount_when_equal_l3928_392891

/-- Proves that the amount in the red coin bank is 12,500 won when both banks have equal amounts -/
theorem red_bank_amount_when_equal (red_initial : ℕ) (yellow_initial : ℕ) 
  (red_daily : ℕ) (yellow_daily : ℕ) :
  red_initial = 8000 →
  yellow_initial = 5000 →
  red_daily = 300 →
  yellow_daily = 500 →
  ∃ d : ℕ, red_initial + d * red_daily = yellow_initial + d * yellow_daily ∧
          red_initial + d * red_daily = 12500 :=
by sorry

end NUMINAMATH_CALUDE_red_bank_amount_when_equal_l3928_392891


namespace NUMINAMATH_CALUDE_school_population_after_new_students_l3928_392801

theorem school_population_after_new_students (initial_avg_age : ℝ) (new_students : ℕ) 
  (new_students_avg_age : ℝ) (avg_age_decrease : ℝ) :
  initial_avg_age = 48 →
  new_students = 120 →
  new_students_avg_age = 32 →
  avg_age_decrease = 4 →
  ∃ (initial_students : ℕ),
    (initial_students + new_students : ℝ) * (initial_avg_age - avg_age_decrease) = 
    initial_students * initial_avg_age + new_students * new_students_avg_age ∧
    initial_students + new_students = 480 :=
by sorry

end NUMINAMATH_CALUDE_school_population_after_new_students_l3928_392801


namespace NUMINAMATH_CALUDE_remaining_pictures_l3928_392802

/-- The number of pictures Megan took at the zoo -/
def zoo_pictures : ℕ := 15

/-- The number of pictures Megan took at the museum -/
def museum_pictures : ℕ := 18

/-- The number of pictures Megan deleted -/
def deleted_pictures : ℕ := 31

/-- The theorem states that the number of pictures Megan still has from her vacation is 2 -/
theorem remaining_pictures :
  zoo_pictures + museum_pictures - deleted_pictures = 2 := by sorry

end NUMINAMATH_CALUDE_remaining_pictures_l3928_392802


namespace NUMINAMATH_CALUDE_volume_of_circumscribed_polyhedron_l3928_392830

/-- A polyhedron circumscribed around a sphere. -/
structure CircumscribedPolyhedron where
  -- The volume of the polyhedron
  volume : ℝ
  -- The radius of the inscribed sphere
  sphereRadius : ℝ
  -- The total surface area of the polyhedron
  surfaceArea : ℝ

/-- 
The volume of a polyhedron circumscribed around a sphere is equal to 
one-third of the product of the sphere's radius and the polyhedron's total surface area.
-/
theorem volume_of_circumscribed_polyhedron (p : CircumscribedPolyhedron) : 
  p.volume = (1 / 3) * p.sphereRadius * p.surfaceArea := by
  sorry

end NUMINAMATH_CALUDE_volume_of_circumscribed_polyhedron_l3928_392830


namespace NUMINAMATH_CALUDE_equation_solution_l3928_392806

theorem equation_solution : 
  ∀ x : ℝ, (x + 1) * (x + 3) = x + 1 ↔ x = -1 ∨ x = -2 := by
sorry

end NUMINAMATH_CALUDE_equation_solution_l3928_392806


namespace NUMINAMATH_CALUDE_floor_sum_example_l3928_392815

theorem floor_sum_example : ⌊(23.8 : ℝ)⌋ + ⌊(-23.8 : ℝ)⌋ = -1 := by
  sorry

end NUMINAMATH_CALUDE_floor_sum_example_l3928_392815


namespace NUMINAMATH_CALUDE_part_one_part_two_l3928_392800

-- Define the function f
def f (x : ℝ) : ℝ := x^2 - 2*x + 3

-- Theorem for the first part
theorem part_one (m : ℝ) : 
  (∀ x : ℝ, m + f x > 0) ↔ m > -2 :=
sorry

-- Theorem for the second part
theorem part_two (m : ℝ) :
  (∃ x : ℝ, m - f x > 0) ↔ m > 2 :=
sorry

end NUMINAMATH_CALUDE_part_one_part_two_l3928_392800


namespace NUMINAMATH_CALUDE_three_distinct_prime_factors_l3928_392839

theorem three_distinct_prime_factors (p q : ℕ) (hp : Prime p) (hq : Prime q) (h_order : q > p ∧ p > 2) :
  ∃ (a b c : ℕ), Prime a ∧ Prime b ∧ Prime c ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ (a * b * c ∣ 2^(p*q) - 1) :=
by sorry

end NUMINAMATH_CALUDE_three_distinct_prime_factors_l3928_392839


namespace NUMINAMATH_CALUDE_goldfish_count_l3928_392845

/-- The number of goldfish in the fish tank -/
def num_goldfish : ℕ := sorry

/-- The number of platyfish in the fish tank -/
def num_platyfish : ℕ := 10

/-- The number of red balls each goldfish plays with -/
def red_balls_per_goldfish : ℕ := 10

/-- The number of white balls each platyfish plays with -/
def white_balls_per_platyfish : ℕ := 5

/-- The total number of balls in the fish tank -/
def total_balls : ℕ := 80

theorem goldfish_count : num_goldfish = 3 := by
  have h1 : num_goldfish * red_balls_per_goldfish + num_platyfish * white_balls_per_platyfish = total_balls := sorry
  sorry

end NUMINAMATH_CALUDE_goldfish_count_l3928_392845


namespace NUMINAMATH_CALUDE_exponential_function_not_multiplicative_l3928_392898

theorem exponential_function_not_multiplicative : ¬∀ a b : ℝ, Real.exp (a * b) = Real.exp a * Real.exp b := by
  sorry

end NUMINAMATH_CALUDE_exponential_function_not_multiplicative_l3928_392898


namespace NUMINAMATH_CALUDE_cube_edge_length_range_l3928_392899

theorem cube_edge_length_range (V : ℝ) (a : ℝ) (h1 : V = 9) (h2 : V = a^3) :
  2 < a ∧ a < 2.5 := by
  sorry

end NUMINAMATH_CALUDE_cube_edge_length_range_l3928_392899


namespace NUMINAMATH_CALUDE_fixed_point_of_parabola_l3928_392871

theorem fixed_point_of_parabola (t : ℝ) : 
  5 * (3 : ℝ)^2 + t * 3 - 3 * t = 45 := by sorry

end NUMINAMATH_CALUDE_fixed_point_of_parabola_l3928_392871


namespace NUMINAMATH_CALUDE_l_shape_area_l3928_392804

theorem l_shape_area (a : ℝ) (h1 : a > 0) (h2 : 5 * a^2 = 4 * ((a + 3)^2 - a^2)) :
  (a + 3)^2 - a^2 = 45 := by
  sorry

end NUMINAMATH_CALUDE_l_shape_area_l3928_392804


namespace NUMINAMATH_CALUDE_prob_three_even_out_of_five_l3928_392803

-- Define a fair 6-sided die
def FairDie := Fin 6

-- Define the probability of rolling an even number on a single die
def probEven : ℚ := 1 / 2

-- Define the number of dice
def numDice : ℕ := 5

-- Define the number of dice we want to show even
def numEven : ℕ := 3

-- Theorem statement
theorem prob_three_even_out_of_five :
  (Nat.choose numDice numEven : ℚ) * probEven ^ numDice = 5 / 16 := by
  sorry

end NUMINAMATH_CALUDE_prob_three_even_out_of_five_l3928_392803


namespace NUMINAMATH_CALUDE_three_digit_sum_problem_l3928_392893

theorem three_digit_sum_problem (a b c : ℕ) : 
  a ≠ b ∧ b ≠ c ∧ a ≠ c →
  a < 10 ∧ b < 10 ∧ c < 10 →
  122 * a + 212 * b + 221 * c = 2003 →
  100 * a + 10 * b + c = 345 :=
by sorry

end NUMINAMATH_CALUDE_three_digit_sum_problem_l3928_392893


namespace NUMINAMATH_CALUDE_normal_distribution_probability_l3928_392897

/-- A random variable following a normal distribution with mean 2 and some variance σ² -/
noncomputable def ξ : Real → ℝ := sorry

/-- The probability density function of ξ -/
noncomputable def pdf_ξ : ℝ → ℝ := sorry

/-- The cumulative distribution function of ξ -/
noncomputable def cdf_ξ : ℝ → ℝ := sorry

/-- The condition that P(ξ < 4) = 0.8 -/
axiom cdf_at_4 : cdf_ξ 4 = 0.8

/-- The theorem to prove -/
theorem normal_distribution_probability :
  cdf_ξ 2 - cdf_ξ 0 = 0.3 := by sorry

end NUMINAMATH_CALUDE_normal_distribution_probability_l3928_392897


namespace NUMINAMATH_CALUDE_sequence_inequality_l3928_392838

theorem sequence_inequality (a : ℕ → ℝ) 
  (h_pos : ∀ n, a n > 0)
  (h_ineq : ∀ n, a (n + 2) ≤ (2023 * a n) / (a n * a (n + 1) + 2023)) :
  a 2023 < 1 ∨ a 2024 < 1 := by
sorry

end NUMINAMATH_CALUDE_sequence_inequality_l3928_392838


namespace NUMINAMATH_CALUDE_total_balls_theorem_l3928_392848

/-- The number of balls of wool used for a single item -/
def balls_per_item : String → ℕ
  | "scarf" => 3
  | "sweater" => 4
  | "hat" => 2
  | "mittens" => 1
  | _ => 0

/-- The number of items made by Aaron -/
def aaron_items : String → ℕ
  | "scarf" => 10
  | "sweater" => 5
  | "hat" => 6
  | _ => 0

/-- The number of items made by Enid -/
def enid_items : String → ℕ
  | "sweater" => 8
  | "hat" => 12
  | "mittens" => 4
  | _ => 0

/-- The total number of balls of wool used by both Enid and Aaron -/
def total_balls_used : ℕ := 
  (aaron_items "scarf" * balls_per_item "scarf") +
  (aaron_items "sweater" * balls_per_item "sweater") +
  (aaron_items "hat" * balls_per_item "hat") +
  (enid_items "sweater" * balls_per_item "sweater") +
  (enid_items "hat" * balls_per_item "hat") +
  (enid_items "mittens" * balls_per_item "mittens")

theorem total_balls_theorem : total_balls_used = 122 := by
  sorry

end NUMINAMATH_CALUDE_total_balls_theorem_l3928_392848


namespace NUMINAMATH_CALUDE_monotone_increasing_f_implies_a_range_l3928_392853

noncomputable def f (x : ℝ) : ℝ := (1/2) * x^2 + 2*x - 2 * Real.log x

theorem monotone_increasing_f_implies_a_range (h : Monotone f) :
  (∀ x > 0, 2 * a ≤ x^2 + 2*x) → a ≤ 0 := by sorry

end NUMINAMATH_CALUDE_monotone_increasing_f_implies_a_range_l3928_392853


namespace NUMINAMATH_CALUDE_oil_drilling_probability_l3928_392813

/-- The probability of hitting an oil layer when drilling in a sea area -/
theorem oil_drilling_probability 
  (total_area : ℝ) 
  (oil_area : ℝ) 
  (h1 : total_area = 10000) 
  (h2 : oil_area = 40) : 
  oil_area / total_area = 1 / 250 := by
sorry

end NUMINAMATH_CALUDE_oil_drilling_probability_l3928_392813


namespace NUMINAMATH_CALUDE_orange_boxes_needed_l3928_392889

/-- Calculates the number of boxes needed for oranges given the initial conditions --/
theorem orange_boxes_needed (baskets : ℕ) (oranges_per_basket : ℕ) (oranges_eaten : ℕ) (oranges_per_box : ℕ)
  (h1 : baskets = 7)
  (h2 : oranges_per_basket = 31)
  (h3 : oranges_eaten = 3)
  (h4 : oranges_per_box = 17) :
  (baskets * oranges_per_basket - oranges_eaten + oranges_per_box - 1) / oranges_per_box = 13 := by
  sorry

end NUMINAMATH_CALUDE_orange_boxes_needed_l3928_392889


namespace NUMINAMATH_CALUDE_cost_per_set_is_20_l3928_392878

/-- Represents the manufacturing and sales scenario for horseshoe sets -/
structure HorseshoeManufacturing where
  initialOutlay : ℕ
  sellingPrice : ℕ
  setsSold : ℕ
  profit : ℕ

/-- Calculates the cost per set given the manufacturing scenario -/
def costPerSet (h : HorseshoeManufacturing) : ℚ :=
  (h.sellingPrice * h.setsSold - h.profit - h.initialOutlay) / h.setsSold

/-- Theorem stating that the cost per set is $20 given the specific scenario -/
theorem cost_per_set_is_20 (h : HorseshoeManufacturing) 
  (h_initial : h.initialOutlay = 10000)
  (h_price : h.sellingPrice = 50)
  (h_sold : h.setsSold = 500)
  (h_profit : h.profit = 5000) :
  costPerSet h = 20 := by
  sorry

#eval costPerSet { initialOutlay := 10000, sellingPrice := 50, setsSold := 500, profit := 5000 }

end NUMINAMATH_CALUDE_cost_per_set_is_20_l3928_392878


namespace NUMINAMATH_CALUDE_hyperbola_asymptotes_l3928_392867

/-- The hyperbola and parabola equations -/
def hyperbola (x y b : ℝ) : Prop := x^2 / 4 - y^2 / b^2 = 1

def parabola (x y : ℝ) : Prop := y^2 = 8 * Real.sqrt 2 * x

/-- The right focus of the hyperbola coincides with the focus of the parabola -/
axiom focus_coincide : ∃ (x₀ y₀ : ℝ), 
  (x₀ = 2 * Real.sqrt 2 ∧ y₀ = 0) ∧
  (∀ x y b, hyperbola x y b → (x - x₀)^2 + y^2 = (2 * Real.sqrt 2)^2)

/-- The theorem stating that the asymptotes of the hyperbola are y = ±x -/
theorem hyperbola_asymptotes : 
  ∃ b, ∀ x y, hyperbola x y b → (y = x ∨ y = -x) ∨ (x^2 > y^2) := by sorry

end NUMINAMATH_CALUDE_hyperbola_asymptotes_l3928_392867


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l3928_392837

theorem imaginary_part_of_complex_fraction : Complex.im (5 * Complex.I / (1 + 2 * Complex.I)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l3928_392837


namespace NUMINAMATH_CALUDE_complex_number_properties_l3928_392862

theorem complex_number_properties (x y : ℝ) (h : (1 + Complex.I) * x + (1 - Complex.I) * y = 2) :
  let z := x + Complex.I * y
  (0 < z.re ∧ 0 < z.im) ∧ Complex.abs z = Real.sqrt 2 ∧ z.re = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_number_properties_l3928_392862


namespace NUMINAMATH_CALUDE_shorter_base_length_l3928_392883

/-- A trapezoid with given properties -/
structure Trapezoid where
  long_base : ℝ
  short_base : ℝ
  midline_length : ℝ

/-- The property that the line joining the midpoints of the diagonals 
    is half the difference of the bases -/
def midline_property (t : Trapezoid) : Prop :=
  t.midline_length = (t.long_base - t.short_base) / 2

/-- Theorem stating the length of the shorter base given the conditions -/
theorem shorter_base_length (t : Trapezoid) 
  (h1 : t.long_base = 115)
  (h2 : t.midline_length = 6)
  (h3 : midline_property t) : 
  t.short_base = 103 := by
sorry

end NUMINAMATH_CALUDE_shorter_base_length_l3928_392883


namespace NUMINAMATH_CALUDE_sum_of_divisors_is_96_l3928_392846

-- Define the property of n having exactly 8 divisors, including 1, n, 14, and 21
def has_eight_divisors_with_14_and_21 (n : ℕ) : Prop :=
  (∃ d : Finset ℕ, d.card = 8 ∧ 
    (∀ x, x ∈ d ↔ x ∣ n) ∧
    1 ∈ d ∧ n ∈ d ∧ 14 ∈ d ∧ 21 ∈ d)

-- Theorem stating that if n satisfies the above property, 
-- then the sum of its divisors is 96
theorem sum_of_divisors_is_96 (n : ℕ) 
  (h : has_eight_divisors_with_14_and_21 n) : 
  (Finset.filter (· ∣ n) (Finset.range (n + 1))).sum id = 96 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_divisors_is_96_l3928_392846


namespace NUMINAMATH_CALUDE_right_triangle_area_equality_l3928_392811

/-- Given a right triangle with sides a ≤ b ≤ c, perimeter 2p, and area S,
    prove that p(p-c) = (p-a)(p-b) = S -/
theorem right_triangle_area_equality
  (a b c p S : ℝ)
  (h_right_triangle : a^2 + b^2 = c^2)
  (h_order : a ≤ b ∧ b ≤ c)
  (h_perimeter : a + b + c = 2 * p)
  (h_area : S = (1/2) * a * b) :
  p * (p - c) = (p - a) * (p - b) ∧ p * (p - c) = S :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_area_equality_l3928_392811


namespace NUMINAMATH_CALUDE_m_range_l3928_392884

theorem m_range (m : ℝ) :
  (m^2 + m)^(3/5) ≤ (3 - m)^(3/5) → -3 ≤ m ∧ m ≤ 1 :=
by
  sorry

end NUMINAMATH_CALUDE_m_range_l3928_392884


namespace NUMINAMATH_CALUDE_greatest_root_of_g_l3928_392892

def g (x : ℝ) : ℝ := 20 * x^4 - 21 * x^2 + 5

theorem greatest_root_of_g :
  ∃ (r : ℝ), g r = 0 ∧ r = 1 ∧ ∀ (x : ℝ), g x = 0 → x ≤ r :=
by sorry

end NUMINAMATH_CALUDE_greatest_root_of_g_l3928_392892


namespace NUMINAMATH_CALUDE_disjoint_subsets_remainder_l3928_392851

def T : Finset Nat := Finset.range 15

def disjoint_subsets (S : Finset Nat) : Nat :=
  (3^S.card - 2 * 2^S.card + 1) / 2

theorem disjoint_subsets_remainder (T : Finset Nat) (h : T = Finset.range 15) :
  disjoint_subsets T % 1000 = 686 := by
  sorry

end NUMINAMATH_CALUDE_disjoint_subsets_remainder_l3928_392851


namespace NUMINAMATH_CALUDE_smallest_multiple_divisible_by_all_up_to_20_l3928_392876

/-- The smallest positive integer divisible by all numbers from 1 to 20 -/
def smallestMultiple : Nat := 232792560

/-- Checks if a number is divisible by all integers from 1 to 20 -/
def divisibleByAllUpTo20 (n : Nat) : Prop :=
  ∀ i : Nat, 1 ≤ i ∧ i ≤ 20 → n % i = 0

theorem smallest_multiple_divisible_by_all_up_to_20 :
  divisibleByAllUpTo20 smallestMultiple ∧
  ∀ n : Nat, n > 0 ∧ n < smallestMultiple → ¬(divisibleByAllUpTo20 n) := by
  sorry

#eval smallestMultiple

end NUMINAMATH_CALUDE_smallest_multiple_divisible_by_all_up_to_20_l3928_392876


namespace NUMINAMATH_CALUDE_odd_even_sum_theorem_l3928_392894

def is_odd_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

def is_even_function (g : ℝ → ℝ) : Prop := ∀ x, g (-x) = g x

theorem odd_even_sum_theorem (f g : ℝ → ℝ) 
  (h_odd : is_odd_function f) 
  (h_even : is_even_function g) 
  (h_diff : ∀ x, f x - g x = x^2 + 9*x + 12) : 
  ∀ x, f x + g x = -x^2 + 9*x - 12 :=
by sorry

end NUMINAMATH_CALUDE_odd_even_sum_theorem_l3928_392894


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l3928_392818

theorem sufficient_not_necessary_condition :
  ∃ (S : Set ℝ), 
    (∀ x ∈ S, x^2 - 4*x < 0) ∧ 
    (S ⊂ {x : ℝ | 0 < x ∧ x < 4}) ∧
    S = {x : ℝ | 0 < x ∧ x < 2} :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l3928_392818


namespace NUMINAMATH_CALUDE_function_equivalence_l3928_392807

theorem function_equivalence (x : ℝ) (h : x ≠ 0) :
  (2 * x + 3) / x = 2 + 3 / x := by sorry

end NUMINAMATH_CALUDE_function_equivalence_l3928_392807


namespace NUMINAMATH_CALUDE_distance_point_to_line_l3928_392861

/-- The distance between a point and a horizontal line is the absolute difference
    between their y-coordinates. -/
def distance_point_to_horizontal_line (point : ℝ × ℝ) (line_y : ℝ) : ℝ :=
  |point.2 - line_y|

/-- Theorem: The distance between the point (3, 0) and the line y = 1 is 1. -/
theorem distance_point_to_line : distance_point_to_horizontal_line (3, 0) 1 = 1 := by
  sorry

end NUMINAMATH_CALUDE_distance_point_to_line_l3928_392861


namespace NUMINAMATH_CALUDE_company_employee_increase_l3928_392870

theorem company_employee_increase (jan_employees dec_employees : ℝ) 
  (h_jan : jan_employees = 426.09)
  (h_dec : dec_employees = 490) :
  let increase := dec_employees - jan_employees
  let percentage_increase := (increase / jan_employees) * 100
  ∃ ε > 0, |percentage_increase - 15| < ε :=
by sorry

end NUMINAMATH_CALUDE_company_employee_increase_l3928_392870


namespace NUMINAMATH_CALUDE_divisible_by_24_l3928_392809

theorem divisible_by_24 (n : ℤ) : ∃ k : ℤ, n^4 + 6*n^3 + 11*n^2 + 6*n = 24*k := by
  sorry

end NUMINAMATH_CALUDE_divisible_by_24_l3928_392809


namespace NUMINAMATH_CALUDE_initial_average_calculation_l3928_392833

theorem initial_average_calculation (n : ℕ) (wrong_mark correct_mark : ℝ) (corrected_avg : ℝ) :
  n = 30 ∧ wrong_mark = 90 ∧ correct_mark = 15 ∧ corrected_avg = 57.5 →
  (n * corrected_avg + (wrong_mark - correct_mark)) / n = 60 := by
  sorry

end NUMINAMATH_CALUDE_initial_average_calculation_l3928_392833


namespace NUMINAMATH_CALUDE_minimum_value_implies_a_l3928_392826

def f (a x : ℝ) : ℝ := x^2 + 2*a*x + 2

theorem minimum_value_implies_a (a : ℝ) :
  (∀ x ∈ Set.Icc (-5 : ℝ) 5, f a x ≥ -3) ∧
  (∃ x ∈ Set.Icc (-5 : ℝ) 5, f a x = -3) →
  a = Real.sqrt 5 ∨ a = -Real.sqrt 5 :=
by sorry

end NUMINAMATH_CALUDE_minimum_value_implies_a_l3928_392826


namespace NUMINAMATH_CALUDE_pr_less_than_qr_implies_p_less_than_q_l3928_392890

theorem pr_less_than_qr_implies_p_less_than_q
  (r p q : ℝ) 
  (h1 : r < 0) 
  (h2 : p * q ≠ 0) 
  (h3 : p * r < q * r) : 
  p < q :=
by sorry

end NUMINAMATH_CALUDE_pr_less_than_qr_implies_p_less_than_q_l3928_392890


namespace NUMINAMATH_CALUDE_soccer_field_kids_l3928_392895

theorem soccer_field_kids (initial_kids joining_kids : ℕ) : 
  initial_kids = 14 → joining_kids = 22 → initial_kids + joining_kids = 36 := by
sorry

end NUMINAMATH_CALUDE_soccer_field_kids_l3928_392895


namespace NUMINAMATH_CALUDE_table_tennis_games_l3928_392854

theorem table_tennis_games (total_games : ℕ) 
  (petya_games : ℕ) (kolya_games : ℕ) (vasya_games : ℕ) : 
  petya_games = total_games / 2 →
  kolya_games = total_games / 3 →
  vasya_games = total_games / 5 →
  petya_games + kolya_games + vasya_games ≤ total_games →
  (∃ (games_between_petya_kolya : ℕ), 
    games_between_petya_kolya ≤ 1 ∧
    petya_games + kolya_games + vasya_games + games_between_petya_kolya = total_games) →
  total_games = 30 := by
sorry

end NUMINAMATH_CALUDE_table_tennis_games_l3928_392854


namespace NUMINAMATH_CALUDE_minimize_sum_of_number_and_square_l3928_392866

/-- The function representing the sum of a number and its square -/
def f (x : ℝ) : ℝ := x + x^2

/-- The theorem stating that -1/2 minimizes the function f -/
theorem minimize_sum_of_number_and_square :
  ∀ x : ℝ, f (-1/2) ≤ f x :=
by
  sorry

end NUMINAMATH_CALUDE_minimize_sum_of_number_and_square_l3928_392866


namespace NUMINAMATH_CALUDE_johns_friends_l3928_392880

theorem johns_friends (initial_amount : ℚ) (sweet_cost : ℚ) (friend_gift : ℚ) (final_amount : ℚ)
  (h1 : initial_amount = 20.1)
  (h2 : sweet_cost = 1.05)
  (h3 : friend_gift = 1)
  (h4 : final_amount = 17.05) :
  (initial_amount - sweet_cost - final_amount) / friend_gift = 2 := by
  sorry

end NUMINAMATH_CALUDE_johns_friends_l3928_392880


namespace NUMINAMATH_CALUDE_books_minus_figures_equals_two_l3928_392849

/-- The number of books on Jerry's shelf -/
def initial_books : ℕ := 7

/-- The initial number of action figures on Jerry's shelf -/
def initial_action_figures : ℕ := 3

/-- The number of action figures Jerry added later -/
def added_action_figures : ℕ := 2

/-- The total number of action figures after addition -/
def total_action_figures : ℕ := initial_action_figures + added_action_figures

theorem books_minus_figures_equals_two :
  initial_books - total_action_figures = 2 := by
  sorry

end NUMINAMATH_CALUDE_books_minus_figures_equals_two_l3928_392849


namespace NUMINAMATH_CALUDE_quadratic_form_decomposition_l3928_392875

theorem quadratic_form_decomposition (x y z : ℝ) : 
  x^2 + 2*x*y + 5*y^2 - 6*x*z - 22*y*z + 16*z^2 = 
  (x + (y - 3*z))^2 + (2*y - 4*z)^2 - (3*z)^2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_form_decomposition_l3928_392875


namespace NUMINAMATH_CALUDE_fraction_equality_l3928_392860

theorem fraction_equality : (2 : ℚ) / 5 - (1 : ℚ) / 7 = 1 / ((35 : ℚ) / 9) := by sorry

end NUMINAMATH_CALUDE_fraction_equality_l3928_392860


namespace NUMINAMATH_CALUDE_unique_two_digit_number_l3928_392831

theorem unique_two_digit_number : ∃! n : ℕ, 
  10 ≤ n ∧ n < 100 ∧ 
  (n / 10 ≠ n % 10) ∧ 
  n^2 = (n / 10 + n % 10)^3 := by
  sorry

end NUMINAMATH_CALUDE_unique_two_digit_number_l3928_392831


namespace NUMINAMATH_CALUDE_f_at_2_eq_neg_22_l3928_392820

/-- Given a function f(x) = x^5 - ax^3 + bx - 6 where f(-2) = 10, prove that f(2) = -22 -/
theorem f_at_2_eq_neg_22 (a b : ℝ) (f : ℝ → ℝ) 
    (h1 : ∀ x, f x = x^5 - a*x^3 + b*x - 6)
    (h2 : f (-2) = 10) : 
  f 2 = -22 := by sorry

end NUMINAMATH_CALUDE_f_at_2_eq_neg_22_l3928_392820


namespace NUMINAMATH_CALUDE_infinitely_many_divisible_by_15_l3928_392814

def v : ℕ → ℤ
  | 0 => 0
  | 1 => 1
  | (n + 2) => 8 * v (n + 1) - v n

theorem infinitely_many_divisible_by_15 :
  ∀ N : ℕ, ∃ k : ℕ, k > N ∧ 15 ∣ v (15 * k) :=
sorry

end NUMINAMATH_CALUDE_infinitely_many_divisible_by_15_l3928_392814


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l3928_392850

theorem arithmetic_sequence_common_difference
  (a : ℕ → ℤ) -- a is the arithmetic sequence
  (h1 : a 5 = 3) -- given condition: a_5 = 3
  (h2 : a 6 = -2) -- given condition: a_6 = -2
  : ∃ d : ℤ, d = -5 ∧ ∀ n : ℕ, a (n + 1) = a n + d :=
by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l3928_392850


namespace NUMINAMATH_CALUDE_min_value_reciprocal_sum_min_value_reciprocal_sum_equality_l3928_392864

theorem min_value_reciprocal_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  (1 / a + 2 / b) ≥ 3 + 2 * Real.sqrt 2 :=
by sorry

theorem min_value_reciprocal_sum_equality (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  (1 / a + 2 / b = 3 + 2 * Real.sqrt 2) ↔ (b / a = 2 * a / b) :=
by sorry

end NUMINAMATH_CALUDE_min_value_reciprocal_sum_min_value_reciprocal_sum_equality_l3928_392864


namespace NUMINAMATH_CALUDE_cubic_polynomials_common_roots_l3928_392877

theorem cubic_polynomials_common_roots (a b : ℝ) : 
  (∃ r s : ℝ, r ≠ s ∧ 
    r^3 + a*r^2 + 15*r + 10 = 0 ∧ 
    r^3 + b*r^2 + 18*r + 12 = 0 ∧
    s^3 + a*s^2 + 15*s + 10 = 0 ∧ 
    s^3 + b*s^2 + 18*s + 12 = 0) →
  a = 3 ∧ b = 4 := by
sorry

end NUMINAMATH_CALUDE_cubic_polynomials_common_roots_l3928_392877


namespace NUMINAMATH_CALUDE_circle_and_tangent_lines_l3928_392882

-- Define the circle C
def circle_C (x y : ℝ) := x^2 + (y + 1)^2 = 2

-- Define the parabola
def parabola (x y : ℝ) := y^2 = 4 * x

-- Define the line y = x
def line_y_eq_x (x y : ℝ) := y = x

-- Define the point P
def point_P : ℝ × ℝ := (-1, 2)

-- Define the two tangent lines
def tangent_line_1 (x y : ℝ) := x + y - 1 = 0
def tangent_line_2 (x y : ℝ) := 7 * x - y + 9 = 0

theorem circle_and_tangent_lines :
  -- Circle C is symmetric about the y-axis
  (∀ x y : ℝ, circle_C x y ↔ circle_C (-x) y) →
  -- Circle C passes through the focus of the parabola y^2 = 4x
  (circle_C 1 0) →
  -- Circle C is divided into two arc lengths with a ratio of 1:2 by the line y = x
  (∃ r : ℝ, r > 0 ∧ ∀ x y : ℝ, circle_C x y → line_y_eq_x x y → 
    (x^2 + y^2)^(1/2) = r ∧ ((x - 0)^2 + (y - (-1))^2)^(1/2) = 2 * r) →
  -- The center of circle C is below the x-axis
  (∃ a : ℝ, a < 0 ∧ ∀ x y : ℝ, circle_C x y ↔ x^2 + (y - a)^2 = 2) →
  -- The equation of circle C is x^2 + (y + 1)^2 = 2
  (∀ x y : ℝ, circle_C x y ↔ x^2 + (y + 1)^2 = 2) ∧
  -- The equations of the tangent lines passing through P(-1, 2) are x + y - 1 = 0 and 7x - y + 9 = 0
  (∀ x y : ℝ, (tangent_line_1 x y ∨ tangent_line_2 x y) ↔
    (∃ t : ℝ, circle_C (point_P.1 + t * (x - point_P.1)) (point_P.2 + t * (y - point_P.2)) ∧
      (∀ s : ℝ, s ≠ t → ¬ circle_C (point_P.1 + s * (x - point_P.1)) (point_P.2 + s * (y - point_P.2))))) :=
by sorry

end NUMINAMATH_CALUDE_circle_and_tangent_lines_l3928_392882


namespace NUMINAMATH_CALUDE_kenny_basketball_time_l3928_392821

/-- Represents Kenny's activities and their durations --/
structure KennyActivities where
  basketball : ℝ
  running : ℝ
  trumpet : ℝ
  swimming : ℝ
  studying : ℝ

/-- Theorem stating the duration of Kenny's basketball playing --/
theorem kenny_basketball_time (k : KennyActivities) 
  (h1 : k.running = 2 * k.basketball)
  (h2 : k.trumpet = 2 * k.running)
  (h3 : k.swimming = 2.5 * k.trumpet)
  (h4 : k.studying = 0.5 * k.swimming)
  (h5 : k.trumpet = 40) : 
  k.basketball = 10 := by
  sorry

end NUMINAMATH_CALUDE_kenny_basketball_time_l3928_392821


namespace NUMINAMATH_CALUDE_belize_homes_count_belize_homes_count_proof_l3928_392885

theorem belize_homes_count : ℕ → Prop :=
  fun total_homes =>
    let white_homes := total_homes / 4
    let non_white_homes := total_homes - white_homes
    let non_white_homes_with_fireplace := non_white_homes / 5
    let non_white_homes_without_fireplace := non_white_homes - non_white_homes_with_fireplace
    non_white_homes_without_fireplace = 240 → total_homes = 400

-- Proof
theorem belize_homes_count_proof : ∃ (total_homes : ℕ), belize_homes_count total_homes :=
  sorry

end NUMINAMATH_CALUDE_belize_homes_count_belize_homes_count_proof_l3928_392885


namespace NUMINAMATH_CALUDE_degree_of_polynomial_power_l3928_392886

/-- The degree of the polynomial (5x^3 + 7x + 2)^10 is 30. -/
theorem degree_of_polynomial_power (x : ℝ) : 
  Polynomial.degree ((5 * X^3 + 7 * X + 2 : Polynomial ℝ)^10) = 30 := by
  sorry

end NUMINAMATH_CALUDE_degree_of_polynomial_power_l3928_392886


namespace NUMINAMATH_CALUDE_afternoon_morning_difference_l3928_392829

/-- The number of campers who went rowing in the morning -/
def morning_campers : ℕ := 52

/-- The number of campers who went rowing in the afternoon -/
def afternoon_campers : ℕ := 61

/-- The theorem states that the difference between the number of campers
    who went rowing in the afternoon and the number of campers who went
    rowing in the morning is 9 -/
theorem afternoon_morning_difference :
  afternoon_campers - morning_campers = 9 := by
  sorry

end NUMINAMATH_CALUDE_afternoon_morning_difference_l3928_392829


namespace NUMINAMATH_CALUDE_min_a_for_absolute_value_inequality_l3928_392810

/-- The minimum value of a that satisfies the condition "if 0 < x < 1, then |x| < a" is 1. -/
theorem min_a_for_absolute_value_inequality : 
  (∀ x : ℝ, 0 < x → x < 1 → |x| < a) ↔ a ≥ 1 :=
by sorry

end NUMINAMATH_CALUDE_min_a_for_absolute_value_inequality_l3928_392810


namespace NUMINAMATH_CALUDE_hyperbola_equation_l3928_392828

theorem hyperbola_equation (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1) →
  (a^2 + b^2 = 16) →
  (b / a = Real.sqrt 55 / 11) →
  (∀ x y : ℝ, x^2 / 11 - y^2 / 5 = 1) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l3928_392828


namespace NUMINAMATH_CALUDE_monomial_sum_l3928_392859

/-- Given two monomials of the same type, prove their sum -/
theorem monomial_sum (m n : ℕ) : 
  (2 : ℤ) * X^m * Y^3 + (-5 : ℤ) * X^1 * Y^(n+1) = (-3 : ℤ) * X^1 * Y^3 :=
by sorry

end NUMINAMATH_CALUDE_monomial_sum_l3928_392859


namespace NUMINAMATH_CALUDE_cheesecake_calories_per_slice_quarter_of_slices_is_two_l3928_392808

/-- Represents a cheesecake with its total calories and number of slices -/
structure Cheesecake where
  totalCalories : ℕ
  numSlices : ℕ

/-- Calculates the number of calories per slice in a cheesecake -/
def caloriesPerSlice (cake : Cheesecake) : ℕ :=
  cake.totalCalories / cake.numSlices

theorem cheesecake_calories_per_slice :
  ∀ (cake : Cheesecake),
    cake.totalCalories = 2800 →
    cake.numSlices = 8 →
    caloriesPerSlice cake = 350 := by
  sorry

/-- Verifies that 25% of the total slices is equal to 2 slices -/
theorem quarter_of_slices_is_two (cake : Cheesecake) :
  cake.numSlices = 8 →
  cake.numSlices / 4 = 2 := by
  sorry

end NUMINAMATH_CALUDE_cheesecake_calories_per_slice_quarter_of_slices_is_two_l3928_392808


namespace NUMINAMATH_CALUDE_bus_travel_fraction_l3928_392841

theorem bus_travel_fraction (total_distance : ℝ) 
  (h1 : total_distance = 105.00000000000003)
  (h2 : (1 : ℝ) / 5 * total_distance + 14 + (2 : ℝ) / 3 * total_distance = total_distance) :
  (total_distance - ((1 : ℝ) / 5 * total_distance + 14)) / total_distance = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_bus_travel_fraction_l3928_392841


namespace NUMINAMATH_CALUDE_kim_gum_needs_l3928_392857

/-- The number of cousins Kim has -/
def num_cousins : ℕ := 4

/-- The number of gum pieces Kim wants to give to each cousin -/
def gum_per_cousin : ℕ := 5

/-- The total number of gum pieces Kim needs -/
def total_gum : ℕ := num_cousins * gum_per_cousin

/-- Theorem stating that the total number of gum pieces Kim needs is 20 -/
theorem kim_gum_needs : total_gum = 20 := by sorry

end NUMINAMATH_CALUDE_kim_gum_needs_l3928_392857


namespace NUMINAMATH_CALUDE_common_material_choices_eq_120_l3928_392842

/-- The number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := Nat.choose n k

/-- The number of ways two students can choose 2 materials each from 6 materials,
    such that they have exactly 1 material in common -/
def commonMaterialChoices : ℕ :=
  choose 6 1 * choose 5 2

theorem common_material_choices_eq_120 : commonMaterialChoices = 120 := by
  sorry


end NUMINAMATH_CALUDE_common_material_choices_eq_120_l3928_392842


namespace NUMINAMATH_CALUDE_remainder_theorem_l3928_392856

theorem remainder_theorem : (7 * 11^24 + 2^24) % 12 = 11 := by
  sorry

end NUMINAMATH_CALUDE_remainder_theorem_l3928_392856


namespace NUMINAMATH_CALUDE_charles_paint_area_l3928_392863

/-- 
Given a wall that requires 320 square feet to be painted and a work ratio of 2:6 between Allen and Charles,
prove that Charles paints 240 square feet.
-/
theorem charles_paint_area (total_area : ℝ) (allen_ratio charles_ratio : ℕ) : 
  total_area = 320 →
  allen_ratio = 2 →
  charles_ratio = 6 →
  (charles_ratio / (allen_ratio + charles_ratio)) * total_area = 240 := by
  sorry

end NUMINAMATH_CALUDE_charles_paint_area_l3928_392863


namespace NUMINAMATH_CALUDE_sum_of_fractions_l3928_392819

theorem sum_of_fractions : (1 : ℚ) / 3 + (1 : ℚ) / 4 = 7 / 12 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_fractions_l3928_392819


namespace NUMINAMATH_CALUDE_steak_cost_solution_l3928_392827

/-- The cost of a steak given the conditions of the problem -/
def steak_cost : ℝ → Prop := λ s =>
  let drink_cost : ℝ := 5
  let tip_paid : ℝ := 8
  let tip_percentage : ℝ := 0.2
  let tip_coverage : ℝ := 0.8
  let total_meal_cost : ℝ := 2 * s + 2 * drink_cost
  tip_paid = tip_coverage * tip_percentage * total_meal_cost ∧ s = 20

theorem steak_cost_solution :
  ∃ s : ℝ, steak_cost s :=
sorry

end NUMINAMATH_CALUDE_steak_cost_solution_l3928_392827


namespace NUMINAMATH_CALUDE_nina_widget_problem_l3928_392817

theorem nina_widget_problem (x : ℝ) 
  (h1 : 15 * x = 25 * (x - 5)) : 
  15 * x = 187.50 := by
  sorry

end NUMINAMATH_CALUDE_nina_widget_problem_l3928_392817


namespace NUMINAMATH_CALUDE_sum_of_special_numbers_l3928_392840

/-- A function that counts the number of divisors of a natural number -/
def count_divisors (n : ℕ) : ℕ := sorry

/-- A function that checks if a number ends with 5 zeros -/
def ends_with_five_zeros (n : ℕ) : Prop := sorry

/-- The set of natural numbers that end with 5 zeros and have 42 divisors -/
def special_numbers : Set ℕ :=
  {n : ℕ | ends_with_five_zeros n ∧ count_divisors n = 42}

theorem sum_of_special_numbers :
  ∃ (a b : ℕ), a ∈ special_numbers ∧ b ∈ special_numbers ∧ a ≠ b ∧ a + b = 700000 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_special_numbers_l3928_392840


namespace NUMINAMATH_CALUDE_puppies_adopted_theorem_l3928_392847

/-- The number of puppies adopted each day from a shelter -/
def puppies_adopted_per_day (initial_puppies additional_puppies adoption_days : ℕ) : ℕ :=
  (initial_puppies + additional_puppies) / adoption_days

/-- Theorem stating the number of puppies adopted each day -/
theorem puppies_adopted_theorem (initial_puppies additional_puppies adoption_days : ℕ) 
  (h1 : initial_puppies = 5)
  (h2 : additional_puppies = 35)
  (h3 : adoption_days = 5) :
  puppies_adopted_per_day initial_puppies additional_puppies adoption_days = 8 := by
  sorry

end NUMINAMATH_CALUDE_puppies_adopted_theorem_l3928_392847


namespace NUMINAMATH_CALUDE_parabola_vertex_fourth_quadrant_l3928_392865

/-- A parabola with equation y = -2(x+a)^2 + c -/
structure Parabola (a c : ℝ) where
  equation : ℝ → ℝ
  eq_def : ∀ x, equation x = -2 * (x + a)^2 + c

/-- The vertex of a parabola -/
def vertex (p : Parabola a c) : ℝ × ℝ := (-a, c)

/-- A point is in the fourth quadrant if its x-coordinate is positive and y-coordinate is negative -/
def in_fourth_quadrant (point : ℝ × ℝ) : Prop :=
  point.1 > 0 ∧ point.2 < 0

/-- Theorem: For a parabola y = -2(x+a)^2 + c with its vertex in the fourth quadrant, a < 0 and c < 0 -/
theorem parabola_vertex_fourth_quadrant {a c : ℝ} (p : Parabola a c) 
  (h : in_fourth_quadrant (vertex p)) : a < 0 ∧ c < 0 := by
  sorry


end NUMINAMATH_CALUDE_parabola_vertex_fourth_quadrant_l3928_392865


namespace NUMINAMATH_CALUDE_original_class_strength_l3928_392879

/-- Given an adult class, prove that the original strength was 12 students. -/
theorem original_class_strength
  (original_avg : ℝ)
  (new_students : ℕ)
  (new_avg : ℝ)
  (avg_decrease : ℝ)
  (h1 : original_avg = 40)
  (h2 : new_students = 12)
  (h3 : new_avg = 32)
  (h4 : avg_decrease = 4)
  : ∃ (x : ℕ), x = 12 ∧ 
    (x : ℝ) * original_avg + (new_students : ℝ) * new_avg = 
    ((x : ℝ) + new_students) * (original_avg - avg_decrease) :=
by
  sorry


end NUMINAMATH_CALUDE_original_class_strength_l3928_392879


namespace NUMINAMATH_CALUDE_pure_imaginary_equation_l3928_392843

/-- A complex number is pure imaginary if its real part is zero and its imaginary part is nonzero -/
def IsPureImaginary (z : ℂ) : Prop := z.re = 0 ∧ z.im ≠ 0

theorem pure_imaginary_equation (z : ℂ) (a : ℝ) 
  (h1 : IsPureImaginary z) 
  (h2 : (1 + Complex.I) * z = 1 - a * Complex.I) : 
  a = 1 := by
  sorry

end NUMINAMATH_CALUDE_pure_imaginary_equation_l3928_392843


namespace NUMINAMATH_CALUDE_function_value_at_zero_l3928_392881

theorem function_value_at_zero 
  (f : ℝ → ℝ) 
  (h1 : ∀ x : ℝ, f (x + 2) = f (x + 1) - f x) 
  (h2 : f 1 = Real.log (3/2)) 
  (h3 : f 2 = Real.log 15) : 
  f 0 = -1 := by sorry

end NUMINAMATH_CALUDE_function_value_at_zero_l3928_392881


namespace NUMINAMATH_CALUDE_lcm_of_3_4_6_15_l3928_392822

def numbers : List ℕ := [3, 4, 6, 15]

theorem lcm_of_3_4_6_15 : Nat.lcm (Nat.lcm (Nat.lcm 3 4) 6) 15 = 60 := by
  sorry

end NUMINAMATH_CALUDE_lcm_of_3_4_6_15_l3928_392822


namespace NUMINAMATH_CALUDE_nonAthleticParentsCount_l3928_392852

/-- Represents the number of students with various athletic parent combinations -/
structure AthleticParents where
  total : Nat
  athleticDad : Nat
  athleticMom : Nat
  bothAthletic : Nat

/-- Calculates the number of students with both non-athletic parents -/
def nonAthleticParents (ap : AthleticParents) : Nat :=
  ap.total - (ap.athleticDad + ap.athleticMom - ap.bothAthletic)

/-- Theorem stating that given the specific numbers in the problem, 
    the number of students with both non-athletic parents is 19 -/
theorem nonAthleticParentsCount : 
  let ap : AthleticParents := {
    total := 45,
    athleticDad := 17,
    athleticMom := 20,
    bothAthletic := 11
  }
  nonAthleticParents ap = 19 := by
  sorry

end NUMINAMATH_CALUDE_nonAthleticParentsCount_l3928_392852


namespace NUMINAMATH_CALUDE_age_ratio_problem_l3928_392825

theorem age_ratio_problem (j e : ℕ) (h1 : j - 6 = 4 * (e - 6)) (h2 : j - 4 = 3 * (e - 4)) :
  ∃ x : ℕ, x = 14 ∧ (j + x) * 2 = (e + x) * 3 :=
sorry

end NUMINAMATH_CALUDE_age_ratio_problem_l3928_392825


namespace NUMINAMATH_CALUDE_A_C_mutually_exclusive_l3928_392874

/-- Represents the sample space of three products -/
structure ThreeProducts where
  product1 : Bool  -- True if defective, False if not defective
  product2 : Bool
  product3 : Bool

/-- Event A: All three products are not defective -/
def A (s : ThreeProducts) : Prop :=
  ¬s.product1 ∧ ¬s.product2 ∧ ¬s.product3

/-- Event B: All three products are defective -/
def B (s : ThreeProducts) : Prop :=
  s.product1 ∧ s.product2 ∧ s.product3

/-- Event C: At least one of the three products is defective -/
def C (s : ThreeProducts) : Prop :=
  s.product1 ∨ s.product2 ∨ s.product3

/-- Theorem: A and C are mutually exclusive -/
theorem A_C_mutually_exclusive :
  ∀ s : ThreeProducts, ¬(A s ∧ C s) :=
by sorry

end NUMINAMATH_CALUDE_A_C_mutually_exclusive_l3928_392874


namespace NUMINAMATH_CALUDE_min_value_theorem_l3928_392887

theorem min_value_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : Real.log (2^x) + Real.log (8^y) = Real.log 2) : 
  (x + y) / (x * y) ≥ 2 * Real.sqrt 3 + 4 := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l3928_392887


namespace NUMINAMATH_CALUDE_max_cos_a_value_l3928_392858

theorem max_cos_a_value (a b c : Real) 
  (h1 : Real.sin a = Real.cos b)
  (h2 : Real.sin b = Real.cos c)
  (h3 : Real.sin c = Real.cos a) :
  ∃ (max_cos_a : Real), max_cos_a = Real.sqrt 2 / 2 ∧ 
    ∀ x, Real.cos a ≤ x → x ≤ max_cos_a :=
by sorry

end NUMINAMATH_CALUDE_max_cos_a_value_l3928_392858


namespace NUMINAMATH_CALUDE_product_325_7_4_7_l3928_392888

/-- Converts a base 7 number to base 10 --/
def to_base_10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * 7^i) 0

/-- Converts a base 10 number to base 7 --/
def to_base_7 (n : Nat) : List Nat :=
  if n = 0 then [0] else
  let rec aux (m : Nat) (acc : List Nat) :=
    if m = 0 then acc else aux (m / 7) ((m % 7) :: acc)
  aux n []

/-- Theorem: The product of 325₇ and 4₇ is equal to 1656₇ in base 7 --/
theorem product_325_7_4_7 : 
  to_base_7 (to_base_10 [5, 2, 3] * to_base_10 [4]) = [6, 5, 6, 1] := by
  sorry

end NUMINAMATH_CALUDE_product_325_7_4_7_l3928_392888


namespace NUMINAMATH_CALUDE_kendra_sunday_shirts_l3928_392816

/-- The number of shirts Kendra wears in two weeks -/
def total_shirts : ℕ := 22

/-- The number of weekdays in two weeks -/
def weekdays : ℕ := 10

/-- The number of days Kendra changes shirts for after-school club in two weeks -/
def club_days : ℕ := 6

/-- The number of Saturdays in two weeks -/
def saturdays : ℕ := 2

/-- The number of Sundays in two weeks -/
def sundays : ℕ := 2

/-- The number of shirts Kendra wears on weekdays for school and club in two weeks -/
def weekday_shirts : ℕ := weekdays + club_days

/-- The number of shirts Kendra wears on Saturdays in two weeks -/
def saturday_shirts : ℕ := saturdays

theorem kendra_sunday_shirts :
  total_shirts - (weekday_shirts + saturday_shirts) = 4 := by
sorry

end NUMINAMATH_CALUDE_kendra_sunday_shirts_l3928_392816


namespace NUMINAMATH_CALUDE_difference_of_odd_squares_divisible_by_eight_l3928_392823

theorem difference_of_odd_squares_divisible_by_eight (a b : ℤ) 
  (ha : ∃ k : ℤ, a = 2 * k + 1) 
  (hb : ∃ m : ℤ, b = 2 * m + 1) : 
  ∃ n : ℤ, a ^ 2 - b ^ 2 = 8 * n := by
  sorry

end NUMINAMATH_CALUDE_difference_of_odd_squares_divisible_by_eight_l3928_392823


namespace NUMINAMATH_CALUDE_quadrilateral_angle_measure_l3928_392805

/-- Given two equations representing interior angles of a quadrilateral,
    prove that the positive solution for x is (1 + √13) / 6. -/
theorem quadrilateral_angle_measure 
  (x y : ℝ) 
  (eq1 : 3 * x^2 - x + 4 = 5) 
  (eq2 : x^2 + y^2 = 9) 
  (h_positive : x > 0) :
  x = (1 + Real.sqrt 13) / 6 := by
  sorry

end NUMINAMATH_CALUDE_quadrilateral_angle_measure_l3928_392805


namespace NUMINAMATH_CALUDE_three_solutions_cubic_equation_l3928_392812

theorem three_solutions_cubic_equation (n : ℕ+) (x y : ℤ) 
  (h : x^3 - 3*x*y^2 + y^3 = n) :
  ∃ (a b c d e f : ℤ), 
    (a^3 - 3*a*b^2 + b^3 = n) ∧ 
    (c^3 - 3*c*d^2 + d^3 = n) ∧ 
    (e^3 - 3*e*f^2 + f^3 = n) ∧ 
    (a, b) ≠ (c, d) ∧ (a, b) ≠ (e, f) ∧ (c, d) ≠ (e, f) :=
by sorry

end NUMINAMATH_CALUDE_three_solutions_cubic_equation_l3928_392812


namespace NUMINAMATH_CALUDE_sum_interior_angles_regular_polygon_l3928_392832

/-- For a regular polygon where each exterior angle measures 20 degrees, 
    the sum of the measures of its interior angles is 2880 degrees. -/
theorem sum_interior_angles_regular_polygon : 
  ∀ (n : ℕ), 
    n > 2 → 
    (360 : ℝ) / n = 20 → 
    (n - 2 : ℝ) * 180 = 2880 := by
  sorry

end NUMINAMATH_CALUDE_sum_interior_angles_regular_polygon_l3928_392832


namespace NUMINAMATH_CALUDE_distance_to_y_axis_l3928_392869

/-- Theorem: For a point P with coordinates (x, -6), if the distance from the x-axis to P
    is half the distance from the y-axis to P, then the distance from the y-axis to P is 12 units. -/
theorem distance_to_y_axis (x : ℝ) :
  let P : ℝ × ℝ := (x, -6)
  let distance_to_x_axis := |P.2|
  let distance_to_y_axis := |P.1|
  distance_to_x_axis = (1/2 : ℝ) * distance_to_y_axis →
  distance_to_y_axis = 12 := by
  sorry

end NUMINAMATH_CALUDE_distance_to_y_axis_l3928_392869


namespace NUMINAMATH_CALUDE_cost_of_45_lilies_l3928_392896

/-- The cost of a bouquet of lilies at Lila's Lily Shop -/
def bouquet_cost (n : ℕ) : ℚ :=
  let base_price := 2 * n  -- $2 per lily
  if n ≤ 30 then base_price else base_price * (1 - 1/10)  -- 10% discount if > 30 lilies

/-- Theorem: The cost of a bouquet with 45 lilies is $81 -/
theorem cost_of_45_lilies :
  bouquet_cost 45 = 81 := by sorry

end NUMINAMATH_CALUDE_cost_of_45_lilies_l3928_392896


namespace NUMINAMATH_CALUDE_square_circle_equal_area_l3928_392834

theorem square_circle_equal_area (r : ℝ) (s : ℝ) : 
  r = 5 →
  s = 2 * r →
  s^2 = π * r^2 →
  s = 5 * Real.sqrt π :=
by sorry

end NUMINAMATH_CALUDE_square_circle_equal_area_l3928_392834


namespace NUMINAMATH_CALUDE_kelsey_ekon_difference_l3928_392855

/-- The number of videos watched by three friends. -/
def total_videos : ℕ := 411

/-- The number of videos watched by Kelsey. -/
def kelsey_videos : ℕ := 160

/-- The number of videos watched by Uma. -/
def uma_videos : ℕ := (total_videos - kelsey_videos + 17) / 2

/-- The number of videos watched by Ekon. -/
def ekon_videos : ℕ := uma_videos - 17

/-- Theorem stating the difference in videos watched between Kelsey and Ekon. -/
theorem kelsey_ekon_difference :
  kelsey_videos - ekon_videos = 43 :=
by sorry

end NUMINAMATH_CALUDE_kelsey_ekon_difference_l3928_392855


namespace NUMINAMATH_CALUDE_beatrix_height_relative_to_georgia_l3928_392873

theorem beatrix_height_relative_to_georgia (B V G : ℝ) 
  (h1 : B = 2 * V) 
  (h2 : V = 2/3 * G) : 
  B = 4/3 * G := by
sorry

end NUMINAMATH_CALUDE_beatrix_height_relative_to_georgia_l3928_392873


namespace NUMINAMATH_CALUDE_cube_difference_divisibility_l3928_392872

theorem cube_difference_divisibility (a b : ℤ) :
  ∃ k : ℤ, (2*a + 1)^3 - (2*b + 1)^3 + 8 = 16 * k := by
  sorry

end NUMINAMATH_CALUDE_cube_difference_divisibility_l3928_392872


namespace NUMINAMATH_CALUDE_cube_difference_l3928_392824

theorem cube_difference (a b : ℝ) (h1 : a + b = 12) (h2 : a * b = 20) :
  a^3 - b^3 = 992 := by
sorry

end NUMINAMATH_CALUDE_cube_difference_l3928_392824


namespace NUMINAMATH_CALUDE_shoe_production_facts_l3928_392868

/-- The daily production cost function -/
def C (n : ℕ) : ℝ := 4000 + 50 * n

/-- The selling price per pair of shoes -/
def sellingPrice : ℝ := 90

/-- All produced shoes are sold out -/
axiom all_sold : ∀ n : ℕ, n > 0 → ∃ revenue : ℝ, revenue = sellingPrice * n

/-- The profit function -/
def P (n : ℕ) : ℝ := sellingPrice * n - C n

theorem shoe_production_facts :
  (C 1000 = 54000) ∧
  (∃ n : ℕ, C n = 48000 ∧ n = 880) ∧
  (∀ n : ℕ, P n = 40 * n - 4000) ∧
  (∃ min_n : ℕ, min_n = 100 ∧ ∀ n : ℕ, n ≥ min_n → P n ≥ 0) :=
by sorry

end NUMINAMATH_CALUDE_shoe_production_facts_l3928_392868


namespace NUMINAMATH_CALUDE_square_of_difference_with_sqrt_l3928_392835

theorem square_of_difference_with_sqrt (x : ℝ) : 
  (7 - Real.sqrt (x^2 - 49*x + 169))^2 = x^2 - 49*x + 218 - 14*Real.sqrt (x^2 - 49*x + 169) := by
  sorry

end NUMINAMATH_CALUDE_square_of_difference_with_sqrt_l3928_392835


namespace NUMINAMATH_CALUDE_max_value_expression_l3928_392836

theorem max_value_expression (x y : ℝ) : 
  (3 * x + 4 * y + 5) / Real.sqrt (x^2 + y^2 + 4) ≤ 5 := by
  sorry

end NUMINAMATH_CALUDE_max_value_expression_l3928_392836
