import Mathlib

namespace NUMINAMATH_CALUDE_inequality_condition_neither_sufficient_nor_necessary_l2138_213879

theorem inequality_condition_neither_sufficient_nor_necessary (a b : ℝ) :
  ¬(∀ a b : ℝ, (a > b → 1/a < 1/b) → a > b) ∧
  ¬(∀ a b : ℝ, a > b → (1/a < 1/b)) :=
sorry

end NUMINAMATH_CALUDE_inequality_condition_neither_sufficient_nor_necessary_l2138_213879


namespace NUMINAMATH_CALUDE_decimal_sum_to_fraction_l2138_213845

theorem decimal_sum_to_fraction :
  (0.4 + 0.05 + 0.006 + 0.0007 + 0.00008 : ℚ) = 22839 / 50000 := by
  sorry

end NUMINAMATH_CALUDE_decimal_sum_to_fraction_l2138_213845


namespace NUMINAMATH_CALUDE_duck_pond_problem_l2138_213801

theorem duck_pond_problem (small_pond : ℕ) (large_pond : ℕ) : 
  large_pond = 80 →
  (small_pond * 20 : ℕ) / 100 + (large_pond * 15 : ℕ) / 100 = ((small_pond + large_pond) * 16 : ℕ) / 100 →
  small_pond = 20 := by
sorry

end NUMINAMATH_CALUDE_duck_pond_problem_l2138_213801


namespace NUMINAMATH_CALUDE_division_of_decimals_l2138_213834

theorem division_of_decimals : (0.45 : ℝ) / 0.005 = 90 := by sorry

end NUMINAMATH_CALUDE_division_of_decimals_l2138_213834


namespace NUMINAMATH_CALUDE_jack_plates_problem_l2138_213870

theorem jack_plates_problem (flower_initial : ℕ) (checked : ℕ) (total_final : ℕ) :
  flower_initial = 4 →
  total_final = 27 →
  total_final = (flower_initial - 1) + checked + 2 * checked →
  checked = 8 := by
  sorry

end NUMINAMATH_CALUDE_jack_plates_problem_l2138_213870


namespace NUMINAMATH_CALUDE_triangle_with_120_degree_angle_divisible_into_isosceles_l2138_213806

-- Define a triangle type
structure Triangle :=
  (a b c : ℝ)
  (sum_to_180 : a + b + c = 180)
  (all_positive : 0 < a ∧ 0 < b ∧ 0 < c)

-- Define an isosceles triangle
def IsIsosceles (t : Triangle) : Prop :=
  t.a = t.b ∨ t.b = t.c ∨ t.c = t.a

-- Define the property of being divisible into two isosceles triangles
def DivisibleIntoTwoIsosceles (t : Triangle) : Prop :=
  ∃ (t1 t2 : Triangle), IsIsosceles t1 ∧ IsIsosceles t2

-- The main theorem
theorem triangle_with_120_degree_angle_divisible_into_isosceles
  (t : Triangle)
  (has_120_degree : t.a = 120 ∨ t.b = 120 ∨ t.c = 120)
  (divisible : DivisibleIntoTwoIsosceles t) :
  (t.b = 30 ∧ t.c = 15) ∨ (t.b = 45 ∧ t.c = 15) ∨
  (t.b = 15 ∧ t.c = 30) ∨ (t.b = 15 ∧ t.c = 45) :=
by sorry


end NUMINAMATH_CALUDE_triangle_with_120_degree_angle_divisible_into_isosceles_l2138_213806


namespace NUMINAMATH_CALUDE_system_solution_l2138_213893

theorem system_solution (a₁ a₂ c₁ c₂ : ℝ) :
  (∃ (x y : ℝ), a₁ * x + y = c₁ ∧ a₂ * x + y = c₂ ∧ x = 5 ∧ y = 10) →
  (∃ (x y : ℝ), a₁ * x + 2 * y = a₁ - c₁ ∧ a₂ * x + 2 * y = a₂ - c₂ ∧ x = -4 ∧ y = -5) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l2138_213893


namespace NUMINAMATH_CALUDE_muffins_per_box_l2138_213802

theorem muffins_per_box (total_muffins : ℕ) (available_boxes : ℕ) (additional_boxes : ℕ) :
  total_muffins = 95 →
  available_boxes = 10 →
  additional_boxes = 9 →
  (total_muffins / (available_boxes + additional_boxes) : ℚ) = 5 := by
sorry

end NUMINAMATH_CALUDE_muffins_per_box_l2138_213802


namespace NUMINAMATH_CALUDE_extremum_implies_deriv_zero_not_always_converse_l2138_213851

open Set
open Function
open Topology

-- Define a structure for differentiable functions on ℝ
structure DiffFunction where
  f : ℝ → ℝ
  diff : Differentiable ℝ f

variable (f : DiffFunction)

-- Define what it means for a function to have an extremum
def has_extremum (f : DiffFunction) : Prop :=
  ∃ x₀ : ℝ, ∀ x : ℝ, f.f x ≤ f.f x₀ ∨ f.f x ≥ f.f x₀

-- Define what it means for f'(x) = 0 to have a solution
def deriv_has_zero (f : DiffFunction) : Prop :=
  ∃ x : ℝ, deriv f.f x = 0

-- State the theorem
theorem extremum_implies_deriv_zero (f : DiffFunction) : 
  has_extremum f → deriv_has_zero f :=
sorry

-- State that the converse is not always true
theorem not_always_converse : 
  ∃ f : DiffFunction, deriv_has_zero f ∧ ¬has_extremum f :=
sorry

end NUMINAMATH_CALUDE_extremum_implies_deriv_zero_not_always_converse_l2138_213851


namespace NUMINAMATH_CALUDE_second_watermelon_weight_l2138_213825

theorem second_watermelon_weight (total_weight first_weight : ℝ) 
  (h1 : total_weight = 14.02)
  (h2 : first_weight = 9.91) : 
  total_weight - first_weight = 4.11 := by
sorry

end NUMINAMATH_CALUDE_second_watermelon_weight_l2138_213825


namespace NUMINAMATH_CALUDE_pencil_cost_theorem_l2138_213883

/-- Calculates the average cost per pencil in cents, rounded to the nearest cent -/
def averageCostPerPencil (pencilCount : ℕ) (pencilCost : ℚ) (shippingCost : ℚ) (discount : ℚ) : ℕ :=
  let totalCost := pencilCost + shippingCost - discount
  let totalCostInCents := (totalCost * 100).floor
  ((totalCostInCents + pencilCount / 2) / pencilCount).toNat

theorem pencil_cost_theorem :
  let pencilCount : ℕ := 150
  let pencilCost : ℚ := 15.5
  let shippingCost : ℚ := 5.75
  let discount : ℚ := 1

  averageCostPerPencil pencilCount pencilCost shippingCost discount = 14 := by
    sorry

#eval averageCostPerPencil 150 15.5 5.75 1

end NUMINAMATH_CALUDE_pencil_cost_theorem_l2138_213883


namespace NUMINAMATH_CALUDE_distinct_three_digit_count_base_6_l2138_213896

/-- The number of three-digit numbers with distinct digits in base b -/
def distinct_three_digit_count (b : ℕ) : ℕ := (b - 1)^2 * (b - 2)

/-- Theorem: In base 6, there are exactly 100 three-digit numbers with distinct digits -/
theorem distinct_three_digit_count_base_6 : distinct_three_digit_count 6 = 100 := by
  sorry

#eval distinct_three_digit_count 6  -- This should evaluate to 100

end NUMINAMATH_CALUDE_distinct_three_digit_count_base_6_l2138_213896


namespace NUMINAMATH_CALUDE_coefficient_of_x_cubed_l2138_213863

theorem coefficient_of_x_cubed (x : ℝ) : 
  let expression := 4*(x^3 - 2*x^4) + 3*(x^2 - 3*x^3 + 4*x^6) - (5*x^4 - 2*x^3)
  ∃ (a b c d e : ℝ), expression = -3*x^3 + a*x^2 + b*x^4 + c*x^6 + d*x + e :=
by
  sorry

end NUMINAMATH_CALUDE_coefficient_of_x_cubed_l2138_213863


namespace NUMINAMATH_CALUDE_triangle_cosine_l2138_213880

theorem triangle_cosine (X Y Z : ℝ) (h1 : X + Y + Z = Real.pi) 
  (h2 : X = Real.pi / 2) (h3 : Y = Real.pi / 4) (h4 : Real.tan Z = 1 / 2) : 
  Real.cos Z = Real.sqrt 5 / 5 := by
  sorry

end NUMINAMATH_CALUDE_triangle_cosine_l2138_213880


namespace NUMINAMATH_CALUDE_quadratic_inequality_roots_l2138_213836

theorem quadratic_inequality_roots (c : ℝ) : 
  (∀ x, -x^2 + c*x + 3 < 0 ↔ x < -3 ∨ x > 2) → c = 5 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_roots_l2138_213836


namespace NUMINAMATH_CALUDE_x_makes_2n_plus_x_composite_x_is_correct_l2138_213823

/-- The number added to 2n to make it not prime when n = 4 -/
def x : ℕ := 1

/-- The smallest n for which 2n + x is not prime -/
def smallest_n : ℕ := 4

theorem x_makes_2n_plus_x_composite : 
  ¬ Nat.Prime (2 * smallest_n + x) ∧ 
  ∀ m < smallest_n, Nat.Prime (2 * m + x) := by
  sorry

theorem x_is_correct : x = 1 := by
  sorry

end NUMINAMATH_CALUDE_x_makes_2n_plus_x_composite_x_is_correct_l2138_213823


namespace NUMINAMATH_CALUDE_f_of_f_of_2_l2138_213857

def f (x : ℝ) : ℝ := 4 * x^2 - 7

theorem f_of_f_of_2 : f (f 2) = 317 := by
  sorry

end NUMINAMATH_CALUDE_f_of_f_of_2_l2138_213857


namespace NUMINAMATH_CALUDE_sarah_copies_3600_pages_l2138_213862

/-- The total number of pages Sarah will copy for two contracts -/
def total_pages (num_people : ℕ) (contract1_pages : ℕ) (contract1_copies : ℕ) 
                (contract2_pages : ℕ) (contract2_copies : ℕ) : ℕ :=
  num_people * (contract1_pages * contract1_copies + contract2_pages * contract2_copies)

/-- Theorem: Sarah will copy 3600 pages in total -/
theorem sarah_copies_3600_pages : 
  total_pages 20 30 3 45 2 = 3600 := by
  sorry

end NUMINAMATH_CALUDE_sarah_copies_3600_pages_l2138_213862


namespace NUMINAMATH_CALUDE_value_of_q_l2138_213892

theorem value_of_q (p q : ℝ) 
  (h1 : 1 < p) 
  (h2 : p < q) 
  (h3 : (1 / p) + (1 / q) = 1) 
  (h4 : p * q = 16 / 3) : 
  q = 4 := by
sorry

end NUMINAMATH_CALUDE_value_of_q_l2138_213892


namespace NUMINAMATH_CALUDE_workshop_ratio_l2138_213856

theorem workshop_ratio (total : ℕ) (novelists : ℕ) (poets : ℕ) : 
  total = 24 → novelists = 15 → poets = total - novelists → 
  ∃ (a b : ℕ), a = 3 ∧ b = 5 ∧ poets * b = novelists * a :=
sorry

end NUMINAMATH_CALUDE_workshop_ratio_l2138_213856


namespace NUMINAMATH_CALUDE_sqrt_eight_times_sqrt_two_l2138_213885

theorem sqrt_eight_times_sqrt_two : Real.sqrt 8 * Real.sqrt 2 = 4 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_eight_times_sqrt_two_l2138_213885


namespace NUMINAMATH_CALUDE_f_properties_l2138_213873

noncomputable section

variable (f : ℝ → ℝ)

-- f is an even function
def even_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

-- f satisfies the given functional equation
def satisfies_equation (f : ℝ → ℝ) : Prop := ∀ x, f (x + 2) = f x + f 1

-- f is monotonically increasing on [0, 1]
def monotone_increasing_on_unit_interval (f : ℝ → ℝ) : Prop :=
  ∀ x y, 0 ≤ x ∧ x < y ∧ y ≤ 1 → f x ≤ f y

-- The graph of f is symmetric about x = 1
def symmetric_about_one (f : ℝ → ℝ) : Prop := ∀ x, f (x + 1) = f (1 - x)

-- f is periodic
def periodic (f : ℝ → ℝ) : Prop := ∃ p > 0, ∀ x, f (x + p) = f x

-- f has local minima at even x-coordinates
def local_minima_at_even (f : ℝ → ℝ) : Prop :=
  ∀ x, ∃ ε > 0, ∀ y, |y - x| < ε → f x ≤ f y

theorem f_properties (heven : even_function f)
                     (heq : satisfies_equation f)
                     (hmon : monotone_increasing_on_unit_interval f) :
  symmetric_about_one f ∧ periodic f ∧ local_minima_at_even f := by sorry

end

end NUMINAMATH_CALUDE_f_properties_l2138_213873


namespace NUMINAMATH_CALUDE_cody_spent_25_tickets_on_beanie_l2138_213821

/-- The number of tickets Cody spent on the beanie -/
def tickets_spent_on_beanie (initial_tickets : ℕ) (additional_tickets : ℕ) (remaining_tickets : ℕ) : ℕ :=
  initial_tickets + additional_tickets - remaining_tickets

/-- Proof that Cody spent 25 tickets on the beanie -/
theorem cody_spent_25_tickets_on_beanie :
  tickets_spent_on_beanie 49 6 30 = 25 := by
  sorry

end NUMINAMATH_CALUDE_cody_spent_25_tickets_on_beanie_l2138_213821


namespace NUMINAMATH_CALUDE_chips_ounces_amber_chips_problem_l2138_213822

/-- Represents the problem of determining the number of ounces in a bag of chips. -/
theorem chips_ounces (total_money : ℚ) (candy_price : ℚ) (candy_ounces : ℚ) 
  (chips_price : ℚ) (max_ounces : ℚ) : ℚ :=
  let candy_bags := total_money / candy_price
  let candy_total_ounces := candy_bags * candy_ounces
  let chips_bags := total_money / chips_price
  let chips_ounces_per_bag := max_ounces / chips_bags
  chips_ounces_per_bag

/-- Proves that given the conditions in the problem, a bag of chips contains 17 ounces. -/
theorem amber_chips_problem : 
  chips_ounces 7 1 12 (14/10) 85 = 17 := by
  sorry

end NUMINAMATH_CALUDE_chips_ounces_amber_chips_problem_l2138_213822


namespace NUMINAMATH_CALUDE_olympiad_solution_l2138_213808

def olympiad_problem (N_a N_b N_c N_ab N_ac N_bc N_abc : ℕ) : Prop :=
  let total := N_a + N_b + N_c + N_ab + N_ac + N_bc + N_abc
  let B_not_A := N_b + N_bc
  let C_not_A := N_c + N_bc
  let A_and_others := N_ab + N_ac + N_abc
  let only_one := N_a + N_b + N_c
  total = 25 ∧
  B_not_A = 2 * C_not_A ∧
  N_a = A_and_others + 1 ∧
  2 * N_a = only_one

theorem olympiad_solution :
  ∀ N_a N_b N_c N_ab N_ac N_bc N_abc,
  olympiad_problem N_a N_b N_c N_ab N_ac N_bc N_abc →
  N_b = 6 := by
sorry

end NUMINAMATH_CALUDE_olympiad_solution_l2138_213808


namespace NUMINAMATH_CALUDE_cos_five_pi_sixth_minus_alpha_l2138_213831

theorem cos_five_pi_sixth_minus_alpha (α : ℝ) (h : Real.sin (π / 3 - α) = 1 / 3) :
  Real.cos (5 * π / 6 - α) = -1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_cos_five_pi_sixth_minus_alpha_l2138_213831


namespace NUMINAMATH_CALUDE_quadratic_root_in_unit_interval_l2138_213812

theorem quadratic_root_in_unit_interval 
  (a b c : ℝ) 
  (h : 2*a + 3*b + 6*c = 0) : 
  ∃ x : ℝ, 0 < x ∧ x < 1 ∧ a*x^2 + b*x + c = 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_in_unit_interval_l2138_213812


namespace NUMINAMATH_CALUDE_complex_magnitude_proof_l2138_213855

theorem complex_magnitude_proof (z : ℂ) (h : z = 1 + Complex.I) : 
  Complex.abs (z^2 - 2*z) = 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_proof_l2138_213855


namespace NUMINAMATH_CALUDE_max_of_three_l2138_213882

theorem max_of_three (a b c : ℝ) :
  let x := max a b
  ∀ m : ℝ, (m = max a (max b c) ↔ (m = x ∨ (c > x ∧ m = c))) :=
by sorry

end NUMINAMATH_CALUDE_max_of_three_l2138_213882


namespace NUMINAMATH_CALUDE_min_reciprocal_sum_l2138_213805

theorem min_reciprocal_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 12) :
  (1 / x + 1 / y) ≥ 1 / 3 ∧ ∃ x y : ℝ, x > 0 ∧ y > 0 ∧ x + y = 12 ∧ 1 / x + 1 / y = 1 / 3 :=
by sorry

end NUMINAMATH_CALUDE_min_reciprocal_sum_l2138_213805


namespace NUMINAMATH_CALUDE_three_digit_number_problem_l2138_213847

def is_single_digit (n : ℕ) : Prop := n ≥ 0 ∧ n ≤ 9

def reverse_number (h t u : ℕ) : ℕ := u * 100 + t * 10 + h

theorem three_digit_number_problem (h t u : ℕ) :
  is_single_digit h ∧
  is_single_digit t ∧
  u = h + 6 ∧
  u + h = 16 ∧
  (h * 100 + t * 10 + u + reverse_number h t u) % 10 = 6 ∧
  ((h * 100 + t * 10 + u + reverse_number h t u) / 10) % 10 = 9 →
  h = 5 ∧ t = 5 ∧ u = 11 := by
sorry

end NUMINAMATH_CALUDE_three_digit_number_problem_l2138_213847


namespace NUMINAMATH_CALUDE_constant_term_binomial_expansion_l2138_213843

theorem constant_term_binomial_expansion :
  ∃ (c : ℝ), c = 7 ∧ 
  ∀ (x : ℝ), x ≠ 0 → 
  ∃ (f : ℝ → ℝ), (λ x => (x^(1/3) + 1/(2*x))^8) = 
    (λ x => c + f x) ∧ (∀ (y : ℝ), y ≠ 0 → f y ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_constant_term_binomial_expansion_l2138_213843


namespace NUMINAMATH_CALUDE_max_candy_types_l2138_213817

/-- A type representing a student --/
def Student : Type := ℕ

/-- A type representing a candy type --/
def CandyType : Type := ℕ

/-- The total number of students --/
def total_students : ℕ := 1000

/-- A function representing whether a student received a certain candy type --/
def received (s : Student) (c : CandyType) : Prop := sorry

/-- The condition that for any 11 types of candy, each student received at least one of those types --/
def condition_eleven (N : ℕ) : Prop :=
  ∀ (s : Student) (cs : Finset CandyType),
    cs.card = 11 → (∃ c ∈ cs, received s c)

/-- The condition that for any two types of candy, there exists a student who received exactly one of those types --/
def condition_two (N : ℕ) : Prop :=
  ∀ (c1 c2 : CandyType),
    c1 ≠ c2 → (∃ s : Student, (received s c1 ∧ ¬received s c2) ∨ (¬received s c1 ∧ received s c2))

/-- The main theorem stating that the maximum possible value of N is 5501 --/
theorem max_candy_types :
  ∃ N : ℕ,
    (∀ N' : ℕ, condition_eleven N' ∧ condition_two N' → N' ≤ N) ∧
    condition_eleven N ∧ condition_two N ∧
    N = 5501 := by sorry

end NUMINAMATH_CALUDE_max_candy_types_l2138_213817


namespace NUMINAMATH_CALUDE_square_area_l2138_213818

-- Define the square WXYZ
structure Square (W X Y Z : ℝ × ℝ) : Prop where
  is_square : true  -- We assume WXYZ is a square

-- Define the points P and Q
def P : ℝ × ℝ := sorry
def Q : ℝ × ℝ := sorry

-- Define the properties of the square and points
def square_properties (W X Y Z : ℝ × ℝ) (P Q : ℝ × ℝ) : Prop :=
  Square W X Y Z ∧
  (∃ (t : ℝ), t > 0 ∧ t < 1 ∧ P = (1 - t) • X + t • Y) ∧  -- P is on XY
  (∃ (s : ℝ), s > 0 ∧ s < 1 ∧ Q = (1 - s) • W + s • Z) ∧  -- Q is on WZ
  (Y.1 - P.1)^2 + (Y.2 - P.2)^2 = 16 ∧  -- YP = 4
  (Q.1 - Z.1)^2 + (Q.2 - Z.2)^2 = 9    -- QZ = 3

-- Define the angle trisection property
def angle_trisected (W P Q : ℝ × ℝ) : Prop :=
  ∃ (θ : ℝ), θ > 0 ∧ 
    (P.2 - W.2) / (P.1 - W.1) = Real.tan θ ∧
    (Q.2 - W.2) / (Q.1 - W.1) = Real.tan (2 * θ)

-- Theorem statement
theorem square_area (W X Y Z : ℝ × ℝ) (P Q : ℝ × ℝ) :
  square_properties W X Y Z P Q →
  angle_trisected W P Q →
  (Y.1 - W.1)^2 + (Y.2 - W.2)^2 = 48 := by
  sorry

end NUMINAMATH_CALUDE_square_area_l2138_213818


namespace NUMINAMATH_CALUDE_remainder_sum_mod_three_l2138_213854

theorem remainder_sum_mod_three
  (a b c d : ℕ)
  (ha : a % 6 = 4)
  (hb : b % 6 = 4)
  (hc : c % 6 = 4)
  (hd : d % 6 = 4) :
  (a + b + c + d) % 3 = 1 := by
  sorry

end NUMINAMATH_CALUDE_remainder_sum_mod_three_l2138_213854


namespace NUMINAMATH_CALUDE_weight_of_a_l2138_213874

/-- Given the weights of 5 people A, B, C, D, and E, prove that A weighs 64 kg -/
theorem weight_of_a (a b c d e : ℝ) : 
  (a + b + c) / 3 = 84 →
  (a + b + c + d) / 4 = 80 →
  e = d + 6 →
  (b + c + d + e) / 4 = 79 →
  a = 64 := by
sorry

end NUMINAMATH_CALUDE_weight_of_a_l2138_213874


namespace NUMINAMATH_CALUDE_inequality_proof_l2138_213894

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h : a / (b + c) = b / (c + a) - c / (a + b)) :
  b / (c + a) ≥ (Real.sqrt 17 - 1) / 4 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2138_213894


namespace NUMINAMATH_CALUDE_mean_height_correct_l2138_213810

/-- The number of players on the basketball team -/
def num_players : ℕ := 16

/-- The total height of all players in inches -/
def total_height : ℕ := 965

/-- The mean height of the players -/
def mean_height : ℚ := 60.31

/-- Theorem stating that the mean height is correct given the number of players and total height -/
theorem mean_height_correct : 
  (total_height : ℚ) / (num_players : ℚ) = mean_height := by sorry

end NUMINAMATH_CALUDE_mean_height_correct_l2138_213810


namespace NUMINAMATH_CALUDE_initial_socks_count_l2138_213853

theorem initial_socks_count (S : ℕ) : 
  (S ≥ 4) →
  (∃ (remaining : ℕ), remaining = S - 4) →
  (∃ (after_donation : ℕ), after_donation = (remaining : ℚ) * (1 / 3 : ℚ)) →
  (after_donation + 13 = 25) →
  S = 40 :=
by sorry

end NUMINAMATH_CALUDE_initial_socks_count_l2138_213853


namespace NUMINAMATH_CALUDE_min_value_sum_squares_l2138_213833

theorem min_value_sum_squares (x y : ℝ) (h : x + y = 4) :
  ∃ (m : ℝ), (∀ a b : ℝ, a + b = 4 → x^2 + y^2 ≤ a^2 + b^2) ∧ m = 8 := by
  sorry

end NUMINAMATH_CALUDE_min_value_sum_squares_l2138_213833


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l2138_213866

theorem arithmetic_sequence_problem (a b c d : ℝ) : 
  (∃ r : ℝ, b = a + r ∧ c = b + r ∧ d = c + r) →  -- arithmetic sequence condition
  (a + b + c + d = 26) →                         -- sum condition
  (b * c = 40) →                                 -- product condition
  ((a, b, c, d) = (2, 5, 8, 11) ∨ (a, b, c, d) = (11, 8, 5, 2)) := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l2138_213866


namespace NUMINAMATH_CALUDE_hyperbola_vertices_distance_hyperbola_vertices_distance_proof_l2138_213830

/-- The distance between the vertices of a hyperbola with equation x^2/16 - y^2/9 = 1 is 8 -/
theorem hyperbola_vertices_distance : ℝ :=
  let hyperbola_equation (x y : ℝ) := x^2/16 - y^2/9 = 1
  let vertices_distance := 8
  vertices_distance

/-- Proof of the theorem -/
theorem hyperbola_vertices_distance_proof : hyperbola_vertices_distance = 8 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_vertices_distance_hyperbola_vertices_distance_proof_l2138_213830


namespace NUMINAMATH_CALUDE_leftover_tarts_sum_l2138_213872

theorem leftover_tarts_sum (cherry_tarts blueberry_tarts peach_tarts : ℝ) 
  (h1 : cherry_tarts = 0.08)
  (h2 : blueberry_tarts = 0.75)
  (h3 : peach_tarts = 0.08) :
  cherry_tarts + blueberry_tarts + peach_tarts = 0.91 := by
    sorry

end NUMINAMATH_CALUDE_leftover_tarts_sum_l2138_213872


namespace NUMINAMATH_CALUDE_fraction_2011_l2138_213861

/-- Euler's totient function -/
def phi (n : ℕ) : ℕ := sorry

/-- The sequence of fractions as described in the problem -/
def fraction_sequence : ℕ → ℚ := sorry

/-- The sum of Euler's totient function up to n -/
def phi_sum (n : ℕ) : ℕ := sorry

theorem fraction_2011 : fraction_sequence 2011 = 49 / 111 := by sorry

end NUMINAMATH_CALUDE_fraction_2011_l2138_213861


namespace NUMINAMATH_CALUDE_parallel_vectors_x_value_l2138_213875

def vector_a (x : ℝ) : ℝ × ℝ := (x, 1)
def vector_b : ℝ × ℝ := (2, -3)

def parallel (v w : ℝ × ℝ) : Prop :=
  ∃ (k : ℝ), v.1 = k * w.1 ∧ v.2 = k * w.2

theorem parallel_vectors_x_value :
  ∀ x : ℝ, parallel (vector_a x) vector_b → x = -2/3 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_x_value_l2138_213875


namespace NUMINAMATH_CALUDE_fraction_simplification_l2138_213848

theorem fraction_simplification : (2 / (3 + Real.sqrt 5)) * (2 / (3 - Real.sqrt 5)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l2138_213848


namespace NUMINAMATH_CALUDE_frank_candies_l2138_213850

def frank_tickets_game1 : ℕ := 33
def frank_tickets_game2 : ℕ := 9
def candy_cost : ℕ := 6

theorem frank_candies : 
  (frank_tickets_game1 + frank_tickets_game2) / candy_cost = 7 := by sorry

end NUMINAMATH_CALUDE_frank_candies_l2138_213850


namespace NUMINAMATH_CALUDE_perception_permutations_l2138_213809

def word_length : ℕ := 10
def p_count : ℕ := 2
def e_count : ℕ := 2

theorem perception_permutations :
  (word_length.factorial) / (p_count.factorial * e_count.factorial) = 907200 := by
  sorry

end NUMINAMATH_CALUDE_perception_permutations_l2138_213809


namespace NUMINAMATH_CALUDE_fixed_point_of_exponential_function_l2138_213849

theorem fixed_point_of_exponential_function (a : ℝ) (ha : a > 0) :
  let f : ℝ → ℝ := λ x ↦ a^(x-1) + 3
  f 1 = 4 := by sorry

end NUMINAMATH_CALUDE_fixed_point_of_exponential_function_l2138_213849


namespace NUMINAMATH_CALUDE_fruit_store_problem_l2138_213841

/-- The price of apples in yuan per kg -/
def apple_price : ℝ := 8

/-- The price of pears in yuan per kg -/
def pear_price : ℝ := 6

/-- The maximum number of kg of apples that can be purchased -/
def max_apple_kg : ℝ := 5

theorem fruit_store_problem :
  (∀ x y : ℝ, x + 3 * y = 26 ∧ 2 * x + y = 22 →
    x = apple_price ∧ y = pear_price) ∧
  (∀ m : ℝ, 8 * m + 6 * (15 - m) ≤ 100 → m ≤ max_apple_kg) :=
by sorry

end NUMINAMATH_CALUDE_fruit_store_problem_l2138_213841


namespace NUMINAMATH_CALUDE_gcd_364_154_l2138_213881

theorem gcd_364_154 : Nat.gcd 364 154 = 14 := by
  sorry

end NUMINAMATH_CALUDE_gcd_364_154_l2138_213881


namespace NUMINAMATH_CALUDE_tan_sum_reciprocal_l2138_213811

theorem tan_sum_reciprocal (u v : ℝ) 
  (h1 : (Real.sin u / Real.cos v) + (Real.sin v / Real.cos u) = 2)
  (h2 : (Real.cos u / Real.sin v) + (Real.cos v / Real.sin u) = 3) :
  (Real.tan u / Real.tan v) + (Real.tan v / Real.tan u) = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_tan_sum_reciprocal_l2138_213811


namespace NUMINAMATH_CALUDE_rhombus_diagonal_length_l2138_213813

/-- 
Proves that in a rhombus with an area of 120 cm² and one diagonal of 20 cm, 
the length of the other diagonal is 12 cm.
-/
theorem rhombus_diagonal_length 
  (area : ℝ) 
  (diagonal1 : ℝ) 
  (diagonal2 : ℝ) 
  (h1 : area = 120) 
  (h2 : diagonal1 = 20) 
  (h3 : area = (diagonal1 * diagonal2) / 2) : 
  diagonal2 = 12 := by
sorry

end NUMINAMATH_CALUDE_rhombus_diagonal_length_l2138_213813


namespace NUMINAMATH_CALUDE_solve_equation_l2138_213800

theorem solve_equation (x : ℝ) (h : 5 * x - 3 = 15 * x + 21) : 3 * (x + 10) = 22.8 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l2138_213800


namespace NUMINAMATH_CALUDE_q_implies_k_range_p_or_q_and_not_p_and_q_implies_k_range_l2138_213838

-- Define proposition p
def p (k : ℝ) : Prop := ∀ x : ℝ, x^2 - k*x + 2*k + 5 ≥ 0

-- Define proposition q
def q (k : ℝ) : Prop := ∃ a b : ℝ, a > 0 ∧ b < 0 ∧ a = 4 - k ∧ b = 1 - k

-- Theorem 1
theorem q_implies_k_range (k : ℝ) : q k → 1 < k ∧ k < 4 := by sorry

-- Theorem 2
theorem p_or_q_and_not_p_and_q_implies_k_range (k : ℝ) : 
  (p k ∨ q k) ∧ ¬(p k ∧ q k) → (-2 ≤ k ∧ k ≤ 1) ∨ (4 ≤ k ∧ k ≤ 10) := by sorry

end NUMINAMATH_CALUDE_q_implies_k_range_p_or_q_and_not_p_and_q_implies_k_range_l2138_213838


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l2138_213890

theorem sqrt_equation_solution :
  ∃! z : ℚ, Real.sqrt (5 - 4 * z) = 7 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l2138_213890


namespace NUMINAMATH_CALUDE_rectangular_field_width_l2138_213868

theorem rectangular_field_width (width length : ℝ) (perimeter : ℝ) : 
  length = (7 / 5) * width →
  perimeter = 2 * length + 2 * width →
  perimeter = 384 →
  width = 80 :=
by
  sorry

end NUMINAMATH_CALUDE_rectangular_field_width_l2138_213868


namespace NUMINAMATH_CALUDE_cubic_sum_theorem_l2138_213895

theorem cubic_sum_theorem (x y z : ℝ) 
  (h1 : x + y + z = 3) 
  (h2 : x*y + y*z + z*x = -3) 
  (h3 : x*y*z = -3) : 
  x^3 + y^3 + z^3 = 45 := by
sorry

end NUMINAMATH_CALUDE_cubic_sum_theorem_l2138_213895


namespace NUMINAMATH_CALUDE_raine_change_calculation_l2138_213827

/-- Calculate the change Raine receives after purchasing items from a gift shop -/
theorem raine_change_calculation (bracelet_price gold_necklace_price mug_price : ℕ)
  (bracelet_quantity gold_necklace_quantity mug_quantity : ℕ)
  (paid_amount : ℕ) :
  bracelet_price = 15 →
  gold_necklace_price = 10 →
  mug_price = 20 →
  bracelet_quantity = 3 →
  gold_necklace_quantity = 2 →
  mug_quantity = 1 →
  paid_amount = 100 →
  paid_amount - (bracelet_price * bracelet_quantity + 
                 gold_necklace_price * gold_necklace_quantity + 
                 mug_price * mug_quantity) = 15 := by
  sorry

end NUMINAMATH_CALUDE_raine_change_calculation_l2138_213827


namespace NUMINAMATH_CALUDE_incorrect_conclusion_l2138_213829

theorem incorrect_conclusion (a b c : ℝ) (h1 : 0 < a) (h2 : a < 1) (h3 : b > c) (h4 : c > 0) :
  ¬((a / b) > (a / c)) :=
sorry

end NUMINAMATH_CALUDE_incorrect_conclusion_l2138_213829


namespace NUMINAMATH_CALUDE_concert_attendance_l2138_213891

/-- The number of buses used for the concert -/
def num_buses : ℕ := 12

/-- The number of students each bus can carry -/
def students_per_bus : ℕ := 57

/-- The total number of students who went to the concert -/
def total_students : ℕ := num_buses * students_per_bus

theorem concert_attendance : total_students = 684 := by
  sorry

end NUMINAMATH_CALUDE_concert_attendance_l2138_213891


namespace NUMINAMATH_CALUDE_chalkboard_area_l2138_213889

theorem chalkboard_area (width : ℝ) (length : ℝ) (area : ℝ) : 
  width = 3.5 →
  length = 2.3 * width →
  area = length * width →
  area = 28.175 := by
sorry

end NUMINAMATH_CALUDE_chalkboard_area_l2138_213889


namespace NUMINAMATH_CALUDE_house_price_proof_l2138_213859

theorem house_price_proof (price_first : ℝ) (price_second : ℝ) : 
  price_second = 2 * price_first →
  price_first + price_second = 600000 →
  price_first = 200000 := by
sorry

end NUMINAMATH_CALUDE_house_price_proof_l2138_213859


namespace NUMINAMATH_CALUDE_extreme_points_count_a_range_l2138_213897

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (x - 1) * Real.exp x - a * x^2

def has_extreme_points (n : ℕ) (f : ℝ → ℝ) : Prop :=
  ∃ (S : Finset ℝ), S.card = n ∧ ∀ x ∈ S, (deriv f) x = 0 ∧
    ∃ ε > 0, ∀ y ∈ Set.Ioo (x - ε) (x + ε), y ≠ x → (deriv f) y ≠ 0

theorem extreme_points_count (a : ℝ) :
  (a ≤ 0 → has_extreme_points 1 (f a)) ∧
  (0 < a ∧ a < 1/2 → has_extreme_points 2 (f a)) ∧
  (a = 1/2 → has_extreme_points 0 (f a)) ∧
  (a > 1/2 → has_extreme_points 2 (f a)) :=
sorry

theorem a_range (a : ℝ) :
  (∀ x : ℝ, f a x + Real.exp x ≥ x^3 + x) → a ≤ Real.exp 1 - 2 :=
sorry

end NUMINAMATH_CALUDE_extreme_points_count_a_range_l2138_213897


namespace NUMINAMATH_CALUDE_work_isothermal_expansion_l2138_213877

/-- Work done during isothermal expansion of an ideal gas -/
theorem work_isothermal_expansion 
  (m μ R T V₁ V₂ : ℝ) 
  (hm : m > 0) 
  (hμ : μ > 0) 
  (hR : R > 0) 
  (hT : T > 0) 
  (hV₁ : V₁ > 0) 
  (hV₂ : V₂ > 0) 
  (hexpand : V₂ > V₁) :
  ∃ A : ℝ, A = (m / μ) * R * T * Real.log (V₂ / V₁) ∧
  (∀ V : ℝ, V > 0 → (m / μ) * R * T = V * (m / μ) * R * T / V) :=
sorry

end NUMINAMATH_CALUDE_work_isothermal_expansion_l2138_213877


namespace NUMINAMATH_CALUDE_no_linear_term_implies_m_value_l2138_213826

theorem no_linear_term_implies_m_value (m : ℝ) : 
  (∀ x : ℝ, (x^2 - x + m) * (x - 8) = x^3 - 9*x^2 + 0*x + (-8*m)) → m = -8 := by
  sorry

end NUMINAMATH_CALUDE_no_linear_term_implies_m_value_l2138_213826


namespace NUMINAMATH_CALUDE_space_station_arrangements_count_l2138_213871

/-- The number of ways to distribute n distinguishable objects into k distinguishable boxes,
    with each box containing at least min and at most max objects. -/
def distribute (n k min max : ℕ) : ℕ := sorry

/-- The number of ways to distribute 6 distinguishable objects into 3 distinguishable boxes,
    with each box containing at least 1 and at most 3 objects. -/
def space_station_arrangements : ℕ := distribute 6 3 1 3

theorem space_station_arrangements_count : space_station_arrangements = 450 := by sorry

end NUMINAMATH_CALUDE_space_station_arrangements_count_l2138_213871


namespace NUMINAMATH_CALUDE_biking_distance_l2138_213840

/-- Calculates the distance traveled given a constant rate and time -/
def distance (rate : ℝ) (time : ℝ) : ℝ := rate * time

/-- Proves that biking at 8 miles per hour for 2.5 hours results in a distance of 20 miles -/
theorem biking_distance :
  let rate : ℝ := 8
  let time : ℝ := 2.5
  distance rate time = 20 := by sorry

end NUMINAMATH_CALUDE_biking_distance_l2138_213840


namespace NUMINAMATH_CALUDE_abc_ordering_l2138_213844

noncomputable def a : ℝ := (1/2)^(1/4 : ℝ)
noncomputable def b : ℝ := (1/3)^(1/2 : ℝ)
noncomputable def c : ℝ := (1/4)^(1/3 : ℝ)

theorem abc_ordering : b < c ∧ c < a := by sorry

end NUMINAMATH_CALUDE_abc_ordering_l2138_213844


namespace NUMINAMATH_CALUDE_base_number_proof_l2138_213815

theorem base_number_proof (x : ℝ) (n : ℕ) 
  (h1 : x^(2*n) + x^(2*n) + x^(2*n) + x^(2*n) = 4^28) 
  (h2 : n = 27) : 
  x = 2 := by
  sorry

end NUMINAMATH_CALUDE_base_number_proof_l2138_213815


namespace NUMINAMATH_CALUDE_complex_fraction_equals_four_l2138_213887

theorem complex_fraction_equals_four :
  1 + (1 / (1 - (1 / (1 + (1 / 2))))) = 4 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_equals_four_l2138_213887


namespace NUMINAMATH_CALUDE_spring_percentage_is_ten_percent_l2138_213804

/-- The percentage of students who chose Spring -/
def spring_percentage (total : ℕ) (spring : ℕ) : ℚ :=
  (spring : ℚ) / (total : ℚ) * 100

/-- Theorem: The percentage of students who chose Spring is 10% -/
theorem spring_percentage_is_ten_percent :
  spring_percentage 10 1 = 10 := by
  sorry

end NUMINAMATH_CALUDE_spring_percentage_is_ten_percent_l2138_213804


namespace NUMINAMATH_CALUDE_decreasing_quadratic_condition_l2138_213888

/-- A function f(x) = ax^2 - b that is decreasing on (-∞, 0) -/
def DecreasingQuadratic (a b : ℝ) : ℝ → ℝ := fun x ↦ a * x^2 - b

/-- The property of being decreasing on (-∞, 0) -/
def IsDecreasingOnNegatives (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y ∧ y < 0 → f x > f y

theorem decreasing_quadratic_condition (a b : ℝ) :
  IsDecreasingOnNegatives (DecreasingQuadratic a b) → a > 0 ∧ b ∈ Set.univ := by
  sorry

end NUMINAMATH_CALUDE_decreasing_quadratic_condition_l2138_213888


namespace NUMINAMATH_CALUDE_i_power_2016_l2138_213824

/-- The complex unit i -/
def i : ℂ := Complex.I

/-- Given properties of i -/
axiom i_power_1 : i^1 = i
axiom i_power_2 : i^2 = -1
axiom i_power_3 : i^3 = -i
axiom i_power_4 : i^4 = 1
axiom i_power_5 : i^5 = i

/-- Theorem: i^2016 = 1 -/
theorem i_power_2016 : i^2016 = 1 := by
  sorry

end NUMINAMATH_CALUDE_i_power_2016_l2138_213824


namespace NUMINAMATH_CALUDE_three_five_power_sum_l2138_213869

theorem three_five_power_sum (x y : ℕ+) (h : 3^(x.val) * 5^(y.val) = 225) : x.val + y.val = 4 := by
  sorry

end NUMINAMATH_CALUDE_three_five_power_sum_l2138_213869


namespace NUMINAMATH_CALUDE_sweet_cookies_eaten_l2138_213860

theorem sweet_cookies_eaten (initial_sweet : ℕ) (final_sweet : ℕ) (eaten_sweet : ℕ) :
  initial_sweet = final_sweet + eaten_sweet →
  eaten_sweet = initial_sweet - final_sweet :=
by sorry

end NUMINAMATH_CALUDE_sweet_cookies_eaten_l2138_213860


namespace NUMINAMATH_CALUDE_fuel_station_problem_l2138_213820

/-- Represents the number of trucks filled up at a fuel station. -/
def num_trucks : ℕ := 2

theorem fuel_station_problem :
  let service_cost : ℚ := 21/10
  let fuel_cost_per_liter : ℚ := 7/10
  let num_minivans : ℕ := 3
  let total_cost : ℚ := 3472/10
  let minivan_capacity : ℚ := 65
  let truck_capacity : ℚ := minivan_capacity * 220/100
  
  let minivan_fuel_cost : ℚ := num_minivans * minivan_capacity * fuel_cost_per_liter
  let minivan_service_cost : ℚ := num_minivans * service_cost
  let total_minivan_cost : ℚ := minivan_fuel_cost + minivan_service_cost
  
  let truck_cost : ℚ := total_cost - total_minivan_cost
  let single_truck_fuel_cost : ℚ := truck_capacity * fuel_cost_per_liter
  let single_truck_total_cost : ℚ := single_truck_fuel_cost + service_cost
  
  num_trucks = (truck_cost / single_truck_total_cost).num :=
by sorry

#check fuel_station_problem

end NUMINAMATH_CALUDE_fuel_station_problem_l2138_213820


namespace NUMINAMATH_CALUDE_bus_speed_excluding_stoppages_l2138_213803

/-- Given a bus that stops for 45 minutes per hour and has an average speed of 15 km/hr including stoppages,
    prove that its average speed excluding stoppages is 60 km/hr. -/
theorem bus_speed_excluding_stoppages (stop_time : ℝ) (avg_speed_with_stops : ℝ) 
  (h1 : stop_time = 45) 
  (h2 : avg_speed_with_stops = 15) :
  let moving_time : ℝ := 60 - stop_time
  let speed_excluding_stops : ℝ := (avg_speed_with_stops * 60) / moving_time
  speed_excluding_stops = 60 := by
sorry

end NUMINAMATH_CALUDE_bus_speed_excluding_stoppages_l2138_213803


namespace NUMINAMATH_CALUDE_problem_statement_l2138_213865

theorem problem_statement (a b c d e : ℝ) 
  (h1 : a * b = 1)  -- a and b are reciprocals
  (h2 : c + d = 0)  -- c and d are opposites
  (h3 : e < 0)      -- e is negative
  (h4 : |e| = 1)    -- absolute value of e is 1
  : (-a*b)^2009 - (c+d)^2010 - e^2011 = 0 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l2138_213865


namespace NUMINAMATH_CALUDE_distance_after_two_hours_l2138_213898

/-- Anna's walking speed in miles per minute -/
def anna_speed : ℚ := 1 / 20

/-- Mark's jogging speed in miles per minute -/
def mark_speed : ℚ := 3 / 40

/-- Duration of walking in minutes -/
def duration : ℕ := 120

/-- The distance between Anna and Mark after walking for the given duration -/
def distance_apart : ℚ := anna_speed * duration + mark_speed * duration

theorem distance_after_two_hours :
  distance_apart = 15 := by sorry

end NUMINAMATH_CALUDE_distance_after_two_hours_l2138_213898


namespace NUMINAMATH_CALUDE_inequality_proof_l2138_213858

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a / Real.sqrt (a^2 + 8*b*c)) + (b / Real.sqrt (b^2 + 8*c*a)) + (c / Real.sqrt (c^2 + 8*a*b)) ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2138_213858


namespace NUMINAMATH_CALUDE_fair_attendance_this_year_l2138_213835

def fair_attendance (this_year next_year last_year : ℕ) : Prop :=
  (next_year = 2 * this_year) ∧
  (last_year = next_year - 200) ∧
  (this_year + next_year + last_year = 2800)

theorem fair_attendance_this_year :
  ∃ (this_year next_year last_year : ℕ),
    fair_attendance this_year next_year last_year ∧ this_year = 600 :=
by
  sorry

end NUMINAMATH_CALUDE_fair_attendance_this_year_l2138_213835


namespace NUMINAMATH_CALUDE_complex_equation_system_l2138_213837

theorem complex_equation_system (p q r s t u : ℂ) 
  (hp : p ≠ 0) (hq : q ≠ 0) (hr : r ≠ 0) (hs : s ≠ 0) (ht : t ≠ 0) (hu : u ≠ 0)
  (eq1 : p = (q + r) / (s - 3))
  (eq2 : q = (p + r) / (t - 3))
  (eq3 : r = (p + q) / (u - 3))
  (eq4 : s * t + s * u + t * u = 7)
  (eq5 : s + t + u = 4) :
  s * t * u = 6 := by
sorry

end NUMINAMATH_CALUDE_complex_equation_system_l2138_213837


namespace NUMINAMATH_CALUDE_middle_carriages_passengers_l2138_213828

/-- Represents a train with carriages and passengers -/
structure Train where
  num_carriages : Nat
  total_passengers : Nat
  block_passengers : Nat
  block_size : Nat

/-- Calculates the number of passengers in the middle two carriages -/
def middle_two_passengers (t : Train) : Nat :=
  t.total_passengers - (4 * t.block_passengers - 3 * t.total_passengers)

/-- Theorem stating that for a train with given specifications, 
    the middle two carriages contain 96 passengers -/
theorem middle_carriages_passengers 
  (t : Train) 
  (h1 : t.num_carriages = 18) 
  (h2 : t.total_passengers = 700) 
  (h3 : t.block_passengers = 199) 
  (h4 : t.block_size = 5) : 
  middle_two_passengers t = 96 := by
  sorry

end NUMINAMATH_CALUDE_middle_carriages_passengers_l2138_213828


namespace NUMINAMATH_CALUDE_number_subtracted_from_x_l2138_213864

-- Define the problem conditions
def problem_conditions (x y z a : ℤ) : Prop :=
  ((x - a) * (y - 5) * (z - 2) = 1000) ∧
  (∀ (x' y' z' : ℤ), ((x' - a) * (y' - 5) * (z' - 2) = 1000) → (x + y + z ≤ x' + y' + z')) ∧
  (x + y + z = 7)

-- State the theorem
theorem number_subtracted_from_x :
  ∃ (x y z a : ℤ), problem_conditions x y z a ∧ a = -30 :=
sorry

end NUMINAMATH_CALUDE_number_subtracted_from_x_l2138_213864


namespace NUMINAMATH_CALUDE_price_reduction_for_target_profit_max_profit_price_reduction_l2138_213886

/-- Profit function given price reduction x -/
def profit (x : ℝ) : ℝ := (80 - x) * (40 + 2 * x)

/-- Theorem for part 1 of the problem -/
theorem price_reduction_for_target_profit :
  profit 40 = 4800 := by sorry

/-- Theorem for part 2 of the problem -/
theorem max_profit_price_reduction :
  ∀ x : ℝ, profit x ≤ profit 30 ∧ profit 30 = 5000 := by sorry

end NUMINAMATH_CALUDE_price_reduction_for_target_profit_max_profit_price_reduction_l2138_213886


namespace NUMINAMATH_CALUDE_modulus_of_z_l2138_213878

-- Define the complex number z
def z : ℂ := Complex.I * (2 - Complex.I)

-- State the theorem
theorem modulus_of_z : Complex.abs z = Real.sqrt 5 := by sorry

end NUMINAMATH_CALUDE_modulus_of_z_l2138_213878


namespace NUMINAMATH_CALUDE_difference_of_three_times_number_and_five_l2138_213814

theorem difference_of_three_times_number_and_five (x : ℝ) : 3 * x - 5 = 15 → 3 * x - 5 = 15 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_three_times_number_and_five_l2138_213814


namespace NUMINAMATH_CALUDE_distance_XY_is_1000_l2138_213819

/-- The distance between two points X and Y --/
def distance : ℝ := sorry

/-- The time taken to travel from X to Y --/
def time_XY : ℝ := 10

/-- The time taken to travel from Y to X --/
def time_YX : ℝ := 4

/-- The average speed for the entire journey --/
def avg_speed : ℝ := 142.85714285714286

/-- Theorem stating that the distance between X and Y is 1000 miles --/
theorem distance_XY_is_1000 : distance = 1000 := by sorry

end NUMINAMATH_CALUDE_distance_XY_is_1000_l2138_213819


namespace NUMINAMATH_CALUDE_jamie_score_l2138_213839

theorem jamie_score (team_total : ℝ) (num_players : ℕ) (other_players_avg : ℝ) 
  (h1 : team_total = 60)
  (h2 : num_players = 6)
  (h3 : other_players_avg = 4.8) : 
  team_total - (num_players - 1) * other_players_avg = 36 :=
by sorry

end NUMINAMATH_CALUDE_jamie_score_l2138_213839


namespace NUMINAMATH_CALUDE_hardware_store_lcm_l2138_213807

theorem hardware_store_lcm : Nat.lcm 13 (Nat.lcm 19 (Nat.lcm 8 (Nat.lcm 11 (Nat.lcm 17 23)))) = 772616 := by
  sorry

end NUMINAMATH_CALUDE_hardware_store_lcm_l2138_213807


namespace NUMINAMATH_CALUDE_triangle_problem_l2138_213816

-- Define the triangle ABC
structure Triangle :=
  (a b c : ℝ)
  (A B C : ℝ)

-- State the theorem
theorem triangle_problem (abc : Triangle) 
  (h1 : Real.tan (π/4 - abc.C) = Real.sqrt 3 - 2)
  (h2 : abc.c = Real.sqrt 7)
  (h3 : abc.a + abc.b = 5) :
  abc.C = π/3 ∧ 
  (1/2 * abc.a * abc.b * Real.sin abc.C = 3 * Real.sqrt 3 / 2) :=
by sorry

end NUMINAMATH_CALUDE_triangle_problem_l2138_213816


namespace NUMINAMATH_CALUDE_sarahs_laundry_l2138_213876

theorem sarahs_laundry (machine_capacity : ℕ) (sweaters : ℕ) (loads : ℕ) (shirts : ℕ) : 
  machine_capacity = 5 →
  sweaters = 2 →
  loads = 9 →
  shirts = loads * machine_capacity - sweaters →
  shirts = 43 := by
sorry

end NUMINAMATH_CALUDE_sarahs_laundry_l2138_213876


namespace NUMINAMATH_CALUDE_committee_formation_count_l2138_213846

def total_members : ℕ := 12
def committee_size : ℕ := 5
def incompatible_members : ℕ := 2

theorem committee_formation_count :
  (Nat.choose total_members committee_size) -
  (Nat.choose (total_members - incompatible_members) (committee_size - incompatible_members)) = 672 := by
  sorry

end NUMINAMATH_CALUDE_committee_formation_count_l2138_213846


namespace NUMINAMATH_CALUDE_flu_infection_rate_flu_infection_rate_proof_l2138_213832

theorem flu_infection_rate : ℝ → Prop :=
  fun x => (1 + x + x * (1 + x) = 144) → x = 11

-- The proof of the theorem
theorem flu_infection_rate_proof : flu_infection_rate 11 := by
  sorry

end NUMINAMATH_CALUDE_flu_infection_rate_flu_infection_rate_proof_l2138_213832


namespace NUMINAMATH_CALUDE_cost_price_calculation_l2138_213852

/-- The selling price of the computer table -/
def selling_price : ℝ := 5750

/-- The markup percentage applied by the shop owner -/
def markup_percentage : ℝ := 15

/-- The cost price of the computer table -/
def cost_price : ℝ := 5000

/-- Theorem stating that the given cost price is correct based on the selling price and markup -/
theorem cost_price_calculation : 
  selling_price = cost_price * (1 + markup_percentage / 100) := by
  sorry

end NUMINAMATH_CALUDE_cost_price_calculation_l2138_213852


namespace NUMINAMATH_CALUDE_fraction_zero_implies_x_equals_one_l2138_213867

theorem fraction_zero_implies_x_equals_one (x : ℝ) :
  (x^2 - 1) / (x + 1) = 0 → x = 1 := by
sorry

end NUMINAMATH_CALUDE_fraction_zero_implies_x_equals_one_l2138_213867


namespace NUMINAMATH_CALUDE_fraction_simplification_l2138_213884

theorem fraction_simplification :
  (2 * (Real.sqrt 2 + Real.sqrt 6)) / (3 * Real.sqrt (2 + Real.sqrt 3)) = 4 / 3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l2138_213884


namespace NUMINAMATH_CALUDE_angle_A_measure_l2138_213899

/-- A triangle with an internal point creating three smaller triangles -/
structure TriangleWithInternalPoint where
  /-- Angle B of the large triangle -/
  angle_B : ℝ
  /-- Angle C of the large triangle -/
  angle_C : ℝ
  /-- Angle D at the internal point -/
  angle_D : ℝ
  /-- Angle A of one of the smaller triangles -/
  angle_A : ℝ
  /-- The sum of angles in a triangle is 180° -/
  triangle_sum : angle_B + angle_C + angle_D + (180 - angle_A) = 180

/-- Theorem: If m∠B = 50°, m∠C = 40°, and m∠D = 30°, then m∠A = 120° -/
theorem angle_A_measure (t : TriangleWithInternalPoint)
    (hB : t.angle_B = 50)
    (hC : t.angle_C = 40)
    (hD : t.angle_D = 30) :
    t.angle_A = 120 := by
  sorry


end NUMINAMATH_CALUDE_angle_A_measure_l2138_213899


namespace NUMINAMATH_CALUDE_lines_perpendicular_to_plane_are_parallel_l2138_213842

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular and parallel relations
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Line → Line → Prop)

-- State the theorem
theorem lines_perpendicular_to_plane_are_parallel 
  (m n : Line) (α : Plane) :
  perpendicular m α → perpendicular n α → parallel m n :=
sorry

end NUMINAMATH_CALUDE_lines_perpendicular_to_plane_are_parallel_l2138_213842
