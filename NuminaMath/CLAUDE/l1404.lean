import Mathlib

namespace NUMINAMATH_CALUDE_min_value_absolute_sum_l1404_140400

theorem min_value_absolute_sum (x : ℝ) :
  |x + 1| + 2 * |x - 5| + |2 * x - 7| + |(x - 11) / 2| ≥ 45 / 8 ∧
  ∃ x : ℝ, |x + 1| + 2 * |x - 5| + |2 * x - 7| + |(x - 11) / 2| = 45 / 8 :=
by sorry

end NUMINAMATH_CALUDE_min_value_absolute_sum_l1404_140400


namespace NUMINAMATH_CALUDE_quadratic_inequality_empty_solution_set_l1404_140496

theorem quadratic_inequality_empty_solution_set (a : ℝ) : 
  (∀ x : ℝ, a * x^2 + 2 * a * x + 2 ≥ 0) → 0 ≤ a ∧ a ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_empty_solution_set_l1404_140496


namespace NUMINAMATH_CALUDE_max_shoe_pairs_l1404_140446

theorem max_shoe_pairs (initial_pairs : ℕ) (lost_shoes : ℕ) (max_remaining_pairs : ℕ) : 
  initial_pairs = 27 → lost_shoes = 9 → max_remaining_pairs = 18 →
  max_remaining_pairs = initial_pairs - lost_shoes / 2 := by
sorry

end NUMINAMATH_CALUDE_max_shoe_pairs_l1404_140446


namespace NUMINAMATH_CALUDE_modulo_six_equivalence_l1404_140424

theorem modulo_six_equivalence : 47^1987 - 22^1987 ≡ 1 [ZMOD 6] := by sorry

end NUMINAMATH_CALUDE_modulo_six_equivalence_l1404_140424


namespace NUMINAMATH_CALUDE_largest_prime_factor_l1404_140431

theorem largest_prime_factor : 
  (∃ p : ℕ, Nat.Prime p ∧ p ∣ (16^4 + 2 * 16^2 + 1 - 13^4) ∧ 
    ∀ q : ℕ, Nat.Prime q → q ∣ (16^4 + 2 * 16^2 + 1 - 13^4) → q ≤ p) ∧
  Nat.Prime 71 ∧ 
  71 ∣ (16^4 + 2 * 16^2 + 1 - 13^4) :=
by sorry

end NUMINAMATH_CALUDE_largest_prime_factor_l1404_140431


namespace NUMINAMATH_CALUDE_line_through_point_l1404_140403

theorem line_through_point (b : ℚ) : 
  (b * (-3) - (b - 1) * 5 = b - 3) ↔ (b = 8 / 9) := by sorry

end NUMINAMATH_CALUDE_line_through_point_l1404_140403


namespace NUMINAMATH_CALUDE_wage_difference_l1404_140430

theorem wage_difference (w1 w2 : ℝ) 
  (h1 : w1 > 0) 
  (h2 : w2 > 0) 
  (h3 : 0.4 * w2 = 1.6 * (0.2 * w1)) : 
  (w1 - w2) / w1 = 0.2 := by
  sorry

end NUMINAMATH_CALUDE_wage_difference_l1404_140430


namespace NUMINAMATH_CALUDE_vowel_initial_probability_l1404_140410

/-- The probability of selecting a student with vowel initials -/
theorem vowel_initial_probability 
  (total_students : ℕ) 
  (vowels : List Char) 
  (students_per_vowel : ℕ) : 
  total_students = 34 → 
  vowels = ['A', 'E', 'I', 'O', 'U', 'Y'] → 
  students_per_vowel = 2 → 
  (students_per_vowel * vowels.length : ℚ) / total_students = 6 / 17 := by
  sorry

end NUMINAMATH_CALUDE_vowel_initial_probability_l1404_140410


namespace NUMINAMATH_CALUDE_connie_tickets_connie_redeemed_fifty_tickets_l1404_140461

theorem connie_tickets : ℕ → Prop := fun total =>
  let koala := total / 2
  let earbuds := 10
  let bracelets := 15
  (koala + earbuds + bracelets = total) → total = 50

-- Proof
theorem connie_redeemed_fifty_tickets : ∃ total, connie_tickets total :=
  sorry

end NUMINAMATH_CALUDE_connie_tickets_connie_redeemed_fifty_tickets_l1404_140461


namespace NUMINAMATH_CALUDE_melanie_attended_games_l1404_140423

theorem melanie_attended_games 
  (total_games : ℕ) 
  (missed_games : ℕ) 
  (attended_games : ℕ) 
  (h1 : total_games = 64) 
  (h2 : missed_games = 32) 
  (h3 : attended_games = total_games - missed_games) : 
  attended_games = 32 :=
by sorry

end NUMINAMATH_CALUDE_melanie_attended_games_l1404_140423


namespace NUMINAMATH_CALUDE_total_pages_left_to_read_l1404_140411

/-- Calculates the total number of pages left to read from three books -/
def pagesLeftToRead (book1Total book1Read book2Total book2Read book3Total book3Read : ℕ) : ℕ :=
  (book1Total - book1Read) + (book2Total - book2Read) + (book3Total - book3Read)

/-- Theorem: The total number of pages left to read from three books is 1442 -/
theorem total_pages_left_to_read : 
  pagesLeftToRead 563 147 849 389 700 134 = 1442 := by
  sorry

end NUMINAMATH_CALUDE_total_pages_left_to_read_l1404_140411


namespace NUMINAMATH_CALUDE_age_ratio_proof_l1404_140429

def sachin_age : ℕ := 28
def age_difference : ℕ := 8

def rahul_age : ℕ := sachin_age + age_difference

theorem age_ratio_proof : 
  (sachin_age : ℚ) / (rahul_age : ℚ) = 7 / 9 := by sorry

end NUMINAMATH_CALUDE_age_ratio_proof_l1404_140429


namespace NUMINAMATH_CALUDE_system_solution_ratio_l1404_140470

theorem system_solution_ratio (x y c d : ℝ) (h1 : 4*x - 2*y = c) (h2 : 6*y - 12*x = d) (h3 : d ≠ 0) :
  c / d = -1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_ratio_l1404_140470


namespace NUMINAMATH_CALUDE_prob_exactly_four_questions_value_l1404_140426

/-- The probability of correctly answering a single question -/
def p : ℝ := 0.8

/-- The number of questions in the competition -/
def n : ℕ := 5

/-- The event that a contestant exactly answers 4 questions before advancing -/
def exactly_four_questions (outcomes : Fin 4 → Bool) : Prop :=
  outcomes 1 = false ∧ outcomes 2 = true ∧ outcomes 3 = true

/-- The probability of the event that a contestant exactly answers 4 questions before advancing -/
def prob_exactly_four_questions : ℝ :=
  (1 - p) * p * p

theorem prob_exactly_four_questions_value :
  prob_exactly_four_questions = 0.128 := by
  sorry


end NUMINAMATH_CALUDE_prob_exactly_four_questions_value_l1404_140426


namespace NUMINAMATH_CALUDE_quadratic_equation_proof_l1404_140450

theorem quadratic_equation_proof (m : ℝ) (p : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ x^2 - 2*x + m - 1 = 0 ∧ y^2 - 2*y + m - 1 = 0) →
  (p^2 - 2*p + m - 1 = 0) →
  ((p^2 - 2*p + 3) * (m + 4) = 7) →
  m = -3 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_proof_l1404_140450


namespace NUMINAMATH_CALUDE_circle_area_with_diameter_8_l1404_140468

theorem circle_area_with_diameter_8 (π : ℝ) :
  let diameter : ℝ := 8
  let radius : ℝ := diameter / 2
  let area : ℝ := π * radius ^ 2
  area = 16 * π := by sorry

end NUMINAMATH_CALUDE_circle_area_with_diameter_8_l1404_140468


namespace NUMINAMATH_CALUDE_two_distinct_roots_l1404_140421

/-- The equation has exactly two distinct real roots for x when p is in the specified range -/
theorem two_distinct_roots (p : ℝ) : 
  (∃! x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
    Real.sqrt (2*p + 1 - x₁^2) + Real.sqrt (3*x₁ + p + 4) = Real.sqrt (x₁^2 + 9*x₁ + 3*p + 9) ∧
    Real.sqrt (2*p + 1 - x₂^2) + Real.sqrt (3*x₂ + p + 4) = Real.sqrt (x₂^2 + 9*x₂ + 3*p + 9)) ↔
  (-1/4 < p ∧ p ≤ 0) ∨ p ≥ 2 :=
sorry

end NUMINAMATH_CALUDE_two_distinct_roots_l1404_140421


namespace NUMINAMATH_CALUDE_sqrt_200_simplification_l1404_140433

theorem sqrt_200_simplification : Real.sqrt 200 = 10 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_200_simplification_l1404_140433


namespace NUMINAMATH_CALUDE_percent_equality_l1404_140407

theorem percent_equality (x : ℝ) : (60 / 100 * 600 = 50 / 100 * x) → x = 720 := by
  sorry

end NUMINAMATH_CALUDE_percent_equality_l1404_140407


namespace NUMINAMATH_CALUDE_arithmetic_mean_odd_primes_under_30_l1404_140492

def odd_primes_under_30 : List Nat := [3, 5, 7, 11, 13, 17, 19, 23, 29]

def arithmetic_mean (list : List Nat) : Rat :=
  (list.sum : Rat) / list.length

theorem arithmetic_mean_odd_primes_under_30 :
  arithmetic_mean odd_primes_under_30 = 14 := by sorry

end NUMINAMATH_CALUDE_arithmetic_mean_odd_primes_under_30_l1404_140492


namespace NUMINAMATH_CALUDE_sqrt_factor_inside_l1404_140456

theorem sqrt_factor_inside (x : ℝ) (h : x > 0) :
  -2 * Real.sqrt (5/2) = -Real.sqrt 10 :=
by sorry

end NUMINAMATH_CALUDE_sqrt_factor_inside_l1404_140456


namespace NUMINAMATH_CALUDE_tommys_balloons_l1404_140487

theorem tommys_balloons (initial_balloons : ℝ) (mom_gave : ℝ) (total_balloons : ℝ)
  (h1 : mom_gave = 78.5)
  (h2 : total_balloons = 132.25)
  (h3 : total_balloons = initial_balloons + mom_gave) :
  initial_balloons = 53.75 := by
  sorry

end NUMINAMATH_CALUDE_tommys_balloons_l1404_140487


namespace NUMINAMATH_CALUDE_line_equation_from_slope_and_intercept_l1404_140479

/-- Given a line with slope 6 and y-intercept -4, its equation is 6x - y - 4 = 0 -/
theorem line_equation_from_slope_and_intercept :
  ∀ (f : ℝ → ℝ),
  (∀ x y : ℝ, f y - f x = 6 * (y - x)) →  -- slope is 6
  (f 0 = -4) →                           -- y-intercept is -4
  ∀ x : ℝ, 6 * x - f x - 4 = 0 :=
by sorry


end NUMINAMATH_CALUDE_line_equation_from_slope_and_intercept_l1404_140479


namespace NUMINAMATH_CALUDE_f_domain_is_open_interval_l1404_140435

/-- The domain of the function f(x) = ln((3 - x)(x + 1)) -/
def f_domain : Set ℝ :=
  {x : ℝ | (3 - x) * (x + 1) > 0}

/-- Theorem stating that the domain of f(x) = ln((3 - x)(x + 1)) is (-1, 3) -/
theorem f_domain_is_open_interval :
  f_domain = Set.Ioo (-1) 3 :=
by
  sorry

#check f_domain_is_open_interval

end NUMINAMATH_CALUDE_f_domain_is_open_interval_l1404_140435


namespace NUMINAMATH_CALUDE_fraction_sum_20_equals_10_9_l1404_140440

def fraction_sum (n : ℕ) : ℚ :=
  (Finset.range n).sum (λ i => 2 / ((i + 1) * (i + 4)))

theorem fraction_sum_20_equals_10_9 : fraction_sum 20 = 10 / 9 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_20_equals_10_9_l1404_140440


namespace NUMINAMATH_CALUDE_maisie_flyers_l1404_140405

theorem maisie_flyers : 
  ∀ (maisie_flyers : ℕ), 
  (71 : ℕ) = 2 * maisie_flyers + 5 → 
  maisie_flyers = 33 :=
by
  sorry

end NUMINAMATH_CALUDE_maisie_flyers_l1404_140405


namespace NUMINAMATH_CALUDE_jenny_jump_distance_l1404_140497

/-- The sum of the first n terms of a geometric series with first term a and common ratio r -/
def geometric_sum (a : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

/-- The number of jumps Jenny makes -/
def num_jumps : ℕ := 7

/-- The fraction of remaining distance Jenny jumps each time -/
def jump_fraction : ℚ := 1/4

/-- The common ratio of the geometric series representing Jenny's jumps -/
def common_ratio : ℚ := 1 - jump_fraction

theorem jenny_jump_distance :
  geometric_sum jump_fraction common_ratio num_jumps = 14197/16384 := by
  sorry

end NUMINAMATH_CALUDE_jenny_jump_distance_l1404_140497


namespace NUMINAMATH_CALUDE_probability_all_selected_l1404_140409

theorem probability_all_selected (p_ram p_ravi p_raj : ℚ) 
  (h_ram : p_ram = 2/7)
  (h_ravi : p_ravi = 1/5)
  (h_raj : p_raj = 3/8) :
  p_ram * p_ravi * p_raj = 3/140 := by
  sorry

end NUMINAMATH_CALUDE_probability_all_selected_l1404_140409


namespace NUMINAMATH_CALUDE_a_bounds_l1404_140448

theorem a_bounds (a b c d : ℝ) 
  (sum_condition : a + b + c + d = 3)
  (square_sum_condition : a^2 + 2*b^2 + 3*c^2 + 6*d^2 = 5) :
  1 ≤ a ∧ a ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_a_bounds_l1404_140448


namespace NUMINAMATH_CALUDE_binomial_12_11_l1404_140477

theorem binomial_12_11 : (12 : ℕ).choose 11 = 12 := by
  sorry

end NUMINAMATH_CALUDE_binomial_12_11_l1404_140477


namespace NUMINAMATH_CALUDE_min_value_expression_equality_condition_l1404_140422

theorem min_value_expression (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  4 * a^3 + 8 * b^3 + 18 * c^3 + 1 / (9 * a * b * c) ≥ 8 / Real.sqrt 3 :=
by sorry

theorem equality_condition (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (4 * a^3 + 8 * b^3 + 18 * c^3 + 1 / (9 * a * b * c) = 8 / Real.sqrt 3) ↔
  (4 * a^3 = 8 * b^3 ∧ 8 * b^3 = 18 * c^3 ∧ 24 * a * b * c = 1 / (9 * a * b * c)) :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_equality_condition_l1404_140422


namespace NUMINAMATH_CALUDE_solve_system_l1404_140475

theorem solve_system (c d : ℤ) 
  (eq1 : 5 + c = 6 - d) 
  (eq2 : 6 + d = 9 + c) : 
  5 - c = 6 := by
sorry

end NUMINAMATH_CALUDE_solve_system_l1404_140475


namespace NUMINAMATH_CALUDE_grass_field_width_l1404_140406

theorem grass_field_width 
  (length : ℝ) 
  (path_width : ℝ) 
  (path_area : ℝ) 
  (h1 : length = 75) 
  (h2 : path_width = 2.8) 
  (h3 : path_area = 1518.72) : 
  ∃ width : ℝ, 
    (length + 2 * path_width) * (width + 2 * path_width) - length * width = path_area ∧ 
    width = 190.6 := by
  sorry

end NUMINAMATH_CALUDE_grass_field_width_l1404_140406


namespace NUMINAMATH_CALUDE_smallest_three_digit_candy_number_l1404_140444

theorem smallest_three_digit_candy_number : ∃ n : ℕ,
  (100 ≤ n ∧ n < 1000) ∧
  (n - 7) % 9 = 0 ∧
  (n + 9) % 7 = 0 ∧
  (∀ m : ℕ, (100 ≤ m ∧ m < n) → (m - 7) % 9 ≠ 0 ∨ (m + 9) % 7 ≠ 0) ∧
  n = 124 := by
sorry

end NUMINAMATH_CALUDE_smallest_three_digit_candy_number_l1404_140444


namespace NUMINAMATH_CALUDE_number_difference_l1404_140480

theorem number_difference (A B : ℝ) (h1 : A > 0) (h2 : B > 0) (h3 : 0.075 * A = 0.125 * B) (h4 : A = 2430) : A - B = 972 := by
  sorry

end NUMINAMATH_CALUDE_number_difference_l1404_140480


namespace NUMINAMATH_CALUDE_solution_count_is_correct_l1404_140498

/-- The number of groups of integer solutions for the equation xyz = 2009 -/
def solution_count : ℕ := 72

/-- A function that counts the number of groups of integer solutions for xyz = 2009 -/
noncomputable def count_solutions : ℕ :=
  -- Implementation details are omitted
  sorry

/-- Theorem stating that the number of groups of integer solutions for xyz = 2009 is 72 -/
theorem solution_count_is_correct : count_solutions = solution_count := by
  sorry

end NUMINAMATH_CALUDE_solution_count_is_correct_l1404_140498


namespace NUMINAMATH_CALUDE_equation_equivalence_l1404_140455

theorem equation_equivalence (x y : ℝ) : 
  (3 * x^2 + 4 * x + 7 * y + 2 = 0) ∧ (3 * x + 2 * y + 5 = 0) →
  4 * y^2 + 33 * y + 11 = 0 := by
  sorry

end NUMINAMATH_CALUDE_equation_equivalence_l1404_140455


namespace NUMINAMATH_CALUDE_xy_value_l1404_140466

theorem xy_value (x y : ℝ) 
  (h1 : (4:ℝ)^x / (2:ℝ)^(x+y) = 16)
  (h2 : (9:ℝ)^(x+y) / (3:ℝ)^(5*y) = 81) : 
  x * y = 32 := by
  sorry

end NUMINAMATH_CALUDE_xy_value_l1404_140466


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l1404_140460

theorem imaginary_part_of_z (i : ℂ) (h : i * i = -1) : 
  Complex.im (i / (i - 1)) = -1/2 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l1404_140460


namespace NUMINAMATH_CALUDE_local_minimum_implies_a_eq_2_l1404_140486

/-- The function f(x) = x(x-a)² has a local minimum at x = 2 -/
def has_local_minimum (f : ℝ → ℝ) (a : ℝ) : Prop :=
  ∃ δ > 0, ∀ x, |x - 2| < δ → f x ≥ f 2

/-- The function f(x) = x(x-a)² -/
def f (a : ℝ) (x : ℝ) : ℝ := x * (x - a)^2

theorem local_minimum_implies_a_eq_2 :
  ∀ a : ℝ, has_local_minimum (f a) a → a = 2 := by
  sorry

end NUMINAMATH_CALUDE_local_minimum_implies_a_eq_2_l1404_140486


namespace NUMINAMATH_CALUDE_min_fraction_value_l1404_140443

theorem min_fraction_value (a x y : ℕ) (ha : a > 100) (hx : x > 100) (hy : y > 100)
  (h : y^2 - 1 = a^2 * (x^2 - 1)) :
  ∃ (k : ℚ), k = 2 ∧ (∀ (a' x' : ℕ), a' > 100 → x' > 100 → ∃ (y' : ℕ), y' > 100 ∧ 
    y'^2 - 1 = a'^2 * (x'^2 - 1) → (a' : ℚ) / x' ≥ k) ∧
  (∃ (a'' x'' y'' : ℕ), a'' > 100 ∧ x'' > 100 ∧ y'' > 100 ∧
    y''^2 - 1 = a''^2 * (x''^2 - 1) ∧ (a'' : ℚ) / x'' = k) :=
by sorry

end NUMINAMATH_CALUDE_min_fraction_value_l1404_140443


namespace NUMINAMATH_CALUDE_outdoor_dining_area_expansion_l1404_140463

/-- The total area of three sections of an outdoor dining area -/
theorem outdoor_dining_area_expansion (rectangle_area rectangle_width : ℝ)
                                      (semicircle_radius : ℝ)
                                      (triangle_base triangle_height : ℝ) :
  rectangle_area = 35 →
  rectangle_width = 7 →
  semicircle_radius = 4 →
  triangle_base = 5 →
  triangle_height = 6 →
  rectangle_area + (π * semicircle_radius ^ 2) / 2 + (triangle_base * triangle_height) / 2 = 35 + 8 * π + 15 := by
  sorry

end NUMINAMATH_CALUDE_outdoor_dining_area_expansion_l1404_140463


namespace NUMINAMATH_CALUDE_sector_central_angle_l1404_140457

theorem sector_central_angle (r : ℝ) (α : ℝ) : 
  r > 0 → 
  r * α = 6 → 
  (1/2) * r * r * α = 6 → 
  α = 3 := by
sorry

end NUMINAMATH_CALUDE_sector_central_angle_l1404_140457


namespace NUMINAMATH_CALUDE_complex_magnitude_product_l1404_140454

theorem complex_magnitude_product : Complex.abs ((3 * Real.sqrt 2 - 3 * Complex.I) * (2 * Real.sqrt 3 + 6 * Complex.I)) = 36 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_product_l1404_140454


namespace NUMINAMATH_CALUDE_lollipop_consumption_days_l1404_140418

/-- The number of days it takes to finish all lollipops -/
def days_to_finish_lollipops (alison_lollipops henry_extra diane_ratio daily_consumption : ℕ) : ℕ :=
  let henry_lollipops := alison_lollipops + henry_extra
  let diane_lollipops := alison_lollipops * diane_ratio
  let total_lollipops := alison_lollipops + henry_lollipops + diane_lollipops
  total_lollipops / daily_consumption

/-- Theorem stating that it takes 6 days to finish all lollipops under given conditions -/
theorem lollipop_consumption_days :
  days_to_finish_lollipops 60 30 2 45 = 6 := by
  sorry

#eval days_to_finish_lollipops 60 30 2 45

end NUMINAMATH_CALUDE_lollipop_consumption_days_l1404_140418


namespace NUMINAMATH_CALUDE_f_properties_l1404_140445

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ :=
  Real.sin (ω * x) * (Real.sin (ω * x) + Real.cos (ω * x)) - 1/2

theorem f_properties (ω : ℝ) (h_ω : ω > 0) (h_period : (2 * π) / (2 * ω) = 2 * π) :
  let f_max := f ω π
  let f_min := f ω (-π/2)
  let α := π/3
  let β := π/6
  (∀ x ∈ Set.Icc (-π) π, f ω x ≤ f_max) ∧
  (∀ x ∈ Set.Icc (-π) π, f ω x ≥ f_min) ∧
  (f_max = 1/2) ∧
  (f_min = -Real.sqrt 2 / 2) ∧
  (α + 2 * β = 2 * π / 3) ∧
  (f ω (α + π/2) * f ω (2 * β + 3 * π/2) = Real.sqrt 3 / 8) := by
  sorry

end NUMINAMATH_CALUDE_f_properties_l1404_140445


namespace NUMINAMATH_CALUDE_train_speed_l1404_140469

/-- The speed of a train given its length and time to cross a point -/
theorem train_speed (length time : ℝ) (h1 : length = 400) (h2 : time = 20) :
  length / time = 20 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_l1404_140469


namespace NUMINAMATH_CALUDE_vector_magnitude_AB_l1404_140442

/-- The magnitude of the vector from point A(1, 0) to point B(0, -1) is √2 -/
theorem vector_magnitude_AB : Real.sqrt 2 = Real.sqrt ((0 - 1)^2 + (-1 - 0)^2) := by
  sorry

end NUMINAMATH_CALUDE_vector_magnitude_AB_l1404_140442


namespace NUMINAMATH_CALUDE_remainder_mod_five_l1404_140415

theorem remainder_mod_five : (1234 * 1456 * 1789 * 2005 + 123) % 5 = 3 := by
  sorry

end NUMINAMATH_CALUDE_remainder_mod_five_l1404_140415


namespace NUMINAMATH_CALUDE_triangle_square_side_ratio_l1404_140464

theorem triangle_square_side_ratio (t s : ℝ) : 
  (3 * t = 12) →  -- Perimeter of equilateral triangle
  (4 * s = 12) →  -- Perimeter of square
  (t / s = 4 / 3) :=  -- Ratio of side lengths
by
  sorry

end NUMINAMATH_CALUDE_triangle_square_side_ratio_l1404_140464


namespace NUMINAMATH_CALUDE_new_average_score_l1404_140494

theorem new_average_score (n : ℕ) (initial_avg : ℚ) (new_score : ℚ) :
  n = 9 →
  initial_avg = 80 →
  new_score = 100 →
  (n * initial_avg + new_score) / (n + 1) = 82 := by
  sorry

end NUMINAMATH_CALUDE_new_average_score_l1404_140494


namespace NUMINAMATH_CALUDE_tangent_length_is_6_l1404_140474

-- Define the circle C
def circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 - 4*x - 2*y + 1 = 0

-- Define the line passing through the center
def line_equation (x y : ℝ) (a : ℝ) : Prop :=
  x + a*y - 1 = 0

-- Define the center of the circle
def circle_center : ℝ × ℝ := (2, 1)

-- Define point A
def point_A (a : ℝ) : ℝ × ℝ := (-4, a)

-- Define the radius of the circle
def circle_radius : ℝ := 2

-- Theorem statement
theorem tangent_length_is_6 :
  ∃ (a : ℝ),
    line_equation (circle_center.1) (circle_center.2) a ∧
    (∃ (x y : ℝ), circle_equation x y ∧
      ∃ (B : ℝ × ℝ), B.1 = x ∧ B.2 = y ∧
        (point_A a).1 - B.1 = a * (B.2 - (point_A a).2) ∧
        Real.sqrt (((point_A a).1 - B.1)^2 + ((point_A a).2 - B.2)^2) = 6) :=
by sorry

end NUMINAMATH_CALUDE_tangent_length_is_6_l1404_140474


namespace NUMINAMATH_CALUDE_regular_pentagon_perimeter_l1404_140432

/-- The perimeter of a regular pentagon with side length 15 cm is 75 cm. -/
theorem regular_pentagon_perimeter :
  ∀ (side_length perimeter : ℝ),
  side_length = 15 →
  perimeter = 5 * side_length →
  perimeter = 75 :=
by
  sorry

end NUMINAMATH_CALUDE_regular_pentagon_perimeter_l1404_140432


namespace NUMINAMATH_CALUDE_greater_sum_from_inequalities_l1404_140412

theorem greater_sum_from_inequalities (a b c d : ℝ) 
  (h1 : a^2 + b > c^2 + d) 
  (h2 : a + b^2 > c + d^2) 
  (h3 : a ≥ (1/2 : ℝ)) 
  (h4 : b ≥ (1/2 : ℝ)) 
  (h5 : c ≥ (1/2 : ℝ)) 
  (h6 : d ≥ (1/2 : ℝ)) : 
  a + b > c + d := by
  sorry

end NUMINAMATH_CALUDE_greater_sum_from_inequalities_l1404_140412


namespace NUMINAMATH_CALUDE_intersection_line_of_two_circles_l1404_140451

/-- Given two circles with equations x^2 + y^2 + 4x - 4y - 1 = 0 and x^2 + y^2 + 2x - 13 = 0,
    the line passing through their intersection points has the equation x - 2y + 6 = 0 -/
theorem intersection_line_of_two_circles (x y : ℝ) : 
  (x^2 + y^2 + 4*x - 4*y - 1 = 0) ∧ (x^2 + y^2 + 2*x - 13 = 0) →
  (x - 2*y + 6 = 0) :=
by sorry

end NUMINAMATH_CALUDE_intersection_line_of_two_circles_l1404_140451


namespace NUMINAMATH_CALUDE_water_hyacinth_demonstrates_interconnection_and_diversity_l1404_140472

/-- Represents the introduction and effects of water hyacinth -/
structure WaterHyacinthIntroduction where
  introduced_as_fodder : Prop
  rapid_spread : Prop
  decrease_native_species : Prop
  water_pollution : Prop
  increase_mosquitoes : Prop

/-- Represents the philosophical conclusions drawn from the water hyacinth case -/
structure PhilosophicalConclusions where
  universal_interconnection : Prop
  diverse_connections : Prop

/-- Theorem stating that the introduction of water hyacinth demonstrates universal interconnection and diverse connections -/
theorem water_hyacinth_demonstrates_interconnection_and_diversity 
  (wh : WaterHyacinthIntroduction) : PhilosophicalConclusions :=
by sorry

end NUMINAMATH_CALUDE_water_hyacinth_demonstrates_interconnection_and_diversity_l1404_140472


namespace NUMINAMATH_CALUDE_javelin_throw_distance_l1404_140453

theorem javelin_throw_distance (first second third : ℝ) 
  (h1 : first = 2 * second)
  (h2 : first = (1 / 2) * third)
  (h3 : first + second + third = 1050) :
  first = 300 := by
  sorry

end NUMINAMATH_CALUDE_javelin_throw_distance_l1404_140453


namespace NUMINAMATH_CALUDE_tenth_term_of_arithmetic_progression_l1404_140491

/-- The nth term of an arithmetic progression -/
def arithmeticProgressionTerm (a : ℤ) (d : ℤ) (n : ℕ) : ℤ :=
  a + (n - 1) * d

/-- Theorem: The 10th term of an arithmetic progression with first term 8 and common difference 2 is 26 -/
theorem tenth_term_of_arithmetic_progression :
  arithmeticProgressionTerm 8 2 10 = 26 := by
  sorry

end NUMINAMATH_CALUDE_tenth_term_of_arithmetic_progression_l1404_140491


namespace NUMINAMATH_CALUDE_hex_to_decimal_equality_l1404_140467

/-- Represents a hexadecimal digit as a natural number -/
def HexDigit := Fin 16

/-- Converts a base-6 number to decimal -/
def toDecimal (a b c d e : Fin 6) : ℕ :=
  e + 6 * d + 6^2 * c + 6^3 * b + 6^4 * a

/-- The theorem stating that if 3m502₍₆₎ = 4934 in decimal, then m = 4 -/
theorem hex_to_decimal_equality (m : Fin 6) :
  toDecimal 3 m 5 0 2 = 4934 → m = 4 := by
  sorry


end NUMINAMATH_CALUDE_hex_to_decimal_equality_l1404_140467


namespace NUMINAMATH_CALUDE_z_purely_imaginary_and_fourth_quadrant_l1404_140438

def z (m : ℝ) : ℂ := Complex.mk (m^2 - 8*m + 15) (m^2 - 5*m)

theorem z_purely_imaginary_and_fourth_quadrant :
  (∃! m : ℝ, (z m).re = 0 ∧ (z m).im ≠ 0 ∧ m = 3) ∧
  (¬∃ m : ℝ, (z m).re > 0 ∧ (z m).im < 0) := by
  sorry

end NUMINAMATH_CALUDE_z_purely_imaginary_and_fourth_quadrant_l1404_140438


namespace NUMINAMATH_CALUDE_prob_not_losing_l1404_140416

/-- The probability of Hou Yifan winning a chess match against a computer -/
def prob_win : ℝ := 0.65

/-- The probability of a draw in a chess match between Hou Yifan and a computer -/
def prob_draw : ℝ := 0.25

/-- Theorem: The probability of Hou Yifan not losing is 0.9 -/
theorem prob_not_losing : prob_win + prob_draw = 0.9 := by
  sorry

end NUMINAMATH_CALUDE_prob_not_losing_l1404_140416


namespace NUMINAMATH_CALUDE_mary_remaining_money_l1404_140471

-- Define the initial amount Mary received
def initial_amount : ℚ := 150

-- Define the original price of the video game
def game_price : ℚ := 60

-- Define the discount rate for the video game
def game_discount_rate : ℚ := 0.15

-- Define the percentage spent on goggles
def goggles_spend_rate : ℚ := 0.20

-- Define the sales tax rate for the goggles
def goggles_tax_rate : ℚ := 0.08

-- Function to calculate the discounted price of the video game
def discounted_game_price : ℚ :=
  game_price * (1 - game_discount_rate)

-- Function to calculate the amount left after buying the video game
def amount_after_game : ℚ :=
  initial_amount - discounted_game_price

-- Function to calculate the price of the goggles before tax
def goggles_price_before_tax : ℚ :=
  amount_after_game * goggles_spend_rate

-- Function to calculate the total price of the goggles including tax
def goggles_total_price : ℚ :=
  goggles_price_before_tax * (1 + goggles_tax_rate)

-- Theorem stating that Mary has $77.62 left after her shopping trip
theorem mary_remaining_money :
  initial_amount - discounted_game_price - goggles_total_price = 77.62 := by
  sorry

end NUMINAMATH_CALUDE_mary_remaining_money_l1404_140471


namespace NUMINAMATH_CALUDE_lcm_gcd_fraction_lower_bound_lcm_gcd_fraction_bound_achievable_l1404_140408

theorem lcm_gcd_fraction_lower_bound (a b c : ℕ+) (h : a ≠ b ∧ b ≠ c ∧ c ≠ a) :
  (Nat.lcm a b + Nat.lcm b c + Nat.lcm c a : ℚ) / (Nat.gcd a b + Nat.gcd b c + Nat.gcd c a) ≥ 5 / 2 :=
sorry

theorem lcm_gcd_fraction_bound_achievable :
  ∃ (a b c : ℕ+), a ≠ b ∧ b ≠ c ∧ c ≠ a ∧
  (Nat.lcm a b + Nat.lcm b c + Nat.lcm c a : ℚ) / (Nat.gcd a b + Nat.gcd b c + Nat.gcd c a) = 5 / 2 :=
sorry

end NUMINAMATH_CALUDE_lcm_gcd_fraction_lower_bound_lcm_gcd_fraction_bound_achievable_l1404_140408


namespace NUMINAMATH_CALUDE_test_question_count_l1404_140499

theorem test_question_count (total_points : ℕ) (total_questions : ℕ) 
  (h1 : total_points = 200)
  (h2 : total_questions = 30)
  (h3 : ∃ (five_point_count ten_point_count : ℕ), 
    five_point_count + ten_point_count = total_questions ∧
    5 * five_point_count + 10 * ten_point_count = total_points) :
  ∃ (five_point_count : ℕ), five_point_count = 20 ∧
    ∃ (ten_point_count : ℕ), 
      five_point_count + ten_point_count = total_questions ∧
      5 * five_point_count + 10 * ten_point_count = total_points :=
by
  sorry

end NUMINAMATH_CALUDE_test_question_count_l1404_140499


namespace NUMINAMATH_CALUDE_min_distance_to_hyperbola_l1404_140489

/-- The minimum distance between A(4,4) and P(x, 1/x) where x > 0 is √14 -/
theorem min_distance_to_hyperbola :
  let A : ℝ × ℝ := (4, 4)
  let P : ℝ → ℝ × ℝ := fun x ↦ (x, 1/x)
  let distance (x : ℝ) : ℝ := Real.sqrt ((P x).1 - A.1)^2 + ((P x).2 - A.2)^2
  ∀ x > 0, distance x ≥ Real.sqrt 14 ∧ ∃ x₀ > 0, distance x₀ = Real.sqrt 14 :=
by sorry

end NUMINAMATH_CALUDE_min_distance_to_hyperbola_l1404_140489


namespace NUMINAMATH_CALUDE_white_tile_count_in_specific_arrangement_l1404_140459

/-- Represents the tiling arrangement of a large square --/
structure TilingArrangement where
  side_length : ℕ
  black_tile_count : ℕ
  black_tile_size : ℕ
  red_tile_size : ℕ
  white_tile_width : ℕ
  white_tile_length : ℕ

/-- Calculates the number of white tiles in the tiling arrangement --/
def count_white_tiles (t : TilingArrangement) : ℕ :=
  sorry

/-- Theorem stating the number of white tiles in the specific arrangement --/
theorem white_tile_count_in_specific_arrangement :
  ∀ t : TilingArrangement,
    t.side_length = 81 ∧
    t.black_tile_count = 81 ∧
    t.black_tile_size = 1 ∧
    t.red_tile_size = 2 ∧
    t.white_tile_width = 1 ∧
    t.white_tile_length = 2 →
    count_white_tiles t = 2932 :=
  sorry

end NUMINAMATH_CALUDE_white_tile_count_in_specific_arrangement_l1404_140459


namespace NUMINAMATH_CALUDE_lily_pad_half_coverage_l1404_140484

/-- Represents the number of days it takes for lily pads to cover the entire lake -/
def full_coverage_days : ℕ := 39

/-- Represents the growth factor of lily pads per day -/
def daily_growth_factor : ℕ := 2

/-- Calculates the number of days required to cover half the lake -/
def half_coverage_days : ℕ := full_coverage_days - 1

theorem lily_pad_half_coverage :
  half_coverage_days = 38 :=
sorry

end NUMINAMATH_CALUDE_lily_pad_half_coverage_l1404_140484


namespace NUMINAMATH_CALUDE_correct_option_b_l1404_140419

variable (y : ℝ)

theorem correct_option_b (y : ℝ) :
  (-2 * y^3) * (-y) = 2 * y^4 ∧
  (-y^3) * (-y) ≠ -y ∧
  ((-2*y)^3) * (-y) ≠ -8 * y^4 ∧
  ((-y)^12) * (-y) ≠ -3 * y^13 :=
sorry

end NUMINAMATH_CALUDE_correct_option_b_l1404_140419


namespace NUMINAMATH_CALUDE_f_decreasing_on_interval_l1404_140401

-- Define the function
def f (x : ℝ) : ℝ := 2 * x^3 - 6 * x^2 + 11

-- State the theorem
theorem f_decreasing_on_interval :
  ∀ x ∈ Set.Ioo 0 2, StrictMonoOn f (Set.Ioo 0 2) := by
  sorry

end NUMINAMATH_CALUDE_f_decreasing_on_interval_l1404_140401


namespace NUMINAMATH_CALUDE_solution_set_inequality_l1404_140462

/-- Given that the solution set of ax^2 - bx - 1 ≥ 0 is [-1/2, -1/3], 
    prove that the solution set of ax^2 - bx - 1 < 0 is (2, 3) -/
theorem solution_set_inequality (a b : ℝ) : 
  (∀ x, ax^2 - b*x - 1 ≥ 0 ↔ x ∈ Set.Icc (-1/2) (-1/3)) →
  (∀ x, ax^2 - b*x - 1 < 0 ↔ x ∈ Set.Ioo 2 3) :=
by sorry

end NUMINAMATH_CALUDE_solution_set_inequality_l1404_140462


namespace NUMINAMATH_CALUDE_monotonic_increasing_interval_l1404_140447

def f (x : ℝ) := x^2 - 2

theorem monotonic_increasing_interval :
  ∀ x y : ℝ, 0 ≤ x ∧ x < y → f x < f y :=
by sorry

end NUMINAMATH_CALUDE_monotonic_increasing_interval_l1404_140447


namespace NUMINAMATH_CALUDE_littleTwelve_game_count_l1404_140439

/-- Represents a basketball conference with two divisions -/
structure BasketballConference where
  teamsPerDivision : ℕ
  inDivisionGames : ℕ
  crossDivisionGames : ℕ

/-- Calculates the total number of games in the conference -/
def totalGames (conf : BasketballConference) : ℕ :=
  2 * (conf.teamsPerDivision.choose 2 * conf.inDivisionGames) + 
  conf.teamsPerDivision * conf.teamsPerDivision * conf.crossDivisionGames

/-- The Little Twelve Basketball Conference -/
def littleTwelve : BasketballConference := {
  teamsPerDivision := 6
  inDivisionGames := 2
  crossDivisionGames := 1
}

theorem littleTwelve_game_count : totalGames littleTwelve = 96 := by
  sorry

end NUMINAMATH_CALUDE_littleTwelve_game_count_l1404_140439


namespace NUMINAMATH_CALUDE_binary_101111_is_47_l1404_140413

def binary_to_decimal (b : List Bool) : ℕ :=
  (List.enum b).foldl (fun acc (i, bit) => acc + if bit then 2^i else 0) 0

theorem binary_101111_is_47 :
  binary_to_decimal [true, true, true, true, true, false] = 47 := by
  sorry

end NUMINAMATH_CALUDE_binary_101111_is_47_l1404_140413


namespace NUMINAMATH_CALUDE_h_satisfies_equation_l1404_140481

-- Define the polynomial h(x)
def h (x : ℝ) : ℝ := -2*x^5 - x^3 + 5*x^2 - 6*x - 3

-- State the theorem
theorem h_satisfies_equation : 
  ∀ x : ℝ, 2*x^5 + 4*x^3 - 3*x^2 + x + 7 + h x = -x^3 + 2*x^2 - 5*x + 4 :=
by
  sorry

end NUMINAMATH_CALUDE_h_satisfies_equation_l1404_140481


namespace NUMINAMATH_CALUDE_f_period_and_g_max_l1404_140478

noncomputable def f (x : ℝ) : ℝ := Real.sin (x + Real.pi / 2) * Real.cos (x - Real.pi / 3)

noncomputable def g (x : ℝ) : ℝ := 2 * Real.sin (2 * x + Real.pi / 6)

theorem f_period_and_g_max :
  (∃ (p : ℝ), p > 0 ∧ ∀ (x : ℝ), f (x + p) = f x ∧ ∀ (q : ℝ), q > 0 ∧ (∀ (x : ℝ), f (x + q) = f x) → p ≤ q) ∧
  (∃ (M : ℝ), M = 2 ∧ ∀ (x : ℝ), x ∈ Set.Icc 0 (Real.pi / 3) → g x ≤ M) :=
sorry

end NUMINAMATH_CALUDE_f_period_and_g_max_l1404_140478


namespace NUMINAMATH_CALUDE_fraction_equation_l1404_140404

theorem fraction_equation : 45 / (8 - 3/7) = 315/53 := by sorry

end NUMINAMATH_CALUDE_fraction_equation_l1404_140404


namespace NUMINAMATH_CALUDE_button_problem_l1404_140441

/-- Proof of the button problem -/
theorem button_problem (green : ℕ) (yellow : ℕ) (blue : ℕ) (total : ℕ) : 
  green = 90 →
  yellow = green + 10 →
  total = 275 →
  total = green + yellow + blue →
  green - blue = 5 := by sorry

end NUMINAMATH_CALUDE_button_problem_l1404_140441


namespace NUMINAMATH_CALUDE_exclusive_or_implies_disjunction_l1404_140434

theorem exclusive_or_implies_disjunction (p q : Prop) : 
  ((p ∧ ¬q) ∨ (¬p ∧ q)) → (p ∨ q) :=
by
  sorry

end NUMINAMATH_CALUDE_exclusive_or_implies_disjunction_l1404_140434


namespace NUMINAMATH_CALUDE_ninetieth_term_is_13_l1404_140452

def sequence_sum (n : ℕ) : ℕ := n * (n + 1) / 2

theorem ninetieth_term_is_13 :
  ∃ (seq : ℕ → ℕ),
    (∀ n : ℕ, ∀ k : ℕ, k > sequence_sum n → k ≤ sequence_sum (n + 1) → seq k = n + 1) →
    seq 90 = 13 :=
by
  sorry

end NUMINAMATH_CALUDE_ninetieth_term_is_13_l1404_140452


namespace NUMINAMATH_CALUDE_convex_quadrilateral_probability_l1404_140488

/-- The number of points on the circle -/
def n : ℕ := 8

/-- The number of chords to be selected -/
def k : ℕ := 4

/-- The total number of possible chords -/
def total_chords : ℕ := n.choose 2

/-- The number of ways to select k chords from total_chords -/
def total_selections : ℕ := total_chords.choose k

/-- The number of ways to select k points from n points -/
def favorable_outcomes : ℕ := n.choose k

/-- The probability of forming a convex quadrilateral -/
def probability : ℚ := favorable_outcomes / total_selections

theorem convex_quadrilateral_probability : probability = 2 / 585 := by
  sorry

end NUMINAMATH_CALUDE_convex_quadrilateral_probability_l1404_140488


namespace NUMINAMATH_CALUDE_hyperbola_properties_l1404_140436

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := y^2 - x^2 = 4

-- Define what it means for a focus to be on the y-axis
def focus_on_y_axis (h : (ℝ → ℝ → Prop)) : Prop :=
  ∃ c : ℝ, ∀ x y : ℝ, h x y → (x = 0 ∧ y = c) ∨ (x = 0 ∧ y = -c)

-- Define what it means for asymptotes to be perpendicular
def perpendicular_asymptotes (h : (ℝ → ℝ → Prop)) : Prop :=
  ∃ m : ℝ, ∀ x y : ℝ, h x y → (y = m*x ∨ y = -m*x) ∧ m * (-1/m) = -1

-- Theorem statement
theorem hyperbola_properties :
  focus_on_y_axis hyperbola ∧ perpendicular_asymptotes hyperbola :=
sorry

end NUMINAMATH_CALUDE_hyperbola_properties_l1404_140436


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l1404_140465

/-- Given a geometric sequence {aₙ} with a₁ = 1 and common ratio q ≠ 1,
    if -3a₁, -a₂, and a₃ form an arithmetic sequence,
    then the sum of the first 4 terms (S₄) equals -20. -/
theorem geometric_sequence_sum (q : ℝ) (h1 : q ≠ 1) : 
  let a : ℕ → ℝ := λ n => q^(n-1)
  ∀ n, a n = q^(n-1)
  → -3 * (a 1) + (a 3) = 2 * (-a 2)
  → (a 1) = 1
  → (a 1) + (a 2) + (a 3) + (a 4) = -20 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l1404_140465


namespace NUMINAMATH_CALUDE_max_gcd_of_coprime_linear_combination_l1404_140437

theorem max_gcd_of_coprime_linear_combination (m n : ℕ) :
  Nat.gcd m n = 1 →
  ∃ a b : ℕ, Nat.gcd (m + 2000 * n) (n + 2000 * m) = 2000^2 - 1 ∧
            ∀ c d : ℕ, Nat.gcd (c + 2000 * d) (d + 2000 * c) ≤ 2000^2 - 1 :=
by sorry

end NUMINAMATH_CALUDE_max_gcd_of_coprime_linear_combination_l1404_140437


namespace NUMINAMATH_CALUDE_geometric_sequence_property_l1404_140425

/-- A geometric sequence -/
def geometric_sequence (α : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, α (n + 1) = r * α n

/-- The theorem stating that if α_4 · α_5 · α_6 = 27 in a geometric sequence, then α_5 = 3 -/
theorem geometric_sequence_property (α : ℕ → ℝ) :
  geometric_sequence α → α 4 * α 5 * α 6 = 27 → α 5 = 3 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_property_l1404_140425


namespace NUMINAMATH_CALUDE_martin_goldfish_l1404_140473

/-- Calculates the number of goldfish after a given number of weeks -/
def goldfish_after_weeks (initial : ℕ) (die_per_week : ℕ) (buy_per_week : ℕ) (weeks : ℕ) : ℤ :=
  initial - (die_per_week * weeks : ℕ) + (buy_per_week * weeks : ℕ)

/-- Theorem stating the number of goldfish Martin will have after 7 weeks -/
theorem martin_goldfish : goldfish_after_weeks 18 5 3 7 = 4 := by
  sorry

end NUMINAMATH_CALUDE_martin_goldfish_l1404_140473


namespace NUMINAMATH_CALUDE_sum_f_positive_l1404_140449

def f (x : ℝ) : ℝ := x^3 + x

theorem sum_f_positive (a b c : ℝ) (hab : a + b > 0) (hbc : b + c > 0) (hca : c + a > 0) :
  f a + f b + f c > 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_f_positive_l1404_140449


namespace NUMINAMATH_CALUDE_lottery_theorem_l1404_140483

/-- Calculates the remaining amount for fun after deducting taxes, student loan payment, savings, and stock market investment from lottery winnings. -/
def remaining_for_fun (lottery_winnings : ℚ) : ℚ :=
  let after_taxes := lottery_winnings / 2
  let after_student_loans := after_taxes - (after_taxes / 3)
  let after_savings := after_student_loans - 1000
  let stock_investment := 1000 / 5
  after_savings - stock_investment

/-- Theorem stating that given a lottery winning of 12006, the remaining amount for fun is 2802. -/
theorem lottery_theorem : remaining_for_fun 12006 = 2802 := by
  sorry

#eval remaining_for_fun 12006

end NUMINAMATH_CALUDE_lottery_theorem_l1404_140483


namespace NUMINAMATH_CALUDE_initial_boarders_count_l1404_140485

theorem initial_boarders_count (B D : ℕ) : 
  (B : ℚ) / D = 2 / 5 →  -- Original ratio
  ((B + 15 : ℚ) / D = 1 / 2) →  -- New ratio after 15 boarders joined
  B = 60 := by
sorry

end NUMINAMATH_CALUDE_initial_boarders_count_l1404_140485


namespace NUMINAMATH_CALUDE_allan_brought_six_balloons_l1404_140458

/-- The number of balloons Jake initially brought to the park -/
def jake_initial_balloons : ℕ := 2

/-- The number of balloons Jake bought at the park -/
def jake_bought_balloons : ℕ := 3

/-- The difference between Allan's and Jake's balloon count -/
def allan_jake_difference : ℕ := 1

/-- The total number of balloons Jake had in the park -/
def jake_total_balloons : ℕ := jake_initial_balloons + jake_bought_balloons

/-- The number of balloons Allan brought to the park -/
def allan_balloons : ℕ := jake_total_balloons + allan_jake_difference

theorem allan_brought_six_balloons : allan_balloons = 6 := by
  sorry

end NUMINAMATH_CALUDE_allan_brought_six_balloons_l1404_140458


namespace NUMINAMATH_CALUDE_product_evaluation_l1404_140420

theorem product_evaluation : (3 + 1) * (3^2 + 1^2) * (3^4 + 1^4) = 3280 := by
  sorry

end NUMINAMATH_CALUDE_product_evaluation_l1404_140420


namespace NUMINAMATH_CALUDE_f_derivative_at_zero_l1404_140427

noncomputable def f (x : ℝ) : ℝ := (x + 2) * Real.exp x

theorem f_derivative_at_zero : 
  deriv f 0 = 3 := by sorry

end NUMINAMATH_CALUDE_f_derivative_at_zero_l1404_140427


namespace NUMINAMATH_CALUDE_log_sum_power_twenty_l1404_140417

theorem log_sum_power_twenty (log_2 log_5 : ℝ) (h : log_2 + log_5 = 1) :
  (log_2 + log_5)^20 = 1 := by
  sorry

end NUMINAMATH_CALUDE_log_sum_power_twenty_l1404_140417


namespace NUMINAMATH_CALUDE_exactly_ten_maas_l1404_140482

-- Define the set S
variable (S : Type)

-- Define pib and maa as elements of S
variable (pib maa : S)

-- Define a relation to represent that a maa belongs to a pib
variable (belongs_to : S → S → Prop)

-- P1: Every pib is a collection of maas
axiom P1 : ∀ p : S, (∃ m : S, belongs_to m p) → p = pib

-- P2: Any three distinct pibs intersect at exactly one maa
axiom P2 : ∀ p1 p2 p3 : S, p1 = pib ∧ p2 = pib ∧ p3 = pib ∧ p1 ≠ p2 ∧ p2 ≠ p3 ∧ p1 ≠ p3 →
  ∃! m : S, belongs_to m p1 ∧ belongs_to m p2 ∧ belongs_to m p3

-- P3: Every maa belongs to exactly three pibs
axiom P3 : ∀ m : S, m = maa →
  ∃! p1 p2 p3 : S, p1 = pib ∧ p2 = pib ∧ p3 = pib ∧
    belongs_to m p1 ∧ belongs_to m p2 ∧ belongs_to m p3 ∧
    p1 ≠ p2 ∧ p2 ≠ p3 ∧ p1 ≠ p3

-- P4: There are exactly five pibs
axiom P4 : ∃! (p1 p2 p3 p4 p5 : S),
  p1 = pib ∧ p2 = pib ∧ p3 = pib ∧ p4 = pib ∧ p5 = pib ∧
  p1 ≠ p2 ∧ p1 ≠ p3 ∧ p1 ≠ p4 ∧ p1 ≠ p5 ∧
  p2 ≠ p3 ∧ p2 ≠ p4 ∧ p2 ≠ p5 ∧
  p3 ≠ p4 ∧ p3 ≠ p5 ∧
  p4 ≠ p5

-- Theorem: There are exactly ten maas
theorem exactly_ten_maas : ∃! (m1 m2 m3 m4 m5 m6 m7 m8 m9 m10 : S),
  m1 = maa ∧ m2 = maa ∧ m3 = maa ∧ m4 = maa ∧ m5 = maa ∧
  m6 = maa ∧ m7 = maa ∧ m8 = maa ∧ m9 = maa ∧ m10 = maa ∧
  m1 ≠ m2 ∧ m1 ≠ m3 ∧ m1 ≠ m4 ∧ m1 ≠ m5 ∧ m1 ≠ m6 ∧ m1 ≠ m7 ∧ m1 ≠ m8 ∧ m1 ≠ m9 ∧ m1 ≠ m10 ∧
  m2 ≠ m3 ∧ m2 ≠ m4 ∧ m2 ≠ m5 ∧ m2 ≠ m6 ∧ m2 ≠ m7 ∧ m2 ≠ m8 ∧ m2 ≠ m9 ∧ m2 ≠ m10 ∧
  m3 ≠ m4 ∧ m3 ≠ m5 ∧ m3 ≠ m6 ∧ m3 ≠ m7 ∧ m3 ≠ m8 ∧ m3 ≠ m9 ∧ m3 ≠ m10 ∧
  m4 ≠ m5 ∧ m4 ≠ m6 ∧ m4 ≠ m7 ∧ m4 ≠ m8 ∧ m4 ≠ m9 ∧ m4 ≠ m10 ∧
  m5 ≠ m6 ∧ m5 ≠ m7 ∧ m5 ≠ m8 ∧ m5 ≠ m9 ∧ m5 ≠ m10 ∧
  m6 ≠ m7 ∧ m6 ≠ m8 ∧ m6 ≠ m9 ∧ m6 ≠ m10 ∧
  m7 ≠ m8 ∧ m7 ≠ m9 ∧ m7 ≠ m10 ∧
  m8 ≠ m9 ∧ m8 ≠ m10 ∧
  m9 ≠ m10 := by
  sorry

end NUMINAMATH_CALUDE_exactly_ten_maas_l1404_140482


namespace NUMINAMATH_CALUDE_compound_propositions_truth_l1404_140402

-- Define propositions p and q
def p : Prop := ∀ x y : ℝ, x > y → -x < -y
def q : Prop := ∀ x y : ℝ, x < y → x^2 < y^2

-- Theorem statement
theorem compound_propositions_truth :
  (p ∧ q = False) ∧
  (p ∨ q = True) ∧
  (p ∧ (¬q) = True) ∧
  ((¬p) ∨ q = False) :=
by sorry

end NUMINAMATH_CALUDE_compound_propositions_truth_l1404_140402


namespace NUMINAMATH_CALUDE_best_standing_for_consistent_93rd_l1404_140414

/-- Represents a cycling competition -/
structure CyclingCompetition where
  stages : ℕ
  participants : ℕ
  daily_position : ℕ

/-- The best possible overall standing for a competitor -/
def best_possible_standing (comp : CyclingCompetition) : ℕ :=
  comp.participants - min (comp.stages * (comp.participants - comp.daily_position)) (comp.participants - 1)

/-- Theorem: In a 14-stage competition with 100 participants, 
    a competitor finishing 93rd each day can achieve 2nd place at best -/
theorem best_standing_for_consistent_93rd :
  let comp : CyclingCompetition := ⟨14, 100, 93⟩
  best_possible_standing comp = 2 := by
  sorry

#eval best_possible_standing ⟨14, 100, 93⟩

end NUMINAMATH_CALUDE_best_standing_for_consistent_93rd_l1404_140414


namespace NUMINAMATH_CALUDE_paperboy_delivery_ways_l1404_140428

/-- Represents the number of valid delivery sequences for n houses --/
def P : ℕ → ℕ
| 0 => 1  -- Base case for 0 houses
| 1 => 2  -- Base case for 1 house
| 2 => 4  -- Base case for 2 houses
| 3 => 8  -- Base case for 3 houses
| 4 => 15 -- Base case for 4 houses
| n + 5 => P (n + 4) + P (n + 3) + P (n + 2) + P (n + 1)

/-- The number of houses the paperboy delivers to --/
def num_houses : ℕ := 12

/-- Theorem stating the number of ways to deliver newspapers to 12 houses --/
theorem paperboy_delivery_ways : P num_houses = 2873 := by
  sorry

end NUMINAMATH_CALUDE_paperboy_delivery_ways_l1404_140428


namespace NUMINAMATH_CALUDE_correct_reasoning_definitions_l1404_140490

-- Define the types of reasoning
inductive ReasoningType
  | Inductive
  | Deductive
  | Analogical

-- Define the direction of reasoning
inductive ReasoningDirection
  | PartToWhole
  | GeneralToSpecific
  | SpecificToSpecific

-- Define a function that maps a reasoning type to its correct direction
def correct_reasoning_direction : ReasoningType → ReasoningDirection
  | ReasoningType.Inductive => ReasoningDirection.PartToWhole
  | ReasoningType.Deductive => ReasoningDirection.GeneralToSpecific
  | ReasoningType.Analogical => ReasoningDirection.SpecificToSpecific

-- Theorem stating that the correct reasoning directions are as defined
theorem correct_reasoning_definitions :
  (correct_reasoning_direction ReasoningType.Inductive = ReasoningDirection.PartToWhole) ∧
  (correct_reasoning_direction ReasoningType.Deductive = ReasoningDirection.GeneralToSpecific) ∧
  (correct_reasoning_direction ReasoningType.Analogical = ReasoningDirection.SpecificToSpecific) :=
by sorry

end NUMINAMATH_CALUDE_correct_reasoning_definitions_l1404_140490


namespace NUMINAMATH_CALUDE_only_two_and_five_plus_25_square_l1404_140495

theorem only_two_and_five_plus_25_square (N : ℕ+) : 
  (∀ p : ℕ, Nat.Prime p → p ∣ N → (p = 2 ∨ p = 5)) →
  (∃ k : ℕ, N + 25 = k^2) →
  (N = 200 ∨ N = 2000) := by
sorry

end NUMINAMATH_CALUDE_only_two_and_five_plus_25_square_l1404_140495


namespace NUMINAMATH_CALUDE_unique_zamena_assignment_l1404_140476

def digit := Fin 5

structure Assignment where
  Z : digit
  A : digit
  M : digit
  E : digit
  N : digit
  H : digit

def satisfies_inequalities (a : Assignment) : Prop :=
  (3 > a.A.val + 1) ∧ 
  (a.A.val > a.M.val) ∧ 
  (a.M.val < a.E.val) ∧ 
  (a.E.val < a.H.val) ∧ 
  (a.H.val < a.A.val)

def all_different (a : Assignment) : Prop :=
  a.Z ≠ a.A ∧ a.Z ≠ a.M ∧ a.Z ≠ a.E ∧ a.Z ≠ a.N ∧ a.Z ≠ a.H ∧
  a.A ≠ a.M ∧ a.A ≠ a.E ∧ a.A ≠ a.N ∧ a.A ≠ a.H ∧
  a.M ≠ a.E ∧ a.M ≠ a.N ∧ a.M ≠ a.H ∧
  a.E ≠ a.N ∧ a.E ≠ a.H ∧
  a.N ≠ a.H

def zamena_value (a : Assignment) : ℕ :=
  100000 * (a.Z.val + 1) + 10000 * (a.A.val + 1) + 1000 * (a.M.val + 1) +
  100 * (a.E.val + 1) + 10 * (a.N.val + 1) + (a.A.val + 1)

theorem unique_zamena_assignment :
  ∀ a : Assignment, 
    satisfies_inequalities a → all_different a → 
    zamena_value a = 541234 :=
sorry

end NUMINAMATH_CALUDE_unique_zamena_assignment_l1404_140476


namespace NUMINAMATH_CALUDE_Q_subset_P_l1404_140493

def P : Set ℝ := {x | x < 2}
def Q : Set ℝ := {y | y < 1}

theorem Q_subset_P : Q ⊆ P := by
  sorry

end NUMINAMATH_CALUDE_Q_subset_P_l1404_140493
