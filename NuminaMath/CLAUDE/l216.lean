import Mathlib

namespace NUMINAMATH_CALUDE_max_reach_is_nine_feet_l216_21659

/-- The maximum height Barry and Larry can reach when Barry stands on Larry's shoulders -/
def max_reach (barry_reach : ℝ) (larry_height : ℝ) (larry_shoulder_ratio : ℝ) : ℝ :=
  larry_height * larry_shoulder_ratio + barry_reach

/-- Theorem stating the maximum height Barry and Larry can reach -/
theorem max_reach_is_nine_feet :
  max_reach 5 5 0.8 = 9 := by
  sorry

#eval max_reach 5 5 0.8

end NUMINAMATH_CALUDE_max_reach_is_nine_feet_l216_21659


namespace NUMINAMATH_CALUDE_convex_polygon_mean_inequality_l216_21666

/-- For a convex n-gon, the arithmetic mean of side lengths is less than the arithmetic mean of diagonal lengths -/
theorem convex_polygon_mean_inequality {n : ℕ} (hn : n ≥ 3) 
  (P : ℝ) (D : ℝ) (hP : P > 0) (hD : D > 0) :
  P / n < (2 * D) / (n * (n - 3)) := by
  sorry

end NUMINAMATH_CALUDE_convex_polygon_mean_inequality_l216_21666


namespace NUMINAMATH_CALUDE_sum_interior_angles_30_vertices_l216_21663

/-- The sum of interior angles of faces in a convex polyhedron with given number of vertices -/
def sum_interior_angles (vertices : ℕ) : ℝ :=
  (vertices - 2) * 180

/-- Theorem: The sum of interior angles of faces in a convex polyhedron with 30 vertices is 5040° -/
theorem sum_interior_angles_30_vertices :
  sum_interior_angles 30 = 5040 := by
  sorry

end NUMINAMATH_CALUDE_sum_interior_angles_30_vertices_l216_21663


namespace NUMINAMATH_CALUDE_divisor_problem_l216_21658

theorem divisor_problem (x d : ℤ) (h1 : x % d = 7) (h2 : (x + 11) % 31 = 18) (h3 : d > 7) : d = 31 := by
  sorry

end NUMINAMATH_CALUDE_divisor_problem_l216_21658


namespace NUMINAMATH_CALUDE_xy_inequality_l216_21601

theorem xy_inequality (x y : ℝ) (h : x^2 + y^2 - x*y = 1) : 
  (x + y ≥ -2) ∧ (x^2 + y^2 ≤ 2) := by
  sorry

end NUMINAMATH_CALUDE_xy_inequality_l216_21601


namespace NUMINAMATH_CALUDE_fraction_difference_simplification_l216_21636

theorem fraction_difference_simplification : 
  ∃ q : ℕ+, (1011 : ℚ) / 1010 - 1010 / 1011 = (2021 : ℚ) / q ∧ Nat.gcd 2021 q.val = 1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_difference_simplification_l216_21636


namespace NUMINAMATH_CALUDE_sector_area_l216_21681

theorem sector_area (θ r : ℝ) (h1 : θ = 3) (h2 : r = 4) : 
  (1/2) * θ * r^2 = 24 :=
by sorry

end NUMINAMATH_CALUDE_sector_area_l216_21681


namespace NUMINAMATH_CALUDE_remainder_101_37_mod_100_l216_21694

theorem remainder_101_37_mod_100 : 101^37 % 100 = 1 := by
  sorry

end NUMINAMATH_CALUDE_remainder_101_37_mod_100_l216_21694


namespace NUMINAMATH_CALUDE_rope_division_l216_21664

theorem rope_division (initial_length : ℝ) (initial_cuts : ℕ) (final_cuts : ℕ) : 
  initial_length = 200 →
  initial_cuts = 4 →
  final_cuts = 2 →
  (initial_length / initial_cuts) / final_cuts = 25 := by
  sorry

end NUMINAMATH_CALUDE_rope_division_l216_21664


namespace NUMINAMATH_CALUDE_sugar_water_concentration_l216_21642

theorem sugar_water_concentration (a b m : ℝ) 
  (h1 : b > a) (h2 : a > 0) (h3 : m > 0) : 
  a / b < (a + m) / (b + m) := by
sorry

end NUMINAMATH_CALUDE_sugar_water_concentration_l216_21642


namespace NUMINAMATH_CALUDE_min_value_sum_of_powers_l216_21613

theorem min_value_sum_of_powers (a b x y : ℝ) (n : ℕ) 
  (ha : 0 < a) (hb : 0 < b) (hx : 0 < x) (hy : 0 < y) (hn : 0 < n) (hsum : x + y = 1) :
  a / x^n + b / y^n ≥ (a^(1/(n+1:ℝ)) + b^(1/(n+1:ℝ)))^(n+1) ∧
  (a / x^n + b / y^n = (a^(1/(n+1:ℝ)) + b^(1/(n+1:ℝ)))^(n+1) ↔ 
    x = a^(1/(n+1:ℝ)) / (a^(1/(n+1:ℝ)) + b^(1/(n+1:ℝ))) ∧ 
    y = b^(1/(n+1:ℝ)) / (a^(1/(n+1:ℝ)) + b^(1/(n+1:ℝ)))) :=
by sorry

end NUMINAMATH_CALUDE_min_value_sum_of_powers_l216_21613


namespace NUMINAMATH_CALUDE_smoking_lung_cancer_study_l216_21611

theorem smoking_lung_cancer_study (confidence : Real) 
  (h1 : confidence = 0.99) : 
  let error_probability := 1 - confidence
  error_probability ≤ 0.01 := by
sorry

end NUMINAMATH_CALUDE_smoking_lung_cancer_study_l216_21611


namespace NUMINAMATH_CALUDE_company_workers_l216_21683

theorem company_workers (total : ℕ) (men : ℕ) : 
  (total / 3 : ℚ) = (total / 3 : ℕ) →
  (2 * total / 10 : ℚ) * (total / 3 : ℚ) = ((2 * total / 10 : ℕ) * (total / 3 : ℕ) : ℚ) →
  (4 * total / 10 : ℚ) * (2 * total / 3 : ℚ) = ((4 * total / 10 : ℕ) * (2 * total / 3 : ℕ) : ℚ) →
  men = 112 →
  (4 * total / 10 : ℚ) * (2 * total / 3 : ℚ) + (total / 3 : ℚ) - (2 * total / 10 : ℚ) * (total / 3 : ℚ) = men →
  total - men = 98 :=
by sorry

end NUMINAMATH_CALUDE_company_workers_l216_21683


namespace NUMINAMATH_CALUDE_gcd_lcm_2970_1722_l216_21628

theorem gcd_lcm_2970_1722 : 
  (Nat.gcd 2970 1722 = 6) ∧ (Nat.lcm 2970 1722 = 856170) := by
  sorry

end NUMINAMATH_CALUDE_gcd_lcm_2970_1722_l216_21628


namespace NUMINAMATH_CALUDE_jills_water_volume_l216_21618

/-- Represents the number of jars of each size -/
def jars_per_size : ℕ := 48 / 3

/-- Represents the volume of a quart in gallons -/
def quart_volume : ℚ := 1 / 4

/-- Represents the volume of a half-gallon in gallons -/
def half_gallon_volume : ℚ := 1 / 2

/-- Represents the volume of a gallon in gallons -/
def gallon_volume : ℚ := 1

/-- Calculates the total volume of water in gallons -/
def total_water_volume : ℚ :=
  jars_per_size * quart_volume +
  jars_per_size * half_gallon_volume +
  jars_per_size * gallon_volume

theorem jills_water_volume :
  total_water_volume = 28 := by
  sorry

end NUMINAMATH_CALUDE_jills_water_volume_l216_21618


namespace NUMINAMATH_CALUDE_smallest_number_l216_21623

def digits : List Nat := [1, 4, 5]

def is_permutation (n : Nat) : Prop :=
  let digits_of_n := n.digits 10
  digits_of_n.length = digits.length ∧ digits_of_n.toFinset = digits.toFinset

theorem smallest_number :
  ∀ n : Nat, is_permutation n → 145 ≤ n := by
  sorry

end NUMINAMATH_CALUDE_smallest_number_l216_21623


namespace NUMINAMATH_CALUDE_work_completion_time_l216_21680

/-- The number of days it takes for worker A to complete the work alone -/
def days_A : ℝ := 6

/-- The number of days it takes for worker B to complete the work alone -/
def days_B : ℝ := 5

/-- The number of days it takes for workers A, B, and C to complete the work together -/
def days_ABC : ℝ := 2

/-- The number of days it takes for worker C to complete the work alone -/
def days_C : ℝ := 7.5

theorem work_completion_time :
  (1 / days_A) + (1 / days_B) + (1 / days_C) = (1 / days_ABC) := by
  sorry

end NUMINAMATH_CALUDE_work_completion_time_l216_21680


namespace NUMINAMATH_CALUDE_exchange_rate_l216_21685

def goose_to_duck : ℕ := 2
def pigeon_to_duck : ℕ := 5

theorem exchange_rate (geese : ℕ) : 
  geese * (goose_to_duck * pigeon_to_duck) = 
  geese * 10 := by sorry

end NUMINAMATH_CALUDE_exchange_rate_l216_21685


namespace NUMINAMATH_CALUDE_permutation_remainders_l216_21612

theorem permutation_remainders (a : Fin 11 → Fin 11) (h : Function.Bijective a) :
  ∃ i j : Fin 11, i ≠ j ∧ (i.val + 1) * (a i).val ≡ (j.val + 1) * (a j).val [ZMOD 11] := by
  sorry

end NUMINAMATH_CALUDE_permutation_remainders_l216_21612


namespace NUMINAMATH_CALUDE_negation_of_positive_square_plus_two_is_false_l216_21667

theorem negation_of_positive_square_plus_two_is_false : 
  ¬(∃ x : ℝ, x^2 + 2 ≤ 0) := by
sorry

end NUMINAMATH_CALUDE_negation_of_positive_square_plus_two_is_false_l216_21667


namespace NUMINAMATH_CALUDE_coefficient_x_squared_is_80_l216_21649

/-- The coefficient of x^2 in the expansion of (2x + 1/x^2)^5 -/
def coefficient_x_squared : ℕ :=
  (Nat.choose 5 1) * (2^4)

/-- Theorem stating that the coefficient of x^2 in the expansion of (2x + 1/x^2)^5 is 80 -/
theorem coefficient_x_squared_is_80 : coefficient_x_squared = 80 := by
  sorry

end NUMINAMATH_CALUDE_coefficient_x_squared_is_80_l216_21649


namespace NUMINAMATH_CALUDE_daniel_water_bottles_l216_21602

/-- The number of bottles Daniel filled for the rugby team -/
def rugby_bottles : ℕ := by sorry

theorem daniel_water_bottles :
  let total_bottles : ℕ := 254
  let football_players : ℕ := 11
  let football_bottles_per_player : ℕ := 6
  let soccer_bottles : ℕ := 53
  let lacrosse_extra_bottles : ℕ := 12
  let coach_bottles : ℕ := 2
  let num_teams : ℕ := 4

  let football_bottles := football_players * football_bottles_per_player
  let lacrosse_bottles := football_bottles + lacrosse_extra_bottles
  let total_coach_bottles := coach_bottles * num_teams

  rugby_bottles = total_bottles - (football_bottles + soccer_bottles + lacrosse_bottles + total_coach_bottles) :=
by sorry

end NUMINAMATH_CALUDE_daniel_water_bottles_l216_21602


namespace NUMINAMATH_CALUDE_race_time_difference_l216_21627

/-- Represents the time difference between two runners in a race -/
def timeDifference (malcolmSpeed joshuaSpeed raceDistance : ℝ) : ℝ :=
  joshuaSpeed * raceDistance - malcolmSpeed * raceDistance

/-- Proves that the time difference between Malcolm and Joshua in a 12-mile race is 24 minutes -/
theorem race_time_difference :
  timeDifference 5 7 12 = 24 := by
  sorry

end NUMINAMATH_CALUDE_race_time_difference_l216_21627


namespace NUMINAMATH_CALUDE_deduction_is_three_l216_21695

/-- Calculates the deduction per idle day for a worker --/
def calculate_deduction_per_idle_day (total_days : ℕ) (pay_rate : ℕ) (total_payment : ℕ) (idle_days : ℕ) : ℕ :=
  let working_days := total_days - idle_days
  let total_earnings := working_days * pay_rate
  (total_earnings - total_payment) / idle_days

/-- Theorem: Given the conditions, the deduction per idle day is 3 --/
theorem deduction_is_three :
  calculate_deduction_per_idle_day 60 20 280 40 = 3 := by
  sorry

#eval calculate_deduction_per_idle_day 60 20 280 40

end NUMINAMATH_CALUDE_deduction_is_three_l216_21695


namespace NUMINAMATH_CALUDE_fliers_distribution_theorem_l216_21693

def total_fliers : ℕ := 10000

def morning_fraction : ℚ := 1 / 5
def afternoon_fraction : ℚ := 1 / 4
def evening_fraction : ℚ := 1 / 3

def remaining_fliers : ℕ := 4000

theorem fliers_distribution_theorem :
  (total_fliers : ℚ) * (1 - morning_fraction) * (1 - afternoon_fraction) * (1 - evening_fraction) = remaining_fliers := by
  sorry

end NUMINAMATH_CALUDE_fliers_distribution_theorem_l216_21693


namespace NUMINAMATH_CALUDE_cone_cylinder_equal_volume_l216_21697

/-- Given a cylinder M with base radius 2 and height 2√3/3, and a cone N whose base diameter
    equals its slant height, if the volumes of M and N are equal, then the base radius of cone N is 2. -/
theorem cone_cylinder_equal_volume (r : ℝ) : 
  let cylinder_volume := π * 2^2 * (2 * Real.sqrt 3 / 3)
  let cone_volume := (1/3) * π * r^2 * (Real.sqrt 3 * r)
  (2 * r = Real.sqrt 3 * r) → (cylinder_volume = cone_volume) → r = 2 := by
  sorry

end NUMINAMATH_CALUDE_cone_cylinder_equal_volume_l216_21697


namespace NUMINAMATH_CALUDE_derivative_sin_2x_at_pi_3_l216_21670

theorem derivative_sin_2x_at_pi_3 :
  let f : ℝ → ℝ := fun x ↦ Real.sin (2 * x)
  (deriv f) (π / 3) = -1 := by
  sorry

end NUMINAMATH_CALUDE_derivative_sin_2x_at_pi_3_l216_21670


namespace NUMINAMATH_CALUDE_prob_two_eggplants_germination_rate_expected_value_X_l216_21656

-- Define the number of plots
def num_plots : ℕ := 4

-- Define the probability of planting eggplant in each plot
def prob_eggplant : ℚ := 1/3

-- Define the probability of planting cucumber in each plot
def prob_cucumber : ℚ := 2/3

-- Define the emergence rate of eggplant seeds
def emergence_rate_eggplant : ℚ := 95/100

-- Define the emergence rate of cucumber seeds
def emergence_rate_cucumber : ℚ := 98/100

-- Define the number of rows
def num_rows : ℕ := 2

-- Define the number of columns
def num_columns : ℕ := 2

-- Theorem for the probability of exactly 2 plots planting eggplants
theorem prob_two_eggplants : 
  (Nat.choose num_plots 2 : ℚ) * prob_eggplant^2 * prob_cucumber^2 = 8/27 := by sorry

-- Theorem for the germination rate of seeds for each plot
theorem germination_rate : 
  prob_eggplant * emergence_rate_eggplant + prob_cucumber * emergence_rate_cucumber = 97/100 := by sorry

-- Define the random variable X as the number of rows planting cucumbers
def X : Fin 3 → ℚ
| 0 => 1/25
| 1 => 16/25
| 2 => 8/25

-- Theorem for the expected value of X
theorem expected_value_X : 
  Finset.sum (Finset.range 3) (λ i => (i : ℚ) * X i) = 32/25 := by sorry

end NUMINAMATH_CALUDE_prob_two_eggplants_germination_rate_expected_value_X_l216_21656


namespace NUMINAMATH_CALUDE_negation_of_forall_positive_negation_of_proposition_l216_21605

theorem negation_of_forall_positive (P : ℝ → Prop) :
  (¬ ∀ x > 0, P x) ↔ (∃ x > 0, ¬ P x) :=
by sorry

theorem negation_of_proposition :
  (¬ ∀ x > 0, x^2 + x > 0) ↔ (∃ x > 0, x^2 + x ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_forall_positive_negation_of_proposition_l216_21605


namespace NUMINAMATH_CALUDE_largest_five_digit_congruent_to_19_mod_26_l216_21687

theorem largest_five_digit_congruent_to_19_mod_26 : 
  ∀ n : ℕ, 
    10000 ≤ n ∧ n ≤ 99999 ∧ n ≡ 19 [MOD 26] → 
    n ≤ 99989 :=
by sorry

end NUMINAMATH_CALUDE_largest_five_digit_congruent_to_19_mod_26_l216_21687


namespace NUMINAMATH_CALUDE_sqrt_fraction_equality_l216_21657

theorem sqrt_fraction_equality : 
  (Real.sqrt ((8:ℝ)^2 + 15^2)) / (Real.sqrt (36 + 64)) = 17 / 10 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_fraction_equality_l216_21657


namespace NUMINAMATH_CALUDE_quadratic_equation_properties_l216_21654

theorem quadratic_equation_properties :
  ∀ (k : ℝ), 
  -- The equation has two distinct real roots
  (∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ 2 * x₁^2 + k * x₁ - 1 = 0 ∧ 2 * x₂^2 + k * x₂ - 1 = 0) ∧
  -- When one root is -1, the other is 1/2 and k = 1
  (2 * (-1)^2 + k * (-1) - 1 = 0 → k = 1 ∧ 2 * (1/2)^2 + 1 * (1/2) - 1 = 0) :=
by sorry


end NUMINAMATH_CALUDE_quadratic_equation_properties_l216_21654


namespace NUMINAMATH_CALUDE_candy_distribution_l216_21691

/-- The number of children in the circle -/
def num_children : ℕ := 73

/-- The total number of candies distributed -/
def total_candies : ℕ := 2020

/-- The position of the n-th candy distribution -/
def candy_position (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The number of unique positions reached after distributing all candies -/
def unique_positions : ℕ := 37

theorem candy_distribution :
  num_children - unique_positions = 36 :=
sorry

end NUMINAMATH_CALUDE_candy_distribution_l216_21691


namespace NUMINAMATH_CALUDE_power_mod_seventeen_l216_21616

theorem power_mod_seventeen : 7^2048 % 17 = 1 := by
  sorry

end NUMINAMATH_CALUDE_power_mod_seventeen_l216_21616


namespace NUMINAMATH_CALUDE_tomato_plants_l216_21607

theorem tomato_plants (first_plant : ℕ) : 
  (∃ (second_plant third_plant fourth_plant : ℕ),
    second_plant = first_plant + 4 ∧
    third_plant = 3 * (first_plant + second_plant) ∧
    fourth_plant = 3 * (first_plant + second_plant) ∧
    first_plant + second_plant + third_plant + fourth_plant = 140) →
  first_plant = 8 := by
  sorry

end NUMINAMATH_CALUDE_tomato_plants_l216_21607


namespace NUMINAMATH_CALUDE_cookie_circle_properties_l216_21606

/-- Given a circle described by the equation x^2 + y^2 + 10 = 6x + 12y,
    this theorem proves its radius, circumference, and area. -/
theorem cookie_circle_properties :
  let equation := fun (x y : ℝ) => x^2 + y^2 + 10 = 6*x + 12*y
  ∃ (center : ℝ × ℝ) (r : ℝ),
    (∀ x y, equation x y ↔ (x - center.1)^2 + (y - center.2)^2 = r^2) ∧
    r = Real.sqrt 35 ∧
    2 * Real.pi * r = 2 * Real.pi * Real.sqrt 35 ∧
    Real.pi * r^2 = 35 * Real.pi :=
by sorry

end NUMINAMATH_CALUDE_cookie_circle_properties_l216_21606


namespace NUMINAMATH_CALUDE_vector_magnitude_proof_l216_21603

/-- Given two vectors HK and AE in a vector space, prove that if HK = 1/4 * AE and 
    the magnitude of 4 * HK is 4.8, then the magnitude of AE is 4.8. -/
theorem vector_magnitude_proof 
  (V : Type*) [NormedAddCommGroup V] [InnerProductSpace ℝ V] 
  (HK AE : V) 
  (h1 : HK = (1/4 : ℝ) • AE) 
  (h2 : ‖(4 : ℝ) • HK‖ = 4.8) : 
  ‖AE‖ = 4.8 := by
  sorry

#check vector_magnitude_proof

end NUMINAMATH_CALUDE_vector_magnitude_proof_l216_21603


namespace NUMINAMATH_CALUDE_rabbit_hop_time_l216_21690

/-- Proves that a rabbit hopping at 5 miles per hour takes 24 minutes to cover 2 miles -/
theorem rabbit_hop_time :
  let speed : ℝ := 5  -- miles per hour
  let distance : ℝ := 2  -- miles
  let time_hours : ℝ := distance / speed
  let minutes_per_hour : ℝ := 60
  let time_minutes : ℝ := time_hours * minutes_per_hour
  time_minutes = 24 := by sorry

end NUMINAMATH_CALUDE_rabbit_hop_time_l216_21690


namespace NUMINAMATH_CALUDE_exists_prime_divisor_l216_21640

/-- Definition of the sequence a_n -/
def a (c : ℕ) : ℕ → ℕ
  | 0 => c
  | n + 1 => (a c n)^3 - 4*c*(a c n)^2 + 5*c^2*(a c n) + c

/-- Main theorem statement -/
theorem exists_prime_divisor (c : ℕ) (hc : c ≥ 1) :
  ∀ n ≥ 2, ∃ p : ℕ, Nat.Prime p ∧ p ∣ a c n ∧ ∀ k < n, ¬(p ∣ a c k) :=
sorry

end NUMINAMATH_CALUDE_exists_prime_divisor_l216_21640


namespace NUMINAMATH_CALUDE_negation_of_proposition_l216_21604

theorem negation_of_proposition (p : Prop) :
  (¬(∀ x : ℝ, x > 0 → x^3 / (x - 2) > 0)) ↔ (∃ x : ℝ, x > 0 ∧ 0 ≤ x ∧ x ≤ 2) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_proposition_l216_21604


namespace NUMINAMATH_CALUDE_complex_fraction_evaluation_l216_21660

theorem complex_fraction_evaluation (a b : ℂ) (ha : a ≠ 0) (hb : b ≠ 0) 
  (h : a^2 - a*b + b^2 = 0) : 
  (a^7 + b^7) / (a - b)^7 = 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_evaluation_l216_21660


namespace NUMINAMATH_CALUDE_solution_satisfies_equations_l216_21698

theorem solution_satisfies_equations :
  ∃ (x y : ℚ), 
    (x = 5/2 ∧ y = 3) ∧
    (x + y + 1 = (6 - x) + (6 - y)) ∧
    (x - y + 2 = (x - 2) + (y - 2)) := by
  sorry

end NUMINAMATH_CALUDE_solution_satisfies_equations_l216_21698


namespace NUMINAMATH_CALUDE_multiply_both_sides_by_x_minus_3_l216_21635

variable (f g : ℝ → ℝ)
variable (x : ℝ)

theorem multiply_both_sides_by_x_minus_3 :
  f x = g x → (x - 3) * f x = (x - 3) * g x := by
  sorry

end NUMINAMATH_CALUDE_multiply_both_sides_by_x_minus_3_l216_21635


namespace NUMINAMATH_CALUDE_square_root_division_l216_21686

theorem square_root_division : Real.sqrt 18 / Real.sqrt 2 = 3 := by
  sorry

end NUMINAMATH_CALUDE_square_root_division_l216_21686


namespace NUMINAMATH_CALUDE_coefficient_m4n4_in_expansion_l216_21655

theorem coefficient_m4n4_in_expansion : ∀ m n : ℕ,
  (Nat.choose 8 4 : ℕ) = 70 := by sorry

end NUMINAMATH_CALUDE_coefficient_m4n4_in_expansion_l216_21655


namespace NUMINAMATH_CALUDE_singh_gain_l216_21673

/-- Represents the game outcome for three players -/
structure GameOutcome where
  ashtikar : ℚ
  singh : ℚ
  bhatia : ℚ

/-- The initial amount each player starts with -/
def initial_amount : ℚ := 70

/-- The theorem stating Singh's gain in the game -/
theorem singh_gain (outcome : GameOutcome) : 
  outcome.ashtikar + outcome.singh + outcome.bhatia = 3 * initial_amount ∧
  outcome.ashtikar = (1/2) * outcome.singh ∧
  outcome.bhatia = (1/4) * outcome.singh →
  outcome.singh - initial_amount = 50 := by
sorry

end NUMINAMATH_CALUDE_singh_gain_l216_21673


namespace NUMINAMATH_CALUDE_max_sum_of_entries_l216_21609

def numbers : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19]

def sum_of_entries (top : List ℕ) (left : List ℕ) : ℕ :=
  (top.sum * left.sum)

def is_valid_partition (l1 l2 : List ℕ) : Prop :=
  l1.length = 4 ∧ l2.length = 4 ∧ (l1 ++ l2).toFinset = numbers.toFinset

theorem max_sum_of_entries :
  ∃ (top left : List ℕ), 
    is_valid_partition top left ∧ 
    sum_of_entries top left = 1440 ∧
    ∀ (t l : List ℕ), is_valid_partition t l → sum_of_entries t l ≤ 1440 := by
  sorry

end NUMINAMATH_CALUDE_max_sum_of_entries_l216_21609


namespace NUMINAMATH_CALUDE_shorter_worm_length_l216_21646

theorem shorter_worm_length (worm1_length worm2_length : Real) :
  worm1_length = 0.8 →
  worm2_length = worm1_length + 0.7 →
  min worm1_length worm2_length = 0.8 := by
sorry

end NUMINAMATH_CALUDE_shorter_worm_length_l216_21646


namespace NUMINAMATH_CALUDE_equality_of_variables_l216_21662

theorem equality_of_variables (x₁ x₂ x₃ x₄ x₅ : ℝ) 
  (h_pos : x₁ > 0 ∧ x₂ > 0 ∧ x₃ > 0 ∧ x₄ > 0 ∧ x₅ > 0)
  (h₁ : (x₁^2 - x₃*x₅)*(x₂^2 - x₃*x₅) ≤ 0)
  (h₂ : (x₂^2 - x₄*x₁)*(x₃^2 - x₄*x₁) ≤ 0)
  (h₃ : (x₃^2 - x₅*x₂)*(x₄^2 - x₅*x₂) ≤ 0)
  (h₄ : (x₄^2 - x₁*x₃)*(x₅^2 - x₁*x₃) ≤ 0)
  (h₅ : (x₅^2 - x₂*x₄)*(x₁^2 - x₂*x₄) ≤ 0) :
  x₁ = x₂ ∧ x₂ = x₃ ∧ x₃ = x₄ ∧ x₄ = x₅ :=
by sorry

end NUMINAMATH_CALUDE_equality_of_variables_l216_21662


namespace NUMINAMATH_CALUDE_quadratic_expression_value_l216_21625

theorem quadratic_expression_value (x y : ℝ) 
  (h1 : 3 * x + y = 7) 
  (h2 : x + 3 * y = 8) : 
  10 * x^2 + 13 * x * y + 10 * y^2 = 113 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_expression_value_l216_21625


namespace NUMINAMATH_CALUDE_certain_number_proof_l216_21639

def is_divisible_by (a b : ℕ) : Prop := ∃ k : ℕ, a = b * k

theorem certain_number_proof : 
  ∃! x : ℕ, x > 0 ∧ 
    is_divisible_by (3153 + x) 9 ∧
    is_divisible_by (3153 + x) 70 ∧
    is_divisible_by (3153 + x) 25 ∧
    is_divisible_by (3153 + x) 21 ∧
    ∀ y : ℕ, y > 0 → 
      (is_divisible_by (3153 + y) 9 ∧
       is_divisible_by (3153 + y) 70 ∧
       is_divisible_by (3153 + y) 25 ∧
       is_divisible_by (3153 + y) 21) → 
      x ≤ y :=
by
  sorry

end NUMINAMATH_CALUDE_certain_number_proof_l216_21639


namespace NUMINAMATH_CALUDE_wheat_packets_in_gunny_bag_l216_21615

/-- The maximum number of wheat packets that can be accommodated in a gunny bag -/
def max_wheat_packets (bag_capacity : ℝ) (ton_to_kg : ℝ) (kg_to_g : ℝ) 
  (packet_weight_pounds : ℝ) (packet_weight_ounces : ℝ) 
  (pound_to_kg : ℝ) (ounce_to_g : ℝ) : ℕ :=
  sorry

/-- Theorem stating the maximum number of wheat packets in the gunny bag -/
theorem wheat_packets_in_gunny_bag : 
  max_wheat_packets 13 1000 1000 16 4 0.453592 28.3495 = 1763 := by
  sorry

end NUMINAMATH_CALUDE_wheat_packets_in_gunny_bag_l216_21615


namespace NUMINAMATH_CALUDE_gcd_process_max_rows_l216_21648

/-- Represents the GCD process described in the problem -/
def gcd_process (initial_sequence : List Nat) : Nat :=
  sorry

/-- The maximum number of rows in the GCD process -/
def max_rows : Nat := 501

/-- Theorem stating that the maximum number of rows in the GCD process is 501 -/
theorem gcd_process_max_rows :
  ∀ (seq : List Nat),
    (∀ n ∈ seq, 500 ≤ n ∧ n ≤ 1499) →
    seq.length = 1000 →
    gcd_process seq ≤ max_rows :=
  sorry

end NUMINAMATH_CALUDE_gcd_process_max_rows_l216_21648


namespace NUMINAMATH_CALUDE_square_lake_side_length_l216_21643

/-- Proves the length of each side of a square lake given Jake's swimming and rowing speeds and the time it takes to row around the lake. -/
theorem square_lake_side_length 
  (swimming_speed : ℝ) 
  (rowing_speed : ℝ) 
  (rowing_time : ℝ) 
  (h1 : swimming_speed = 3) 
  (h2 : rowing_speed = 2 * swimming_speed) 
  (h3 : rowing_time = 10) : 
  (rowing_speed * rowing_time) / 4 = 15 := by
  sorry

#check square_lake_side_length

end NUMINAMATH_CALUDE_square_lake_side_length_l216_21643


namespace NUMINAMATH_CALUDE_equal_side_sums_exist_l216_21696

def triangle_numbers : List ℕ := List.range 9 |>.map (· + 2016)

structure TriangleArrangement where
  positions : Fin 9 → ℕ
  is_valid : ∀ n, positions n ∈ triangle_numbers

def side_sum (arr : TriangleArrangement) (side : Fin 3) : ℕ :=
  match side with
  | 0 => arr.positions 0 + arr.positions 1 + arr.positions 2
  | 1 => arr.positions 2 + arr.positions 3 + arr.positions 4
  | 2 => arr.positions 4 + arr.positions 5 + arr.positions 0

theorem equal_side_sums_exist : 
  ∃ (arr : TriangleArrangement), ∀ (i j : Fin 3), side_sum arr i = side_sum arr j :=
sorry

end NUMINAMATH_CALUDE_equal_side_sums_exist_l216_21696


namespace NUMINAMATH_CALUDE_lcm_of_12_and_18_l216_21669

theorem lcm_of_12_and_18 : Nat.lcm 12 18 = 36 := by
  sorry

end NUMINAMATH_CALUDE_lcm_of_12_and_18_l216_21669


namespace NUMINAMATH_CALUDE_orange_apple_difference_l216_21626

/-- The number of apples Leif has -/
def num_apples : ℕ := 14

/-- The number of dozens of oranges Leif has -/
def dozens_oranges : ℕ := 2

/-- The number of oranges in a dozen -/
def oranges_per_dozen : ℕ := 12

/-- The total number of oranges Leif has -/
def num_oranges : ℕ := dozens_oranges * oranges_per_dozen

theorem orange_apple_difference :
  num_oranges - num_apples = 10 := by sorry

end NUMINAMATH_CALUDE_orange_apple_difference_l216_21626


namespace NUMINAMATH_CALUDE_greatest_x_value_l216_21652

theorem greatest_x_value (x : ℤ) (h : (2.134 : ℝ) * (10 : ℝ) ^ (x : ℝ) < 220000) :
  x ≤ 5 ∧ (2.134 : ℝ) * (10 : ℝ) ^ (5 : ℝ) < 220000 :=
sorry

end NUMINAMATH_CALUDE_greatest_x_value_l216_21652


namespace NUMINAMATH_CALUDE_y_coordinates_descending_l216_21677

/-- Given a line y = -2x + b and three points on this line, prove that the y-coordinates are in descending order as x increases. -/
theorem y_coordinates_descending 
  (b : ℝ) 
  (y₁ y₂ y₃ : ℝ) 
  (h1 : y₁ = 4 + b) 
  (h2 : y₂ = 2 + b) 
  (h3 : y₃ = -2 + b) : 
  y₃ < y₂ ∧ y₂ < y₁ := by
sorry

end NUMINAMATH_CALUDE_y_coordinates_descending_l216_21677


namespace NUMINAMATH_CALUDE_total_toys_given_l216_21630

theorem total_toys_given (toy_cars : ℕ) (dolls : ℕ) (board_games : ℕ)
  (h1 : toy_cars = 134)
  (h2 : dolls = 269)
  (h3 : board_games = 87) :
  toy_cars + dolls + board_games = 490 := by
  sorry

end NUMINAMATH_CALUDE_total_toys_given_l216_21630


namespace NUMINAMATH_CALUDE_plain_pancakes_count_l216_21650

theorem plain_pancakes_count (total : ℕ) (blueberry : ℕ) (banana : ℕ) 
  (h1 : total = 67) (h2 : blueberry = 20) (h3 : banana = 24) : 
  total - (blueberry + banana) = 23 := by
  sorry

end NUMINAMATH_CALUDE_plain_pancakes_count_l216_21650


namespace NUMINAMATH_CALUDE_hyperbola_iff_mn_positive_l216_21679

-- Define the condition for a curve to be a hyperbola
def is_hyperbola (m n : ℝ) : Prop :=
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ ∀ (x y : ℝ), m * x^2 - n * y^2 = 1 ↔ (x / a)^2 - (y / b)^2 = 1

-- State the theorem
theorem hyperbola_iff_mn_positive (m n : ℝ) :
  is_hyperbola m n ↔ m * n > 0 := by sorry

end NUMINAMATH_CALUDE_hyperbola_iff_mn_positive_l216_21679


namespace NUMINAMATH_CALUDE_pebble_collection_sum_l216_21622

/-- The sum of an arithmetic sequence with first term 2, common difference 3, and 15 terms -/
def arithmetic_sum : ℕ → ℕ
| n => n * (4 + 3 * (n - 1)) / 2

/-- Theorem stating that the sum of the first 15 terms of the arithmetic sequence is 345 -/
theorem pebble_collection_sum : arithmetic_sum 15 = 345 := by
  sorry

end NUMINAMATH_CALUDE_pebble_collection_sum_l216_21622


namespace NUMINAMATH_CALUDE_geometric_sequence_first_term_l216_21631

theorem geometric_sequence_first_term 
  (a : ℕ → ℝ) 
  (q : ℝ) 
  (h_positive : q > 0) 
  (h_geometric : ∀ n, a (n + 1) = a n * q) 
  (h_condition : a 3 * a 9 = 2 * (a 5)^2) 
  (h_second_term : a 2 = 1) : 
  a 1 = Real.sqrt 2 / 2 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_first_term_l216_21631


namespace NUMINAMATH_CALUDE_binomial_coefficient_third_term_2x_minus_y_power_8_l216_21661

/-- The binomial coefficient of the 3rd term in the expansion of (2x-y)^8 is 28 -/
theorem binomial_coefficient_third_term_2x_minus_y_power_8 :
  Nat.choose 8 2 = 28 := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_third_term_2x_minus_y_power_8_l216_21661


namespace NUMINAMATH_CALUDE_confidence_interval_for_population_mean_l216_21620

-- Define the sample data
def sample_data : List (Float × Nat) := [(-2, 2), (1, 2), (2, 2), (3, 2), (4, 2), (5, 1)]

-- Define the sample size
def n : Nat := 10

-- Define the confidence level
def confidence_level : Float := 0.95

-- Define the critical t-value for 9 degrees of freedom and 95% confidence
def t_critical : Float := 2.262

-- State the theorem
theorem confidence_interval_for_population_mean :
  let sample_mean := (sample_data.map (λ (x, freq) => x * freq.toFloat)).sum / n.toFloat
  let sample_variance := (sample_data.map (λ (x, freq) => freq.toFloat * (x - sample_mean)^2)).sum / (n.toFloat - 1)
  let sample_std_dev := sample_variance.sqrt
  let margin_of_error := t_critical * (sample_std_dev / (n.toFloat.sqrt))
  0.363 < sample_mean - margin_of_error ∧ sample_mean + margin_of_error < 3.837 := by
  sorry


end NUMINAMATH_CALUDE_confidence_interval_for_population_mean_l216_21620


namespace NUMINAMATH_CALUDE_stratified_sampling_theorem_l216_21682

theorem stratified_sampling_theorem (total_population : ℕ) (female_population : ℕ) (sample_size : ℕ) (female_sample : ℕ) :
  total_population = 2400 →
  female_population = 1000 →
  female_sample = 40 →
  (female_sample : ℚ) / sample_size = (female_population : ℚ) / total_population →
  sample_size = 96 :=
by
  sorry

end NUMINAMATH_CALUDE_stratified_sampling_theorem_l216_21682


namespace NUMINAMATH_CALUDE_ramp_cost_calculation_l216_21668

def ramp_installation_cost (permits_cost : ℝ) (contractor_labor_rate : ℝ) 
  (contractor_materials_rate : ℝ) (contractor_days : ℕ) (contractor_hours_per_day : ℝ) 
  (contractor_lunch_break : ℝ) (inspector_rate_discount : ℝ) (inspector_hours_per_day : ℝ) : ℝ :=
  let contractor_work_hours := (contractor_hours_per_day - contractor_lunch_break) * contractor_days
  let contractor_labor_cost := contractor_work_hours * contractor_labor_rate
  let materials_cost := contractor_work_hours * contractor_materials_rate
  let inspector_rate := contractor_labor_rate * (1 - inspector_rate_discount)
  let inspector_cost := inspector_rate * inspector_hours_per_day * contractor_days
  permits_cost + contractor_labor_cost + materials_cost + inspector_cost

theorem ramp_cost_calculation :
  ramp_installation_cost 250 150 50 3 5 0.5 0.8 2 = 3130 := by
  sorry

end NUMINAMATH_CALUDE_ramp_cost_calculation_l216_21668


namespace NUMINAMATH_CALUDE_arithmetic_sequence_property_l216_21678

/-- An arithmetic sequence with the given properties -/
structure ArithmeticSequence where
  length : ℕ
  mean : ℚ
  first : ℚ
  last : ℚ

/-- The new mean after removing the first and last numbers -/
def new_mean (seq : ArithmeticSequence) : ℚ :=
  ((seq.length : ℚ) * seq.mean - seq.first - seq.last) / ((seq.length : ℚ) - 2)

/-- Theorem stating the property of the specific arithmetic sequence -/
theorem arithmetic_sequence_property :
  let seq := ArithmeticSequence.mk 60 42 30 70
  new_mean seq = 41.7241 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_property_l216_21678


namespace NUMINAMATH_CALUDE_unique_k_solution_l216_21653

theorem unique_k_solution : 
  ∃! (k : ℕ), k ≥ 1 ∧ (∃ (n m : ℤ), 9 * n^6 = 2^k + 5 * m^2 + 2) ∧ k = 1 := by
  sorry

end NUMINAMATH_CALUDE_unique_k_solution_l216_21653


namespace NUMINAMATH_CALUDE_quadratic_roots_condition_l216_21651

theorem quadratic_roots_condition (a : ℝ) : 
  (∃ x y : ℝ, x > 0 ∧ y < 0 ∧ x^2 - 2*x + a + 1 = 0 ∧ y^2 - 2*y + a + 1 = 0) ↔ a < -1 :=
sorry

end NUMINAMATH_CALUDE_quadratic_roots_condition_l216_21651


namespace NUMINAMATH_CALUDE_min_cost_grass_seed_l216_21692

/-- Represents a bag of grass seed -/
structure SeedBag where
  weight : Nat
  price : Rat

/-- Calculates the total weight of a list of seed bags -/
def totalWeight (bags : List SeedBag) : Nat :=
  bags.foldl (fun acc bag => acc + bag.weight) 0

/-- Calculates the total price of a list of seed bags -/
def totalPrice (bags : List SeedBag) : Rat :=
  bags.foldl (fun acc bag => acc + bag.price) 0

/-- Theorem: The minimum cost to buy between 65 and 80 pounds of grass seed is $98.75 -/
theorem min_cost_grass_seed (bag5 bag10 bag25 : SeedBag)
    (h1 : bag5.weight = 5 ∧ bag5.price = 1385 / 100)
    (h2 : bag10.weight = 10 ∧ bag10.price = 2040 / 100)
    (h3 : bag25.weight = 25 ∧ bag25.price = 3225 / 100) :
    ∃ (bags : List SeedBag),
      (totalWeight bags ≥ 65 ∧ totalWeight bags ≤ 80) ∧
      totalPrice bags = 9875 / 100 ∧
      ∀ (other_bags : List SeedBag),
        (totalWeight other_bags ≥ 65 ∧ totalWeight other_bags ≤ 80) →
        totalPrice other_bags ≥ 9875 / 100 := by
  sorry

end NUMINAMATH_CALUDE_min_cost_grass_seed_l216_21692


namespace NUMINAMATH_CALUDE_largest_x_satisfying_equation_l216_21632

theorem largest_x_satisfying_equation : 
  ∃ (x : ℝ), ∀ (y : ℝ), 
    (|y^2 - 11*y + 24| + |2*y^2 + 6*y - 56| = |y^2 + 17*y - 80|) → 
    y ≤ x ∧ 
    |x^2 - 11*x + 24| + |2*x^2 + 6*x - 56| = |x^2 + 17*x - 80| ∧
    x = 8 := by
  sorry

end NUMINAMATH_CALUDE_largest_x_satisfying_equation_l216_21632


namespace NUMINAMATH_CALUDE_hyperbola_standard_form_l216_21641

/-- Given a hyperbola with equation x²/m - y²/(m+6) = 1 where m > 0,
    and its conjugate axis is twice the length of its transverse axis,
    prove that the standard form of the hyperbola's equation is x²/2 - y²/8 = 1 -/
theorem hyperbola_standard_form (m : ℝ) (h1 : m > 0) 
  (h2 : ∀ (x y : ℝ), x^2 / m - y^2 / (m + 6) = 1) 
  (h3 : 2 * (Real.sqrt m) = Real.sqrt (m + 6)) :
  ∀ (x y : ℝ), x^2 / 2 - y^2 / 8 = 1 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_standard_form_l216_21641


namespace NUMINAMATH_CALUDE_slope_of_line_l216_21633

theorem slope_of_line (x y : ℝ) :
  4 * x + 7 * y = 28 → (y - 4) = (-4/7) * (x - 0) :=
by sorry

end NUMINAMATH_CALUDE_slope_of_line_l216_21633


namespace NUMINAMATH_CALUDE_jimmy_passing_points_l216_21637

def points_per_exam : ℕ := 20
def number_of_exams : ℕ := 3
def points_lost_for_behavior : ℕ := 5
def additional_points_can_lose : ℕ := 5

def points_to_pass : ℕ := 50

theorem jimmy_passing_points :
  points_to_pass = 
    points_per_exam * number_of_exams - 
    points_lost_for_behavior - 
    additional_points_can_lose :=
by
  sorry

end NUMINAMATH_CALUDE_jimmy_passing_points_l216_21637


namespace NUMINAMATH_CALUDE_total_hotdogs_is_125_l216_21684

/-- The total number of hotdogs brought by two neighbors, where one neighbor brings 75 hotdogs
    and the other brings 25 fewer hotdogs than the first. -/
def total_hotdogs : ℕ :=
  let first_neighbor := 75
  let second_neighbor := first_neighbor - 25
  first_neighbor + second_neighbor

/-- Theorem stating that the total number of hotdogs brought by the neighbors is 125. -/
theorem total_hotdogs_is_125 : total_hotdogs = 125 := by
  sorry

end NUMINAMATH_CALUDE_total_hotdogs_is_125_l216_21684


namespace NUMINAMATH_CALUDE_second_number_proof_l216_21600

theorem second_number_proof : ∃! x : ℤ, 22030 = (555 + x) * (2 * (x - 555)) + 30 := by
  sorry

end NUMINAMATH_CALUDE_second_number_proof_l216_21600


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l216_21629

theorem sqrt_equation_solution : ∃ x : ℝ, x = 2209 / 64 ∧ Real.sqrt x + Real.sqrt (x + 3) = 12 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l216_21629


namespace NUMINAMATH_CALUDE_tips_fraction_of_income_l216_21645

/-- Represents the income structure of a waitress -/
structure WaitressIncome where
  salary : ℚ
  tips : ℚ

/-- Calculates the total income of a waitress -/
def totalIncome (w : WaitressIncome) : ℚ :=
  w.salary + w.tips

/-- Theorem stating that if a waitress's tips are 3/4 of her salary, 
    then 3/7 of her total income comes from tips -/
theorem tips_fraction_of_income 
  (w : WaitressIncome) 
  (h : w.tips = 3/4 * w.salary) : 
  w.tips / totalIncome w = 3/7 := by
  sorry

end NUMINAMATH_CALUDE_tips_fraction_of_income_l216_21645


namespace NUMINAMATH_CALUDE_equation_solutions_l216_21644

theorem equation_solutions :
  (∀ x : ℝ, x^2 - 49 = 0 ↔ x = 7 ∨ x = -7) ∧
  (∀ x : ℝ, 2*(x+1)^2 - 49 = 1 ↔ x = 4 ∨ x = -6) := by
sorry

end NUMINAMATH_CALUDE_equation_solutions_l216_21644


namespace NUMINAMATH_CALUDE_quadratic_root_value_l216_21638

theorem quadratic_root_value (m : ℝ) : 
  m^2 - m - 2 = 0 → 2*m^2 - 2*m + 2022 = 2026 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_value_l216_21638


namespace NUMINAMATH_CALUDE_base_seven_addition_problem_l216_21676

/-- Given a base 7 addition problem 3XY₇ + 52₇ = 42X₇, prove that X + Y = 6 in base 10 -/
theorem base_seven_addition_problem (X Y : Fin 7) :
  (3 * 7 * 7 + X * 7 + Y) + (5 * 7 + 2) = 4 * 7 * 7 + 2 * 7 + X →
  (X : ℕ) + (Y : ℕ) = 6 := by
  sorry

end NUMINAMATH_CALUDE_base_seven_addition_problem_l216_21676


namespace NUMINAMATH_CALUDE_solution_set_equivalence_l216_21614

theorem solution_set_equivalence : 
  {x : ℝ | (x + 3)^2 < 1} = {x : ℝ | -4 < x ∧ x < -2} := by
sorry

end NUMINAMATH_CALUDE_solution_set_equivalence_l216_21614


namespace NUMINAMATH_CALUDE_triangle_side_lengths_l216_21688

def is_valid_triple (a b c : ℕ) : Prop :=
  a ≤ b ∧ b ≤ c ∧ b * b = a * c ∧ (a = 100 ∨ c = 100)

def solution_set : Set (ℕ × ℕ × ℕ) :=
  {(49, 70, 100), (64, 80, 100), (81, 90, 100), (100, 100, 100),
   (100, 110, 121), (100, 120, 144), (100, 130, 169), (100, 140, 196),
   (100, 150, 225), (100, 160, 256)}

theorem triangle_side_lengths :
  ∀ a b c : ℕ, is_valid_triple a b c ↔ (a, b, c) ∈ solution_set :=
by sorry

end NUMINAMATH_CALUDE_triangle_side_lengths_l216_21688


namespace NUMINAMATH_CALUDE_lcm_48_180_l216_21699

theorem lcm_48_180 : Nat.lcm 48 180 = 720 := by sorry

end NUMINAMATH_CALUDE_lcm_48_180_l216_21699


namespace NUMINAMATH_CALUDE_bear_ratio_l216_21621

theorem bear_ratio (black_bears : ℕ) (brown_bears : ℕ) (white_bears : ℕ) :
  black_bears = 60 →
  brown_bears = black_bears + 40 →
  black_bears + brown_bears + white_bears = 190 →
  (black_bears : ℚ) / white_bears = 2 / 1 :=
by sorry

end NUMINAMATH_CALUDE_bear_ratio_l216_21621


namespace NUMINAMATH_CALUDE_smallest_self_descriptive_number_l216_21610

/-- Represents the value of a letter in the alphabet (A=1, B=2, ..., Z=26) -/
def letter_value (c : Char) : ℕ :=
  match c with
  | 'A' => 1 | 'B' => 2 | 'C' => 3 | 'D' => 4 | 'E' => 5 | 'F' => 6 | 'G' => 7
  | 'H' => 8 | 'I' => 9 | 'J' => 10 | 'K' => 11 | 'L' => 12 | 'M' => 13 | 'N' => 14
  | 'O' => 15 | 'P' => 16 | 'Q' => 17 | 'R' => 18 | 'S' => 19 | 'T' => 20
  | 'U' => 21 | 'V' => 22 | 'W' => 23 | 'X' => 24 | 'Y' => 25 | 'Z' => 26
  | _ => 0

/-- Calculates the sum of letter values in a string -/
def string_value (s : String) : ℕ :=
  s.toList.map letter_value |>.sum

/-- Converts a number to its written-out form in French -/
def number_to_french (n : ℕ) : String :=
  match n with
  | 222 => "DEUXCENTVINGTDEUX"
  | _ => ""  -- We only need to define 222 for this problem

theorem smallest_self_descriptive_number :
  ∀ n : ℕ, n < 222 → string_value (number_to_french n) ≠ n ∧
  string_value (number_to_french 222) = 222 := by
  sorry

#eval string_value (number_to_french 222)  -- Should output 222

end NUMINAMATH_CALUDE_smallest_self_descriptive_number_l216_21610


namespace NUMINAMATH_CALUDE_complement_A_U_eq_l216_21617

-- Define the universal set U
def U : Set ℝ := {x : ℝ | -4 < x ∧ x < 4}

-- Define set A
def A : Set ℝ := {x : ℝ | -3 ≤ x ∧ x < 2}

-- Define the complement of A with respect to U
def complement_A_U : Set ℝ := {x : ℝ | x ∈ U ∧ x ∉ A}

-- Theorem statement
theorem complement_A_U_eq :
  complement_A_U = {x : ℝ | (-4 < x ∧ x < -3) ∨ (2 ≤ x ∧ x < 4)} :=
by sorry

end NUMINAMATH_CALUDE_complement_A_U_eq_l216_21617


namespace NUMINAMATH_CALUDE_expression_value_l216_21689

theorem expression_value : 12 * (1 / 15) * 30 - 6 = 18 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l216_21689


namespace NUMINAMATH_CALUDE_max_value_sqrt_sum_l216_21672

theorem max_value_sqrt_sum (x y z : ℝ) (h1 : x ≥ 0) (h2 : y ≥ 0) (h3 : z ≥ 0) (h4 : x + y + z = 6) :
  ∃ (M : ℝ), M = 3 * Real.sqrt 8 ∧ 
  Real.sqrt (3*x + 2) + Real.sqrt (3*y + 2) + Real.sqrt (3*z + 2) ≤ M ∧
  ∃ (x' y' z' : ℝ), x' ≥ 0 ∧ y' ≥ 0 ∧ z' ≥ 0 ∧ x' + y' + z' = 6 ∧
    Real.sqrt (3*x' + 2) + Real.sqrt (3*y' + 2) + Real.sqrt (3*z' + 2) = M :=
by sorry

end NUMINAMATH_CALUDE_max_value_sqrt_sum_l216_21672


namespace NUMINAMATH_CALUDE_problem_statement_l216_21619

theorem problem_statement (C D : ℝ) :
  (∀ x : ℝ, x ≠ 3 → C / (x - 3) + D * (x + 2) = (-2 * x^2 + 8 * x + 28) / (x - 3)) →
  C + D = 20 := by
sorry

end NUMINAMATH_CALUDE_problem_statement_l216_21619


namespace NUMINAMATH_CALUDE_f_equation_l216_21674

-- Define the function f
def f : ℝ → ℝ := fun x => sorry

-- State the theorem
theorem f_equation : ∀ x : ℝ, f (x + 1) = x^2 - 5*x + 4 → f x = x^2 - 7*x + 10 := by
  sorry

end NUMINAMATH_CALUDE_f_equation_l216_21674


namespace NUMINAMATH_CALUDE_f_6n_l216_21675

def f : ℕ → ℤ
  | 0 => 0
  | n + 1 => 
    if n % 6 = 0 ∨ n % 6 = 1 then f n + 3
    else if n % 6 = 2 ∨ n % 6 = 5 then f n + 1
    else f n + 2

theorem f_6n (n : ℕ) : f (6 * n) = 12 * n := by
  sorry

end NUMINAMATH_CALUDE_f_6n_l216_21675


namespace NUMINAMATH_CALUDE_factor_and_divisor_relations_l216_21634

theorem factor_and_divisor_relations : 
  (∃ n : ℤ, 45 = 5 * n) ∧ 
  (209 % 19 = 0 ∧ 95 % 19 = 0) ∧ 
  (∃ m : ℤ, 180 = 9 * m) := by
sorry


end NUMINAMATH_CALUDE_factor_and_divisor_relations_l216_21634


namespace NUMINAMATH_CALUDE_arccos_neg_half_equals_two_pi_thirds_l216_21671

theorem arccos_neg_half_equals_two_pi_thirds :
  Real.arccos (-1/2) = 2*π/3 := by
  sorry

end NUMINAMATH_CALUDE_arccos_neg_half_equals_two_pi_thirds_l216_21671


namespace NUMINAMATH_CALUDE_camera_tax_calculation_l216_21665

/-- Calculate the tax amount given the base price and tax rate -/
def calculateTax (basePrice taxRate : ℝ) : ℝ :=
  basePrice * taxRate

/-- Prove that the tax amount for a $200 camera with 15% tax rate is $30 -/
theorem camera_tax_calculation :
  let basePrice : ℝ := 200
  let taxRate : ℝ := 0.15
  calculateTax basePrice taxRate = 30 := by
sorry

#eval calculateTax 200 0.15

end NUMINAMATH_CALUDE_camera_tax_calculation_l216_21665


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_l216_21608

theorem simplify_and_evaluate (x : ℝ) (h : x = -3) :
  (x^2 - 1) / (x + 2) / (1 - 1 / (x + 2)) = -4 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_l216_21608


namespace NUMINAMATH_CALUDE_system_inequalities_solution_l216_21624

theorem system_inequalities_solution : 
  {x : ℕ | 4 * (x + 1) ≤ 7 * x + 10 ∧ x - 5 < (x - 8) / 3} = {0, 1, 2, 3} := by
  sorry

end NUMINAMATH_CALUDE_system_inequalities_solution_l216_21624


namespace NUMINAMATH_CALUDE_complex_equation_roots_l216_21647

theorem complex_equation_roots : 
  let z₁ : ℂ := 4 - 0.5 * I
  let z₂ : ℂ := -2 + 0.5 * I
  (z₁^2 - 2*z₁ = 7 - 3*I) ∧ (z₂^2 - 2*z₂ = 7 - 3*I) := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_roots_l216_21647
