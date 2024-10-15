import Mathlib

namespace NUMINAMATH_CALUDE_quarterback_throws_l3573_357306

/-- Proves that given the specified conditions, the quarterback stepped back to throw 80 times. -/
theorem quarterback_throws (p_no_throw : ℝ) (p_sack_given_no_throw : ℝ) (num_sacks : ℕ) :
  p_no_throw = 0.3 →
  p_sack_given_no_throw = 0.5 →
  num_sacks = 12 →
  ∃ (total_throws : ℕ), total_throws = 80 ∧ 
    (p_no_throw * p_sack_given_no_throw * total_throws : ℝ) = num_sacks := by
  sorry

#check quarterback_throws

end NUMINAMATH_CALUDE_quarterback_throws_l3573_357306


namespace NUMINAMATH_CALUDE_ant_height_proof_l3573_357350

theorem ant_height_proof (rope_length : ℝ) (base_distance : ℝ) (shadow_rate : ℝ) (time : ℝ) 
  (h_rope : rope_length = 10)
  (h_base : base_distance = 6)
  (h_rate : shadow_rate = 0.3)
  (h_time : time = 5)
  (h_right_triangle : rope_length ^ 2 = base_distance ^ 2 + (rope_length ^ 2 - base_distance ^ 2))
  : ∃ (height : ℝ), 
    height = 2 ∧ 
    (shadow_rate * time) / base_distance = height / (rope_length ^ 2 - base_distance ^ 2).sqrt :=
by sorry

end NUMINAMATH_CALUDE_ant_height_proof_l3573_357350


namespace NUMINAMATH_CALUDE_polynomial_real_root_l3573_357381

theorem polynomial_real_root (b : ℝ) :
  (∃ x : ℝ, x^3 + b*x^2 - x + b = 0) ↔ b ≤ 0 := by sorry

end NUMINAMATH_CALUDE_polynomial_real_root_l3573_357381


namespace NUMINAMATH_CALUDE_trigonometric_equation_solution_l3573_357371

theorem trigonometric_equation_solution (x : Real) :
  (2 * Real.sin (17 * x) + Real.sqrt 3 * Real.cos (5 * x) + Real.sin (5 * x) = 0) ↔
  (∃ k : Int, x = π / 66 * (6 * k - 1) ∨ x = π / 18 * (3 * k + 2)) :=
by sorry

end NUMINAMATH_CALUDE_trigonometric_equation_solution_l3573_357371


namespace NUMINAMATH_CALUDE_first_term_of_geometric_sequence_l3573_357358

/-- A geometric sequence is a sequence where each term after the first is found by multiplying the previous term by a fixed, non-zero number called the common ratio. -/
def IsGeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = r * a n

theorem first_term_of_geometric_sequence (a : ℕ → ℝ) :
  IsGeometricSequence a →
  a 2 = 16 →
  a 4 = 128 →
  a 0 = 4 := by
  sorry

end NUMINAMATH_CALUDE_first_term_of_geometric_sequence_l3573_357358


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l3573_357320

theorem quadratic_inequality_solution_set :
  {x : ℝ | x^2 - 2*x - 5 > 2*x} = {x : ℝ | x > 5} ∪ {x : ℝ | x < -1} := by
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l3573_357320


namespace NUMINAMATH_CALUDE_min_red_chips_l3573_357338

theorem min_red_chips (w b r : ℕ) : 
  b ≥ w / 3 →
  b ≤ r / 4 →
  w + b ≥ 70 →
  r ≥ 72 ∧ ∀ r' : ℕ, (∃ w' b' : ℕ, b' ≥ w' / 3 ∧ b' ≤ r' / 4 ∧ w' + b' ≥ 70) → r' ≥ 72 :=
by sorry

end NUMINAMATH_CALUDE_min_red_chips_l3573_357338


namespace NUMINAMATH_CALUDE_stating_chameleon_change_theorem_l3573_357333

/-- Represents the change in the number of chameleons of a specific color. -/
structure ChameleonChange where
  green : ℤ
  yellow : ℤ

/-- Represents the weather conditions for a month. -/
structure WeatherConditions where
  sunny_days : ℕ
  cloudy_days : ℕ

/-- 
Theorem stating that the increase in green chameleons is equal to 
the increase in yellow chameleons plus the difference between sunny and cloudy days.
-/
theorem chameleon_change_theorem (weather : WeatherConditions) (change : ChameleonChange) :
  change.yellow = 5 →
  weather.sunny_days = 18 →
  weather.cloudy_days = 12 →
  change.green = change.yellow + (weather.sunny_days - weather.cloudy_days) :=
by sorry

end NUMINAMATH_CALUDE_stating_chameleon_change_theorem_l3573_357333


namespace NUMINAMATH_CALUDE_calories_burned_walking_james_walking_calories_l3573_357327

/-- Calculates the calories burned per hour while walking based on dancing data -/
theorem calories_burned_walking (dancing_calories_per_hour : ℝ) 
  (dancing_sessions_per_day : ℕ) (dancing_hours_per_session : ℝ) 
  (dancing_days_per_week : ℕ) (total_calories_per_week : ℝ) : ℝ :=
  let dancing_calories_ratio := 2
  let dancing_hours_per_week := dancing_sessions_per_day * dancing_hours_per_session * dancing_days_per_week
  let walking_calories_per_hour := total_calories_per_week / dancing_hours_per_week / dancing_calories_ratio
  by
    -- Proof goes here
    sorry

/-- Verifies that James burns 300 calories per hour while walking -/
theorem james_walking_calories : 
  calories_burned_walking 600 2 0.5 4 2400 = 300 := by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_calories_burned_walking_james_walking_calories_l3573_357327


namespace NUMINAMATH_CALUDE_divisibility_equivalence_l3573_357387

theorem divisibility_equivalence (x y : ℤ) :
  (2 * x + 3 * y) % 7 = 0 ↔ (5 * x + 4 * y) % 7 = 0 := by
  sorry

end NUMINAMATH_CALUDE_divisibility_equivalence_l3573_357387


namespace NUMINAMATH_CALUDE_democrats_to_participants_ratio_l3573_357385

/-- Proof of the ratio of democrats to total participants in a meeting --/
theorem democrats_to_participants_ratio :
  ∀ (total_participants : ℕ) 
    (female_democrats : ℕ) 
    (female_ratio : ℚ) 
    (male_ratio : ℚ),
  total_participants = 870 →
  female_democrats = 145 →
  female_ratio = 1/2 →
  male_ratio = 1/4 →
  (female_democrats * 2 + (total_participants - female_democrats * 2) * male_ratio) / total_participants = 1/3 :=
by
  sorry

#check democrats_to_participants_ratio

end NUMINAMATH_CALUDE_democrats_to_participants_ratio_l3573_357385


namespace NUMINAMATH_CALUDE_sin_15_plus_cos_15_l3573_357391

theorem sin_15_plus_cos_15 : Real.sin (15 * π / 180) + Real.cos (15 * π / 180) = Real.sqrt 6 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_15_plus_cos_15_l3573_357391


namespace NUMINAMATH_CALUDE_computer_contract_probability_l3573_357353

theorem computer_contract_probability (p_hardware : ℝ) (p_at_least_one : ℝ) (p_both : ℝ) :
  p_hardware = 3/4 →
  p_at_least_one = 5/6 →
  p_both = 0.31666666666666654 →
  1 - (p_at_least_one - p_hardware + p_both) = 0.6 :=
by sorry

end NUMINAMATH_CALUDE_computer_contract_probability_l3573_357353


namespace NUMINAMATH_CALUDE_factor_difference_of_squares_l3573_357332

theorem factor_difference_of_squares (y : ℝ) : 25 - 16 * y^2 = (5 - 4*y) * (5 + 4*y) := by
  sorry

end NUMINAMATH_CALUDE_factor_difference_of_squares_l3573_357332


namespace NUMINAMATH_CALUDE_fraction_inequality_l3573_357365

theorem fraction_inequality (x : ℝ) (h1 : x ≠ 2) (h2 : x ≠ -3) :
  (x^2 + 3*x + 2) / (x^2 + x - 6) ≠ (x + 2) / (x - 2) :=
by sorry

end NUMINAMATH_CALUDE_fraction_inequality_l3573_357365


namespace NUMINAMATH_CALUDE_total_weight_is_600_l3573_357395

/-- Proves that the total weight of Verna, Sherry, Jake, and Laura is 600 pounds given the specified conditions. -/
theorem total_weight_is_600 (haley_weight : ℝ) (verna_weight : ℝ) (sherry_weight : ℝ) (jake_weight : ℝ) (laura_weight : ℝ) : 
  haley_weight = 103 →
  verna_weight = haley_weight + 17 →
  verna_weight = sherry_weight / 2 →
  jake_weight = 3/5 * (haley_weight + verna_weight) →
  laura_weight = sherry_weight - jake_weight →
  verna_weight + sherry_weight + jake_weight + laura_weight = 600 := by
  sorry

#check total_weight_is_600

end NUMINAMATH_CALUDE_total_weight_is_600_l3573_357395


namespace NUMINAMATH_CALUDE_exists_x0_exp_greater_than_square_sum_of_roots_equals_five_l3573_357339

-- Proposition ③
theorem exists_x0_exp_greater_than_square :
  ∃ x₀ : ℝ, ∀ x > x₀, (2 : ℝ) ^ x > x ^ 2 := by sorry

-- Proposition ⑤
theorem sum_of_roots_equals_five :
  let f₁ := fun x : ℝ => x + Real.log 2 * Real.log x / Real.log 10 - 5
  let f₂ := fun x : ℝ => x + (10 : ℝ) ^ x - 5
  ∀ x₁ x₂ : ℝ, f₁ x₁ = 0 → f₂ x₂ = 0 → x₁ + x₂ = 5 := by sorry

end NUMINAMATH_CALUDE_exists_x0_exp_greater_than_square_sum_of_roots_equals_five_l3573_357339


namespace NUMINAMATH_CALUDE_questions_to_write_l3573_357341

theorem questions_to_write 
  (total_mc : ℕ) (total_ps : ℕ) (total_tf : ℕ)
  (frac_mc : ℚ) (frac_ps : ℚ) (frac_tf : ℚ)
  (h1 : total_mc = 50)
  (h2 : total_ps = 30)
  (h3 : total_tf = 40)
  (h4 : frac_mc = 5/8)
  (h5 : frac_ps = 7/12)
  (h6 : frac_tf = 2/5) :
  ↑total_mc - ⌊frac_mc * total_mc⌋ + 
  ↑total_ps - ⌊frac_ps * total_ps⌋ + 
  ↑total_tf - ⌊frac_tf * total_tf⌋ = 56 := by
sorry

end NUMINAMATH_CALUDE_questions_to_write_l3573_357341


namespace NUMINAMATH_CALUDE_smallest_number_l3573_357393

theorem smallest_number (a b c d : ℤ) (ha : a = 0) (hb : b = -1) (hc : c = 1) (hd : d = -5) :
  d ≤ a ∧ d ≤ b ∧ d ≤ c ∧ d ≤ d :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_l3573_357393


namespace NUMINAMATH_CALUDE_parking_theorem_l3573_357305

/-- The number of ways to arrange n distinct objects in k positions --/
def arrange (n : ℕ) (k : ℕ) : ℕ := sorry

/-- The number of ways to choose k items from n items --/
def choose (n : ℕ) (k : ℕ) : ℕ := sorry

/-- The number of ways to park cars in a row with contiguous empty spaces --/
def parkingArrangements (total_spaces : ℕ) (cars : ℕ) : ℕ :=
  arrange cars cars * choose (cars + 1) 1

theorem parking_theorem :
  parkingArrangements 12 8 = arrange 8 8 * choose 9 1 := by sorry

end NUMINAMATH_CALUDE_parking_theorem_l3573_357305


namespace NUMINAMATH_CALUDE_max_abs_z_l3573_357330

theorem max_abs_z (z : ℂ) (h : Complex.abs (z + 3 + 4*I) = 2) :
  ∃ (w : ℂ), Complex.abs w = 2 ∧ Complex.abs (w + 3 + 4*I) = 2 ∧
  ∀ (u : ℂ), Complex.abs (u + 3 + 4*I) = 2 → Complex.abs u ≤ Complex.abs w :=
sorry

end NUMINAMATH_CALUDE_max_abs_z_l3573_357330


namespace NUMINAMATH_CALUDE_pradeep_exam_marks_l3573_357376

/-- The maximum marks in Pradeep's exam -/
def maximum_marks : ℕ := 928

/-- The percentage required to pass the exam -/
def pass_percentage : ℚ := 55 / 100

/-- The marks Pradeep obtained -/
def pradeep_marks : ℕ := 400

/-- The number of marks Pradeep fell short by -/
def shortfall : ℕ := 110

theorem pradeep_exam_marks :
  (pass_percentage * maximum_marks : ℚ) = pradeep_marks + shortfall ∧
  maximum_marks * pass_percentage = (pradeep_marks + shortfall : ℚ) ∧
  ∀ m : ℕ, m > maximum_marks → 
    (pass_percentage * m : ℚ) > (pradeep_marks + shortfall : ℚ) :=
by sorry

end NUMINAMATH_CALUDE_pradeep_exam_marks_l3573_357376


namespace NUMINAMATH_CALUDE_modulus_of_complex_product_l3573_357335

theorem modulus_of_complex_product : ∃ (z : ℂ), z = (1 + Complex.I) * (3 - 4 * Complex.I) ∧ Complex.abs z = 5 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_modulus_of_complex_product_l3573_357335


namespace NUMINAMATH_CALUDE_cans_restocked_day2_is_1500_l3573_357349

/-- Represents the food bank scenario --/
structure FoodBank where
  initial_stock : ℕ
  day1_people : ℕ
  day1_cans_per_person : ℕ
  day1_restock : ℕ
  day2_people : ℕ
  day2_cans_per_person : ℕ
  total_cans_given : ℕ

/-- Calculates the number of cans restocked after the second day --/
def cans_restocked_day2 (fb : FoodBank) : ℕ :=
  fb.total_cans_given - (fb.initial_stock - fb.day1_people * fb.day1_cans_per_person + fb.day1_restock - fb.day2_people * fb.day2_cans_per_person)

/-- Theorem stating that for the given scenario, the number of cans restocked after the second day is 1500 --/
theorem cans_restocked_day2_is_1500 (fb : FoodBank) 
  (h1 : fb.initial_stock = 2000)
  (h2 : fb.day1_people = 500)
  (h3 : fb.day1_cans_per_person = 1)
  (h4 : fb.day1_restock = 1500)
  (h5 : fb.day2_people = 1000)
  (h6 : fb.day2_cans_per_person = 2)
  (h7 : fb.total_cans_given = 2500) :
  cans_restocked_day2 fb = 1500 := by
  sorry

end NUMINAMATH_CALUDE_cans_restocked_day2_is_1500_l3573_357349


namespace NUMINAMATH_CALUDE_boat_upstream_downstream_distance_l3573_357379

/-- Proves that a boat with a given speed in still water, traveling a certain distance upstream in one hour, will travel a specific distance downstream in one hour. -/
theorem boat_upstream_downstream_distance 
  (v : ℝ) -- Speed of the boat in still water (km/h)
  (d_upstream : ℝ) -- Distance traveled upstream in one hour (km)
  (h1 : v = 8) -- The boat's speed in still water is 8 km/h
  (h2 : d_upstream = 5) -- The boat travels 5 km upstream in one hour
  : ∃ d_downstream : ℝ, d_downstream = 11 ∧ d_downstream = v + (v - d_upstream) := by
  sorry

end NUMINAMATH_CALUDE_boat_upstream_downstream_distance_l3573_357379


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l3573_357368

theorem imaginary_part_of_complex_fraction (i : ℂ) : 
  i * i = -1 → Complex.im ((5 + i) / (2 - i)) = 7 / 5 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l3573_357368


namespace NUMINAMATH_CALUDE_conic_section_eccentricity_l3573_357394

theorem conic_section_eccentricity (m : ℝ) : 
  (m^2 = 5 * (16/5)) → 
  (∃ (e : ℝ), (e = Real.sqrt 3 / 2 ∨ e = Real.sqrt 5) ∧ 
   ∃ (a b c : ℝ), (a > 0 ∧ b > 0 ∧ c > 0) ∧
   ((x : ℝ) → (y : ℝ) → x^2 / m + y^2 = 1 → 
    (e = c / a ∧ ((m > 0 → a^2 - b^2 = c^2) ∧ (m < 0 → b^2 - a^2 = c^2))))) :=
by sorry

end NUMINAMATH_CALUDE_conic_section_eccentricity_l3573_357394


namespace NUMINAMATH_CALUDE_binomial_150_1_l3573_357348

theorem binomial_150_1 : Nat.choose 150 1 = 150 := by
  sorry

end NUMINAMATH_CALUDE_binomial_150_1_l3573_357348


namespace NUMINAMATH_CALUDE_infinitely_many_H_points_l3573_357300

/-- The curve C defined by x/4 + y^2 = 1 -/
def C : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1 / 4 + p.2^2 = 1}

/-- The line l defined by x = 4 -/
def l : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1 = 4}

/-- A point P is an H point if there exists a line through P intersecting C at A 
    and l at B, with either |PA| = |PB| or |PA| = |AB| -/
def is_H_point (P : ℝ × ℝ) : Prop :=
  P ∈ C ∧ ∃ (A B : ℝ × ℝ), A ∈ C ∧ B ∈ l ∧ A ≠ P ∧
    (∃ (k m : ℝ), ∀ x y, y = k * x + m → 
      ((x, y) = P ∨ (x, y) = A ∨ (x, y) = B)) ∧
    (dist P A = dist P B ∨ dist P A = dist A B)

/-- There are infinitely many H points on C, but not all points on C are H points -/
theorem infinitely_many_H_points : 
  (∃ (S : Set (ℝ × ℝ)), S ⊆ C ∧ Infinite S ∧ ∀ p ∈ S, is_H_point p) ∧
  (∃ p ∈ C, ¬is_H_point p) := by sorry


end NUMINAMATH_CALUDE_infinitely_many_H_points_l3573_357300


namespace NUMINAMATH_CALUDE_harry_lost_sea_creatures_l3573_357302

theorem harry_lost_sea_creatures (sea_stars seashells snails items_left : ℕ) 
  (h1 : sea_stars = 34)
  (h2 : seashells = 21)
  (h3 : snails = 29)
  (h4 : items_left = 59) :
  sea_stars + seashells + snails - items_left = 25 := by
  sorry

end NUMINAMATH_CALUDE_harry_lost_sea_creatures_l3573_357302


namespace NUMINAMATH_CALUDE_ion_relationship_l3573_357355

/-- Represents an ion with atomic number and charge -/
structure Ion where
  atomic_number : ℕ
  charge : ℤ

/-- Two ions have the same electron shell structure -/
def same_electron_shell (x y : Ion) : Prop :=
  x.atomic_number + x.charge = y.atomic_number - y.charge

theorem ion_relationship {a b n m : ℕ} (X Y : Ion)
  (hX : X.atomic_number = a ∧ X.charge = -n)
  (hY : Y.atomic_number = b ∧ Y.charge = m)
  (h_same_shell : same_electron_shell X Y) :
  a + m = b - n := by
  sorry


end NUMINAMATH_CALUDE_ion_relationship_l3573_357355


namespace NUMINAMATH_CALUDE_polynomial_inequality_l3573_357397

/-- A polynomial with positive real coefficients -/
structure PositivePolynomial where
  coeffs : List ℝ
  all_positive : ∀ c ∈ coeffs, c > 0

/-- Evaluate a polynomial at a given point -/
def evalPoly (p : PositivePolynomial) (x : ℝ) : ℝ :=
  p.coeffs.enum.foldl (λ acc (i, a) => acc + a * x ^ i) 0

/-- The main theorem -/
theorem polynomial_inequality (p : PositivePolynomial) :
  (evalPoly p 1 ≥ 1 / evalPoly p 1) →
  (∀ x : ℝ, x > 0 → evalPoly p (1/x) ≥ 1 / evalPoly p x) :=
sorry

end NUMINAMATH_CALUDE_polynomial_inequality_l3573_357397


namespace NUMINAMATH_CALUDE_range_of_m_l3573_357388

/-- Given the equation (m+3)/(x-1) = 1 where x is a positive number, 
    prove that the range of m is m > -4 and m ≠ -3 -/
theorem range_of_m (m : ℝ) (x : ℝ) (h1 : (m + 3) / (x - 1) = 1) (h2 : x > 0) :
  m > -4 ∧ m ≠ -3 := by
  sorry

end NUMINAMATH_CALUDE_range_of_m_l3573_357388


namespace NUMINAMATH_CALUDE_blue_shirt_percentage_l3573_357315

/-- Proves that the percentage of students wearing blue shirts is 45% -/
theorem blue_shirt_percentage
  (total_students : ℕ)
  (red_shirt_percentage : ℚ)
  (green_shirt_percentage : ℚ)
  (other_colors_count : ℕ)
  (h1 : total_students = 600)
  (h2 : red_shirt_percentage = 23 / 100)
  (h3 : green_shirt_percentage = 15 / 100)
  (h4 : other_colors_count = 102)
  : (1 : ℚ) - (red_shirt_percentage + green_shirt_percentage + (other_colors_count : ℚ) / (total_students : ℚ)) = 45 / 100 := by
  sorry

#check blue_shirt_percentage

end NUMINAMATH_CALUDE_blue_shirt_percentage_l3573_357315


namespace NUMINAMATH_CALUDE_find_number_l3573_357396

theorem find_number : ∃ x : ℝ, 4.75 + 0.303 + x = 5.485 ∧ x = 0.432 := by
  sorry

end NUMINAMATH_CALUDE_find_number_l3573_357396


namespace NUMINAMATH_CALUDE_greatest_three_digit_multiple_of_19_l3573_357346

theorem greatest_three_digit_multiple_of_19 : 
  ∀ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ 19 ∣ n → n ≤ 988 :=
by
  sorry

end NUMINAMATH_CALUDE_greatest_three_digit_multiple_of_19_l3573_357346


namespace NUMINAMATH_CALUDE_infinite_primes_no_fantastic_multiple_infinite_primes_with_fantastic_multiple_l3573_357301

def IsFantastic (n : ℕ) : Prop :=
  ∃ (a b : ℚ), a > 0 ∧ b > 0 ∧ n = ⌊a + 1/a + b + 1/b⌋

theorem infinite_primes_no_fantastic_multiple :
  ∃ (S : Set ℕ), Set.Infinite S ∧ (∀ p ∈ S, Prime p) ∧
    (∀ (p : ℕ) (k : ℕ), p ∈ S → k > 0 → ¬IsFantastic (k * p)) :=
sorry

theorem infinite_primes_with_fantastic_multiple :
  ∃ (S : Set ℕ), Set.Infinite S ∧ (∀ p ∈ S, Prime p) ∧
    (∀ p ∈ S, ∃ (k : ℕ), k > 0 ∧ IsFantastic (k * p)) :=
sorry

end NUMINAMATH_CALUDE_infinite_primes_no_fantastic_multiple_infinite_primes_with_fantastic_multiple_l3573_357301


namespace NUMINAMATH_CALUDE_binomial_prob_X_eq_one_l3573_357354

/-- A random variable X following a binomial distribution B(n, p) with given expectation and variance -/
structure BinomialRV where
  n : ℕ
  p : ℝ
  mean_eq : n * p = 5 / 2
  var_eq : n * p * (1 - p) = 5 / 4

/-- The probability of X = 1 for the given binomial random variable -/
def prob_X_eq_one (X : BinomialRV) : ℝ :=
  X.n.choose 1 * X.p^1 * (1 - X.p)^(X.n - 1)

/-- Theorem stating that P(X=1) = 5/32 for the given binomial random variable -/
theorem binomial_prob_X_eq_one (X : BinomialRV) : prob_X_eq_one X = 5 / 32 := by
  sorry

end NUMINAMATH_CALUDE_binomial_prob_X_eq_one_l3573_357354


namespace NUMINAMATH_CALUDE_semicircle_is_arc_l3573_357342

-- Define a circle
def Circle := Set (ℝ × ℝ)

-- Define an arc
def Arc (c : Circle) := Set (ℝ × ℝ)

-- Define a semicircle
def Semicircle (c : Circle) := Arc c

-- Theorem: A semicircle is an arc
theorem semicircle_is_arc (c : Circle) : Semicircle c → Arc c := by
  sorry

end NUMINAMATH_CALUDE_semicircle_is_arc_l3573_357342


namespace NUMINAMATH_CALUDE_ten_mile_taxi_cost_l3573_357359

def taxi_cost (initial_cost : ℝ) (cost_per_mile : ℝ) (distance : ℝ) : ℝ :=
  initial_cost + cost_per_mile * distance

theorem ten_mile_taxi_cost :
  taxi_cost 2 0.3 10 = 5 := by
  sorry

end NUMINAMATH_CALUDE_ten_mile_taxi_cost_l3573_357359


namespace NUMINAMATH_CALUDE_lee_quiz_probability_l3573_357308

theorem lee_quiz_probability (p : ℚ) (h : p = 5/8) :
  1 - p = 3/8 := by
  sorry

end NUMINAMATH_CALUDE_lee_quiz_probability_l3573_357308


namespace NUMINAMATH_CALUDE_tan_inequality_solution_set_l3573_357399

theorem tan_inequality_solution_set : 
  let S := {x : ℝ | ∃ k : ℤ, k * π - π / 3 < x ∧ x < k * π + Real.arctan 2}
  ∀ x : ℝ, x ∈ S ↔ -Real.sqrt 3 < Real.tan x ∧ Real.tan x < 2 :=
by sorry

end NUMINAMATH_CALUDE_tan_inequality_solution_set_l3573_357399


namespace NUMINAMATH_CALUDE_sqrt_of_neg_seven_squared_l3573_357372

theorem sqrt_of_neg_seven_squared : Real.sqrt ((-7)^2) = 7 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_of_neg_seven_squared_l3573_357372


namespace NUMINAMATH_CALUDE_even_function_four_zeroes_range_l3573_357351

/-- An even function is a function that is symmetric about the y-axis -/
def EvenFunction (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = f x

/-- A function has four distinct zeroes if there exist four different real numbers that make the function equal to zero -/
def HasFourDistinctZeroes (f : ℝ → ℝ) : Prop :=
  ∃ (a b c d : ℝ), a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
    f a = 0 ∧ f b = 0 ∧ f c = 0 ∧ f d = 0

theorem even_function_four_zeroes_range (f : ℝ → ℝ) (h_even : EvenFunction f) :
  (∃ m : ℝ, HasFourDistinctZeroes (fun x => f x - m)) →
  (∀ m : ℝ, m ≠ 0 → ∃ x : ℝ, f x = m) ∧ (¬∃ x : ℝ, f x = 0) :=
sorry

end NUMINAMATH_CALUDE_even_function_four_zeroes_range_l3573_357351


namespace NUMINAMATH_CALUDE_suzanna_distance_l3573_357307

/-- Represents the distance in miles Suzanna cycles in a given time -/
def distance_cycled (time_minutes : ℕ) : ℚ :=
  (time_minutes / 10 : ℚ) * 2

/-- Proves that Suzanna cycles 8 miles in 40 minutes given her steady speed -/
theorem suzanna_distance : distance_cycled 40 = 8 := by
  sorry

end NUMINAMATH_CALUDE_suzanna_distance_l3573_357307


namespace NUMINAMATH_CALUDE_difference_of_squares_l3573_357370

theorem difference_of_squares (a b : ℝ) : (a + 2*b) * (a - 2*b) = a^2 - 4*b^2 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_l3573_357370


namespace NUMINAMATH_CALUDE_larger_number_problem_l3573_357344

theorem larger_number_problem (x y : ℝ) (h_product : x * y = 30) (h_sum : x + y = 13) :
  max x y = 10 := by
  sorry

end NUMINAMATH_CALUDE_larger_number_problem_l3573_357344


namespace NUMINAMATH_CALUDE_second_shift_participation_theorem_l3573_357325

/-- The percentage of second shift employees participating in the pension program -/
def second_shift_participation_rate : ℝ := 40

theorem second_shift_participation_theorem :
  let total_employees : ℕ := 60 + 50 + 40
  let first_shift : ℕ := 60
  let second_shift : ℕ := 50
  let third_shift : ℕ := 40
  let first_shift_rate : ℝ := 20
  let third_shift_rate : ℝ := 10
  let total_participation_rate : ℝ := 24
  let first_shift_participants : ℝ := first_shift_rate / 100 * first_shift
  let third_shift_participants : ℝ := third_shift_rate / 100 * third_shift
  let total_participants : ℝ := total_participation_rate / 100 * total_employees
  let second_shift_participants : ℝ := total_participants - first_shift_participants - third_shift_participants
  second_shift_participation_rate = second_shift_participants / second_shift * 100 :=
by
  sorry

end NUMINAMATH_CALUDE_second_shift_participation_theorem_l3573_357325


namespace NUMINAMATH_CALUDE_pony_price_is_18_l3573_357337

/-- The regular price of Fox jeans in dollars -/
def fox_price : ℝ := 15

/-- The discount rate for Pony jeans as a decimal -/
def pony_discount : ℝ := 0.14

/-- The sum of discount rates for Fox and Pony jeans as a decimal -/
def total_discount : ℝ := 0.22

/-- The total savings from purchasing 5 pairs of jeans -/
def total_savings : ℝ := 8.64

/-- The number of Fox jeans purchased -/
def fox_count : ℕ := 3

/-- The number of Pony jeans purchased -/
def pony_count : ℕ := 2

/-- The regular price of Pony jeans -/
def pony_price : ℝ := 18

theorem pony_price_is_18 :
  fox_count * fox_price * (total_discount - pony_discount) +
  pony_count * pony_price * pony_discount = total_savings :=
by sorry

end NUMINAMATH_CALUDE_pony_price_is_18_l3573_357337


namespace NUMINAMATH_CALUDE_problem_solution_l3573_357384

theorem problem_solution (p q : ℕ) (hp : p < 30) (hq : q < 30) (h_eq : p + q + p * q = 119) :
  p + q = 20 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3573_357384


namespace NUMINAMATH_CALUDE_area_under_curve_l3573_357318

open Real MeasureTheory

-- Define the function
def f (x : ℝ) : ℝ := 3 * x^2

-- State the theorem
theorem area_under_curve : 
  ∫ x in (1)..(2), f x = 7 := by
  sorry

end NUMINAMATH_CALUDE_area_under_curve_l3573_357318


namespace NUMINAMATH_CALUDE_child_support_owed_amount_l3573_357364

/-- Calculates the amount owed in child support given the specified conditions --/
def child_support_owed (
  support_rate : ℝ)
  (initial_salary : ℝ)
  (initial_years : ℕ)
  (raise_percentage : ℝ)
  (raise_years : ℕ)
  (amount_paid : ℝ) : ℝ :=
  let total_initial_income := initial_salary * initial_years
  let raised_salary := initial_salary * (1 + raise_percentage)
  let total_raised_income := raised_salary * raise_years
  let total_income := total_initial_income + total_raised_income
  let total_support_due := total_income * support_rate
  total_support_due - amount_paid

/-- Theorem stating that the amount owed in child support is $69,000 --/
theorem child_support_owed_amount : 
  child_support_owed 0.3 30000 3 0.2 4 1200 = 69000 := by
  sorry

end NUMINAMATH_CALUDE_child_support_owed_amount_l3573_357364


namespace NUMINAMATH_CALUDE_divisibility_by_37_l3573_357303

theorem divisibility_by_37 (a b c : ℕ) :
  (37 ∣ (100 * a + 10 * b + c)) →
  (37 ∣ (100 * b + 10 * c + a)) ∧
  (37 ∣ (100 * c + 10 * a + b)) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_by_37_l3573_357303


namespace NUMINAMATH_CALUDE_knight_moves_equal_for_7x7_l3573_357367

/-- Represents a position on a chessboard -/
structure Position :=
  (x : Nat) (y : Nat)

/-- Represents a knight's move on a chessboard -/
inductive KnightMove : Position → Position → Prop where
  | move_1 {x y : Nat} : KnightMove ⟨x, y⟩ ⟨x + 1, y + 2⟩
  | move_2 {x y : Nat} : KnightMove ⟨x, y⟩ ⟨x + 2, y + 1⟩
  | move_3 {x y : Nat} : KnightMove ⟨x, y⟩ ⟨x + 2, y - 1⟩
  | move_4 {x y : Nat} : KnightMove ⟨x, y⟩ ⟨x + 1, y - 2⟩
  | move_5 {x y : Nat} : KnightMove ⟨x, y⟩ ⟨x - 1, y - 2⟩
  | move_6 {x y : Nat} : KnightMove ⟨x, y⟩ ⟨x - 2, y - 1⟩
  | move_7 {x y : Nat} : KnightMove ⟨x, y⟩ ⟨x - 2, y + 1⟩
  | move_8 {x y : Nat} : KnightMove ⟨x, y⟩ ⟨x - 1, y + 2⟩

/-- The minimum number of moves for a knight to reach a target position from a start position -/
def minKnightMoves (start target : Position) : Nat :=
  sorry

theorem knight_moves_equal_for_7x7 :
  let start := Position.mk 0 0
  let upperRight := Position.mk 6 6
  let lowerRight := Position.mk 6 0
  minKnightMoves start upperRight = minKnightMoves start lowerRight :=
by
  sorry

end NUMINAMATH_CALUDE_knight_moves_equal_for_7x7_l3573_357367


namespace NUMINAMATH_CALUDE_words_with_vowels_count_l3573_357316

def alphabet : Finset Char := {'A', 'B', 'C', 'D', 'E', 'F'}
def vowels : Finset Char := {'A', 'E'}
def consonants : Finset Char := alphabet \ vowels

def word_length : Nat := 5

theorem words_with_vowels_count :
  (alphabet.card ^ word_length) - (consonants.card ^ word_length) = 6752 := by
  sorry

end NUMINAMATH_CALUDE_words_with_vowels_count_l3573_357316


namespace NUMINAMATH_CALUDE_parabola_point_ordering_l3573_357392

/-- Represents a parabola of the form y = -x^2 + 2x + c -/
def Parabola (c : ℝ) := {p : ℝ × ℝ | p.2 = -p.1^2 + 2*p.1 + c}

theorem parabola_point_ordering (c : ℝ) (y₁ y₂ y₃ : ℝ) 
  (h₁ : (0, y₁) ∈ Parabola c)
  (h₂ : (1, y₂) ∈ Parabola c)
  (h₃ : (3, y₃) ∈ Parabola c) :
  y₂ > y₁ ∧ y₁ > y₃ := by
  sorry

end NUMINAMATH_CALUDE_parabola_point_ordering_l3573_357392


namespace NUMINAMATH_CALUDE_decreasing_f_implies_a_bound_l3573_357377

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^3 + a*x^2 + x + 1

-- Define the derivative of f
def f_deriv (a : ℝ) (x : ℝ) : ℝ := 3*x^2 + 2*a*x + 1

-- Theorem statement
theorem decreasing_f_implies_a_bound (a : ℝ) :
  (∀ x ∈ Set.Ioo (-2/3) (-1/3), f_deriv a x < 0) →
  a ≥ 7/4 := by
  sorry

end NUMINAMATH_CALUDE_decreasing_f_implies_a_bound_l3573_357377


namespace NUMINAMATH_CALUDE_card_selection_count_l3573_357363

/-- Represents a standard deck of cards with an additional special suit -/
structure Deck :=
  (standard_cards : Nat)
  (special_suit_cards : Nat)
  (ace_count : Nat)

/-- Represents the selection criteria for the cards -/
structure Selection :=
  (total_cards : Nat)
  (min_aces : Nat)
  (different_suits : Bool)

/-- Calculates the number of ways to choose cards according to the given criteria -/
def choose_cards (d : Deck) (s : Selection) : Nat :=
  sorry

/-- The main theorem to be proved -/
theorem card_selection_count (d : Deck) (s : Selection) :
  d.standard_cards = 52 →
  d.special_suit_cards = 13 →
  d.ace_count = 4 →
  s.total_cards = 5 →
  s.min_aces = 1 →
  s.different_suits = true →
  choose_cards d s = 114244 :=
sorry

end NUMINAMATH_CALUDE_card_selection_count_l3573_357363


namespace NUMINAMATH_CALUDE_impossible_tiling_l3573_357389

/-- Represents a T-tetromino placement on a checkerboard -/
structure TTetromino where
  blackMajor : ℕ  -- number of T-tetrominoes with 3 black squares
  whiteMajor : ℕ  -- number of T-tetrominoes with 3 white squares

/-- The size of the grid -/
def gridSize : ℕ := 10

/-- The total number of squares in the grid -/
def totalSquares : ℕ := gridSize * gridSize

/-- Represents the coloring constraint of the checkerboard pattern -/
def colorConstraint (t : TTetromino) : Prop :=
  3 * t.blackMajor + t.whiteMajor = totalSquares / 2 ∧
  t.blackMajor + 3 * t.whiteMajor = totalSquares / 2

/-- Theorem stating the impossibility of tiling the grid -/
theorem impossible_tiling : ¬ ∃ t : TTetromino, colorConstraint t := by
  sorry

end NUMINAMATH_CALUDE_impossible_tiling_l3573_357389


namespace NUMINAMATH_CALUDE_xy_equals_three_l3573_357310

theorem xy_equals_three (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hdist : x ≠ y)
  (h : x + 3 / x = y + 3 / y) : x * y = 3 := by
  sorry

end NUMINAMATH_CALUDE_xy_equals_three_l3573_357310


namespace NUMINAMATH_CALUDE_root_reciprocal_sum_l3573_357324

theorem root_reciprocal_sum (x₁ x₂ : ℝ) : 
  x₁^2 - 3*x₁ - 1 = 0 → x₂^2 - 3*x₂ - 1 = 0 → x₁ ≠ x₂ → 
  (1/x₁) + (1/x₂) = -3 := by
  sorry

end NUMINAMATH_CALUDE_root_reciprocal_sum_l3573_357324


namespace NUMINAMATH_CALUDE_unique_solution_l3573_357322

def A (x : ℝ) : Set ℝ := {x^2, x+1, -3}
def B (x : ℝ) : Set ℝ := {x-5, 2*x-1, x^2+1}

theorem unique_solution : 
  ∃! x : ℝ, A x ∩ B x = {-3} ∧ x = -1 := by sorry

end NUMINAMATH_CALUDE_unique_solution_l3573_357322


namespace NUMINAMATH_CALUDE_solve_marbles_problem_l3573_357390

def marbles_problem (wolfgang_marbles : ℕ) : Prop :=
  let ludo_marbles := wolfgang_marbles + (wolfgang_marbles / 4)
  let total_wolfgang_ludo := wolfgang_marbles + ludo_marbles
  let michael_marbles := (2 * total_wolfgang_ludo) / 3
  let total_marbles := total_wolfgang_ludo + michael_marbles
  (wolfgang_marbles = 16) →
  (total_marbles / 3 = 20)

theorem solve_marbles_problem :
  marbles_problem 16 := by sorry

end NUMINAMATH_CALUDE_solve_marbles_problem_l3573_357390


namespace NUMINAMATH_CALUDE_line_plane_relationships_l3573_357321

/-- A line in 3D space -/
structure Line3D where
  -- Add necessary fields

/-- A plane in 3D space -/
structure Plane where
  -- Add necessary fields

/-- Defines when a line is parallel to a plane -/
def line_parallel_to_plane (l : Line3D) (p : Plane) : Prop := sorry

/-- Defines when a line is contained in a plane -/
def line_in_plane (l : Line3D) (p : Plane) : Prop := sorry

/-- Defines when two lines are parallel -/
def lines_parallel (l1 l2 : Line3D) : Prop := sorry

/-- Defines when a line intersects a plane -/
def line_intersects_plane (l : Line3D) (p : Plane) : Prop := sorry

/-- Theorem representing the four statements -/
theorem line_plane_relationships :
  (∀ (a b : Line3D) (α : Plane),
    line_parallel_to_plane a α → line_in_plane b α → lines_parallel a b) = False
  ∧
  (∀ (a b : Line3D) (α : Plane) (P : Point),
    line_intersects_plane a α → line_in_plane b α → ¬lines_parallel a b) = True
  ∧
  (∀ (a : Line3D) (α : Plane),
    ¬line_in_plane a α → line_parallel_to_plane a α) = False
  ∧
  (∀ (a b : Line3D) (α : Plane),
    line_parallel_to_plane a α → line_parallel_to_plane b α → lines_parallel a b) = False :=
by sorry

end NUMINAMATH_CALUDE_line_plane_relationships_l3573_357321


namespace NUMINAMATH_CALUDE_modern_literature_marks_l3573_357380

theorem modern_literature_marks
  (geography : ℕ) (history_gov : ℕ) (art : ℕ) (comp_sci : ℕ) (avg : ℚ) :
  geography = 56 →
  history_gov = 60 →
  art = 72 →
  comp_sci = 85 →
  avg = 70.6 →
  ∃ (modern_lit : ℕ),
    (geography + history_gov + art + comp_sci + modern_lit : ℚ) / 5 = avg ∧
    modern_lit = 80 := by
  sorry

end NUMINAMATH_CALUDE_modern_literature_marks_l3573_357380


namespace NUMINAMATH_CALUDE_middle_integer_of_pairwise_sums_l3573_357398

theorem middle_integer_of_pairwise_sums (x y z : ℤ) 
  (h1 : x < y) (h2 : y < z)
  (sum_xy : x + y = 22)
  (sum_xz : x + z = 24)
  (sum_yz : y + z = 16) :
  y = 7 := by
sorry

end NUMINAMATH_CALUDE_middle_integer_of_pairwise_sums_l3573_357398


namespace NUMINAMATH_CALUDE_inscribed_rectangle_area_bounds_l3573_357378

/-- A rectangle inscribed in a unit square -/
structure InscribedRectangle where
  width : ℝ
  height : ℝ
  width_positive : 0 < width
  height_positive : 0 < height
  fits_in_square : width ≤ 1 ∧ height ≤ 1

/-- The area of an inscribed rectangle -/
def area (r : InscribedRectangle) : ℝ := r.width * r.height

/-- An inscribed rectangle is a square if its width equals its height -/
def is_square (r : InscribedRectangle) : Prop := r.width = r.height

theorem inscribed_rectangle_area_bounds (r : InscribedRectangle) :
  (¬ is_square r → 0 < area r ∧ area r < 1/2) ∧
  (is_square r → area r = 1/2) := by
  sorry

end NUMINAMATH_CALUDE_inscribed_rectangle_area_bounds_l3573_357378


namespace NUMINAMATH_CALUDE_circle_probabilities_l3573_357345

/-- A type representing the 10 equally spaced points on a circle -/
inductive CirclePoint
  | one | two | three | four | five | six | seven | eight | nine | ten

/-- Function to check if two points form a diameter -/
def is_diameter (p1 p2 : CirclePoint) : Prop := sorry

/-- Function to check if three points form a right triangle -/
def is_right_triangle (p1 p2 p3 : CirclePoint) : Prop := sorry

/-- Function to check if four points form a rectangle -/
def is_rectangle (p1 p2 p3 p4 : CirclePoint) : Prop := sorry

/-- The number of ways to choose n items from a set of 10 -/
def choose_10 (n : Nat) : Nat := sorry

theorem circle_probabilities :
  (∃ (num_diameters : Nat), 
    num_diameters / choose_10 2 = 1 / 9 ∧
    (∀ p1 p2 : CirclePoint, p1 ≠ p2 → 
      (Nat.card {pair | pair = (p1, p2) ∧ is_diameter p1 p2} = num_diameters))) ∧
  (∃ (num_right_triangles : Nat),
    num_right_triangles / choose_10 3 = 1 / 3 ∧
    (∀ p1 p2 p3 : CirclePoint, p1 ≠ p2 ∧ p2 ≠ p3 ∧ p1 ≠ p3 →
      (Nat.card {triple | triple = (p1, p2, p3) ∧ is_right_triangle p1 p2 p3} = num_right_triangles))) ∧
  (∃ (num_rectangles : Nat),
    num_rectangles / choose_10 4 = 1 / 21 ∧
    (∀ p1 p2 p3 p4 : CirclePoint, p1 ≠ p2 ∧ p2 ≠ p3 ∧ p3 ≠ p4 ∧ p1 ≠ p3 ∧ p1 ≠ p4 ∧ p2 ≠ p4 →
      (Nat.card {quad | quad = (p1, p2, p3, p4) ∧ is_rectangle p1 p2 p3 p4} = num_rectangles))) :=
by sorry

end NUMINAMATH_CALUDE_circle_probabilities_l3573_357345


namespace NUMINAMATH_CALUDE_smallest_discount_value_l3573_357382

theorem smallest_discount_value : ∃ n : ℕ+, 
  (∀ m : ℕ+, m < n → 
    (1 - (m : ℝ) / 100 ≥ (1 - 0.20)^2 ∨ 
     1 - (m : ℝ) / 100 ≥ (1 - 0.15)^3 ∨ 
     1 - (m : ℝ) / 100 ≥ (1 - 0.30) * (1 - 0.10))) ∧ 
  (1 - (n : ℝ) / 100 < (1 - 0.20)^2 ∧ 
   1 - (n : ℝ) / 100 < (1 - 0.15)^3 ∧ 
   1 - (n : ℝ) / 100 < (1 - 0.30) * (1 - 0.10)) ∧ 
  n = 39 := by
  sorry

end NUMINAMATH_CALUDE_smallest_discount_value_l3573_357382


namespace NUMINAMATH_CALUDE_acid_dilution_l3573_357352

/-- Proves that adding 30 ounces of pure water to 50 ounces of a 40% acid solution results in a 25% acid solution -/
theorem acid_dilution (initial_volume : ℝ) (initial_concentration : ℝ) 
  (water_added : ℝ) (final_concentration : ℝ) :
  initial_volume = 50 →
  initial_concentration = 0.4 →
  water_added = 30 →
  final_concentration = 0.25 →
  (initial_volume * initial_concentration) / (initial_volume + water_added) = final_concentration :=
by
  sorry

#check acid_dilution

end NUMINAMATH_CALUDE_acid_dilution_l3573_357352


namespace NUMINAMATH_CALUDE_intersection_height_l3573_357317

/-- The height of the intersection point of lines drawn between two poles -/
theorem intersection_height (h1 h2 d : ℝ) (h1_pos : 0 < h1) (h2_pos : 0 < h2) (d_pos : 0 < d) :
  let x := (h1 * h2 * d) / (h1 * d + h2 * d)
  h1 = 20 → h2 = 80 → d = 100 → x = 16 := by
  sorry


end NUMINAMATH_CALUDE_intersection_height_l3573_357317


namespace NUMINAMATH_CALUDE_sqrt_x_squared_y_simplification_l3573_357311

theorem sqrt_x_squared_y_simplification (x y : ℝ) (h : x * y < 0) :
  Real.sqrt (x^2 * y) = -x * Real.sqrt y := by sorry

end NUMINAMATH_CALUDE_sqrt_x_squared_y_simplification_l3573_357311


namespace NUMINAMATH_CALUDE_right_triangle_sine_calculation_l3573_357313

theorem right_triangle_sine_calculation (D E F : ℝ) :
  0 < D ∧ D < π/2 →
  0 < E ∧ E < π/2 →
  0 < F ∧ F < π/2 →
  D + E + F = π →
  Real.sin D = 5/13 →
  Real.sin E = 1 →
  Real.sin F = 12/13 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_sine_calculation_l3573_357313


namespace NUMINAMATH_CALUDE_max_term_a_l3573_357369

def a (n : ℕ) : ℚ := n / (n^2 + 2020)

theorem max_term_a :
  ∀ k : ℕ, k ≠ 45 → a k ≤ a 45 := by sorry

end NUMINAMATH_CALUDE_max_term_a_l3573_357369


namespace NUMINAMATH_CALUDE_percentage_reduction_l3573_357373

theorem percentage_reduction (P : ℝ) : (85 * P / 100) - 11 = 23 → P = 40 := by
  sorry

end NUMINAMATH_CALUDE_percentage_reduction_l3573_357373


namespace NUMINAMATH_CALUDE_saree_price_l3573_357374

theorem saree_price (final_price : ℝ) (discount1 : ℝ) (discount2 : ℝ) 
  (h1 : final_price = 331.2)
  (h2 : discount1 = 0.1)
  (h3 : discount2 = 0.08) : 
  ∃ original_price : ℝ, 
    original_price = 400 ∧ 
    final_price = original_price * (1 - discount1) * (1 - discount2) :=
sorry

end NUMINAMATH_CALUDE_saree_price_l3573_357374


namespace NUMINAMATH_CALUDE_framed_painting_ratio_l3573_357312

theorem framed_painting_ratio : 
  let painting_size : ℝ := 20
  let frame_side (x : ℝ) := x
  let frame_top_bottom (x : ℝ) := 3 * x
  let framed_width (x : ℝ) := painting_size + 2 * frame_side x
  let framed_height (x : ℝ) := painting_size + 2 * frame_top_bottom x
  let frame_area (x : ℝ) := framed_width x * framed_height x - painting_size^2
  ∃ x : ℝ, 
    x > 0 ∧ 
    frame_area x = painting_size^2 ∧
    (min (framed_width x) (framed_height x)) / (max (framed_width x) (framed_height x)) = 4/7 :=
by sorry

end NUMINAMATH_CALUDE_framed_painting_ratio_l3573_357312


namespace NUMINAMATH_CALUDE_solve_equation_l3573_357383

theorem solve_equation (x : ℝ) (h : 9 - (4/x) = 7 + (8/x)) : x = 6 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l3573_357383


namespace NUMINAMATH_CALUDE_valid_trapezoid_iff_s_gt_8r_l3573_357323

/-- A right-angled tangential trapezoid with an inscribed circle -/
structure RightAngledTangentialTrapezoid where
  /-- Radius of the inscribed circle -/
  r : ℝ
  /-- Perimeter of the trapezoid -/
  s : ℝ
  /-- r is positive -/
  r_pos : r > 0
  /-- s is positive -/
  s_pos : s > 0

/-- Theorem: A valid right-angled tangential trapezoid exists iff s > 8r -/
theorem valid_trapezoid_iff_s_gt_8r (t : RightAngledTangentialTrapezoid) :
  ∃ (trapezoid : RightAngledTangentialTrapezoid), trapezoid.r = t.r ∧ trapezoid.s = t.s ↔ t.s > 8 * t.r :=
by sorry

end NUMINAMATH_CALUDE_valid_trapezoid_iff_s_gt_8r_l3573_357323


namespace NUMINAMATH_CALUDE_ferris_wheel_seats_l3573_357319

-- Define the total number of people that can ride at once
def total_riders : ℕ := 4

-- Define the capacity of each seat
def seat_capacity : ℕ := 2

-- Define the number of seats on the Ferris wheel
def num_seats : ℕ := total_riders / seat_capacity

-- Theorem statement
theorem ferris_wheel_seats : num_seats = 2 := by
  sorry

end NUMINAMATH_CALUDE_ferris_wheel_seats_l3573_357319


namespace NUMINAMATH_CALUDE_line_equation_through_points_l3573_357328

theorem line_equation_through_points (x y : ℝ) : 
  (2 * x - y - 2 = 0) ↔ 
  (∃ t : ℝ, x = 1 - t ∧ y = -2 * t) :=
sorry

end NUMINAMATH_CALUDE_line_equation_through_points_l3573_357328


namespace NUMINAMATH_CALUDE_sum_of_squares_of_roots_l3573_357304

theorem sum_of_squares_of_roots (a b c : ℝ) : 
  (a^3 - 15*a^2 + 25*a - 10 = 0) →
  (b^3 - 15*b^2 + 25*b - 10 = 0) →
  (c^3 - 15*c^2 + 25*c - 10 = 0) →
  a^2 + b^2 + c^2 = 175 := by
sorry

end NUMINAMATH_CALUDE_sum_of_squares_of_roots_l3573_357304


namespace NUMINAMATH_CALUDE_larger_number_problem_l3573_357356

theorem larger_number_problem (L S : ℕ) 
  (h1 : L - S = 20775)
  (h2 : L = 23 * S + 143) :
  L = 21713 := by
sorry

end NUMINAMATH_CALUDE_larger_number_problem_l3573_357356


namespace NUMINAMATH_CALUDE_project_hours_theorem_l3573_357375

/-- Represents the hours worked by three people on a project -/
structure ProjectHours where
  least : ℕ
  middle : ℕ
  most : ℕ

/-- The total hours worked on the project -/
def total_hours (h : ProjectHours) : ℕ := h.least + h.middle + h.most

/-- The condition that the working times are in the ratio 1:2:3 -/
def ratio_condition (h : ProjectHours) : Prop :=
  h.middle = 2 * h.least ∧ h.most = 3 * h.least

/-- The condition that the hardest working person worked 40 hours more than the person who worked the least -/
def difference_condition (h : ProjectHours) : Prop :=
  h.most = h.least + 40

theorem project_hours_theorem (h : ProjectHours) 
  (hc1 : ratio_condition h) 
  (hc2 : difference_condition h) : 
  total_hours h = 120 := by
  sorry

#check project_hours_theorem

end NUMINAMATH_CALUDE_project_hours_theorem_l3573_357375


namespace NUMINAMATH_CALUDE_square_difference_formula_inapplicable_l3573_357336

theorem square_difference_formula_inapplicable :
  ¬∃ (a b : ℝ → ℝ), ∀ x, (x + 1) * (1 + x) = a x ^ 2 - b x ^ 2 :=
sorry

end NUMINAMATH_CALUDE_square_difference_formula_inapplicable_l3573_357336


namespace NUMINAMATH_CALUDE_swimmer_journey_l3573_357360

/-- Swimmer's journey problem -/
theorem swimmer_journey 
  (swimmer_speed : ℝ) 
  (current_speed : ℝ) 
  (distance_PQ : ℝ) 
  (distance_QR : ℝ) 
  (h1 : swimmer_speed = 1)
  (h2 : distance_PQ / (swimmer_speed + current_speed) + distance_QR / swimmer_speed = 3)
  (h3 : distance_QR / (swimmer_speed - current_speed) + distance_PQ / (swimmer_speed - current_speed) = 6)
  (h4 : (distance_PQ + distance_QR) / (swimmer_speed + current_speed) = 5/2)
  : (distance_QR + distance_PQ) / (swimmer_speed - current_speed) = 15/2 := by
  sorry

end NUMINAMATH_CALUDE_swimmer_journey_l3573_357360


namespace NUMINAMATH_CALUDE_remainder_1234567890_mod_99_l3573_357366

theorem remainder_1234567890_mod_99 : 1234567890 % 99 = 72 := by
  sorry

end NUMINAMATH_CALUDE_remainder_1234567890_mod_99_l3573_357366


namespace NUMINAMATH_CALUDE_sqrt_sum_equals_4_sqrt_5_l3573_357309

theorem sqrt_sum_equals_4_sqrt_5 : 
  Real.sqrt (24 - 8 * Real.sqrt 2) + Real.sqrt (24 + 8 * Real.sqrt 2) = 4 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_equals_4_sqrt_5_l3573_357309


namespace NUMINAMATH_CALUDE_f_negative_l3573_357334

-- Define an odd function
def odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

-- Define the function for x > 0
def f_positive (x : ℝ) : ℝ :=
  x^2 - x + 1

-- Theorem statement
theorem f_negative (f : ℝ → ℝ) (h_odd : odd_function f) 
  (h_positive : ∀ x > 0, f x = f_positive x) :
  ∀ x < 0, f x = -x^2 - x - 1 := by
sorry

end NUMINAMATH_CALUDE_f_negative_l3573_357334


namespace NUMINAMATH_CALUDE_wall_bricks_l3573_357386

/-- Represents the number of bricks in the wall -/
def total_bricks : ℕ := 360

/-- Represents Brenda's time to build the wall alone (in hours) -/
def brenda_time : ℕ := 8

/-- Represents Brandon's time to build the wall alone (in hours) -/
def brandon_time : ℕ := 12

/-- Represents the decrease in combined output (in bricks per hour) -/
def output_decrease : ℕ := 15

/-- Represents the time taken to build the wall together (in hours) -/
def combined_time : ℕ := 6

/-- Theorem stating that the number of bricks in the wall is 360 -/
theorem wall_bricks : 
  (combined_time : ℚ) * ((total_bricks / brenda_time + total_bricks / brandon_time) - output_decrease) = total_bricks := by
  sorry

#check wall_bricks

end NUMINAMATH_CALUDE_wall_bricks_l3573_357386


namespace NUMINAMATH_CALUDE_multiple_of_99_sum_of_digits_l3573_357326

theorem multiple_of_99_sum_of_digits (A B : ℕ) : 
  A ≤ 9 → B ≤ 9 → 
  (100000 * A + 15000 + 100 * B + 94) % 99 = 0 →
  A + B = 8 := by
sorry

end NUMINAMATH_CALUDE_multiple_of_99_sum_of_digits_l3573_357326


namespace NUMINAMATH_CALUDE_greatest_divisor_four_consecutive_integers_l3573_357347

theorem greatest_divisor_four_consecutive_integers :
  ∀ n : ℕ, n > 0 →
  ∃ m : ℕ, m > 0 ∧ m ∣ (n * (n + 1) * (n + 2) * (n + 3)) ∧
  ∀ k : ℕ, k > m → ¬(∀ p : ℕ, p > 0 → k ∣ (p * (p + 1) * (p + 2) * (p + 3))) →
  m = 24 :=
by sorry

end NUMINAMATH_CALUDE_greatest_divisor_four_consecutive_integers_l3573_357347


namespace NUMINAMATH_CALUDE_real_part_of_fraction_l3573_357357

theorem real_part_of_fraction (z : ℂ) (x : ℝ) (h1 : z.im ≠ 0) (h2 : Complex.abs z = 2) (h3 : z.re = x) :
  (1 / (1 - z)).re = (1 - x) / (5 - 2 * x) := by
  sorry

end NUMINAMATH_CALUDE_real_part_of_fraction_l3573_357357


namespace NUMINAMATH_CALUDE_combined_swimming_distance_l3573_357314

/-- Given swimming distances for Jamir, Sarah, and Julien, prove their combined weekly distance --/
theorem combined_swimming_distance
  (julien_daily : ℕ)
  (sarah_daily : ℕ)
  (jamir_daily : ℕ)
  (days_in_week : ℕ)
  (h1 : julien_daily = 50)
  (h2 : sarah_daily = 2 * julien_daily)
  (h3 : jamir_daily = sarah_daily + 20)
  (h4 : days_in_week = 7) :
  julien_daily * days_in_week +
  sarah_daily * days_in_week +
  jamir_daily * days_in_week = 1890 := by
sorry

end NUMINAMATH_CALUDE_combined_swimming_distance_l3573_357314


namespace NUMINAMATH_CALUDE_arithmetic_mean_problem_l3573_357362

theorem arithmetic_mean_problem (x : ℚ) : 
  (x + 10 + 20 + 3*x + 18 + (3*x + 6)) / 5 = 30 → x = 96/7 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_problem_l3573_357362


namespace NUMINAMATH_CALUDE_max_roses_proof_l3573_357361

-- Define the pricing structure
def individual_price : ℚ := 6.3
def dozen_price : ℚ := 36
def two_dozen_price : ℚ := 50
def five_dozen_price : ℚ := 110

-- Define Maria's budget constraints
def total_budget : ℚ := 680
def min_red_roses_budget : ℚ := 200

-- Define the function to calculate the maximum number of roses
def max_roses : ℕ := 360

-- Theorem statement
theorem max_roses_proof :
  ∀ (purchase_strategy : ℕ → ℕ → ℕ → ℕ → ℚ),
  (∀ a b c d, purchase_strategy a b c d * individual_price +
              purchase_strategy a b c d * dozen_price / 12 +
              purchase_strategy a b c d * two_dozen_price / 24 +
              purchase_strategy a b c d * five_dozen_price / 60 ≤ total_budget) →
  (∀ a b c d, purchase_strategy a b c d * five_dozen_price / 60 ≥ min_red_roses_budget) →
  (∀ a b c d, purchase_strategy a b c d + purchase_strategy a b c d * 12 +
              purchase_strategy a b c d * 24 + purchase_strategy a b c d * 60 ≤ max_roses) :=
by sorry

end NUMINAMATH_CALUDE_max_roses_proof_l3573_357361


namespace NUMINAMATH_CALUDE_positive_net_return_l3573_357331

/-- Represents the annual interest rate of a mortgage loan as a percentage -/
def mortgage_rate : ℝ := 12.5

/-- Represents the annual dividend rate of preferred shares as a percentage -/
def dividend_rate : ℝ := 17

/-- Calculates the net return from keeping shares and taking a mortgage loan -/
def net_return (dividend : ℝ) (mortgage : ℝ) : ℝ := dividend - mortgage

/-- Theorem stating that the net return is positive given the specified rates -/
theorem positive_net_return : net_return dividend_rate mortgage_rate > 0 := by
  sorry

end NUMINAMATH_CALUDE_positive_net_return_l3573_357331


namespace NUMINAMATH_CALUDE_even_function_property_l3573_357329

def EvenFunction (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

theorem even_function_property (f : ℝ → ℝ) 
  (h_even : EvenFunction f) 
  (h_nonneg : ∀ x ≥ 0, f x = x * (x + 1)) :
  ∀ x < 0, f x = -x * (1 - x) := by
sorry

end NUMINAMATH_CALUDE_even_function_property_l3573_357329


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l3573_357343

theorem quadratic_equation_solution :
  let f : ℝ → ℝ := λ x => 2*x*(x+1) - 3*(x+1)
  ∃ x₁ x₂ : ℝ, x₁ = -1 ∧ x₂ = 3/2 ∧ (∀ x : ℝ, f x = 0 ↔ x = x₁ ∨ x = x₂) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l3573_357343


namespace NUMINAMATH_CALUDE_j_type_sequence_properties_l3573_357340

/-- Definition of a J_k type sequence -/
def is_J_k_type (a : ℕ → ℝ) (k : ℕ) : Prop :=
  ∀ n : ℕ, ∃ m : ℕ, (a (n + k))^2 = a n * a (n + 2*k)

theorem j_type_sequence_properties 
  (a : ℕ → ℝ) 
  (h_pos : ∀ n, a n > 0) :
  (is_J_k_type a 2 ∧ a 2 = 8 ∧ a 8 = 1 → 
    ∀ n : ℕ, a (2*n) = 2^(4-n)) ∧
  (is_J_k_type a 3 ∧ is_J_k_type a 4 → 
    ∃ r : ℝ, ∀ n : ℕ, a (n+1) = r * a n) :=
sorry

end NUMINAMATH_CALUDE_j_type_sequence_properties_l3573_357340
