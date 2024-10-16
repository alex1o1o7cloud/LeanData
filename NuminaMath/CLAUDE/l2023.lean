import Mathlib

namespace NUMINAMATH_CALUDE_deposit_percentage_l2023_202360

theorem deposit_percentage (deposit : ℝ) (remaining : ℝ) :
  deposit = 50 →
  remaining = 950 →
  (deposit / (deposit + remaining)) * 100 = 5 := by
sorry

end NUMINAMATH_CALUDE_deposit_percentage_l2023_202360


namespace NUMINAMATH_CALUDE_percentage_division_problem_l2023_202308

theorem percentage_division_problem : (168 / 100 * 1265) / 6 = 354.2 := by
  sorry

end NUMINAMATH_CALUDE_percentage_division_problem_l2023_202308


namespace NUMINAMATH_CALUDE_cards_in_hospital_eq_403_l2023_202369

/-- The number of cards Mariela received while in the hospital -/
def cards_in_hospital : ℕ := 690 - 287

/-- Theorem stating that Mariela received 403 cards while in the hospital -/
theorem cards_in_hospital_eq_403 : cards_in_hospital = 403 := by
  sorry

end NUMINAMATH_CALUDE_cards_in_hospital_eq_403_l2023_202369


namespace NUMINAMATH_CALUDE_tan_alpha_one_third_implies_cos_2alpha_over_expression_l2023_202319

theorem tan_alpha_one_third_implies_cos_2alpha_over_expression (α : Real) 
  (h : Real.tan α = 1/3) : 
  (Real.cos (2*α)) / (2 * Real.sin α * Real.cos α + (Real.cos α)^2) = 8/15 := by
  sorry

end NUMINAMATH_CALUDE_tan_alpha_one_third_implies_cos_2alpha_over_expression_l2023_202319


namespace NUMINAMATH_CALUDE_existence_of_special_numbers_l2023_202381

theorem existence_of_special_numbers :
  ∃ (a b c : ℕ), 
    (a > 10^10) ∧ 
    (b > 10^10) ∧ 
    (c > 10^10) ∧ 
    ((a * b * c) % (a + 2012) = 0) ∧
    ((a * b * c) % (b + 2012) = 0) ∧
    ((a * b * c) % (c + 2012) = 0) :=
by sorry

end NUMINAMATH_CALUDE_existence_of_special_numbers_l2023_202381


namespace NUMINAMATH_CALUDE_science_club_election_l2023_202325

theorem science_club_election (total_candidates : Nat) (past_officers : Nat) (positions : Nat) :
  total_candidates = 20 →
  past_officers = 10 →
  positions = 4 →
  (Nat.choose total_candidates positions -
   (Nat.choose (total_candidates - past_officers) positions +
    Nat.choose past_officers 1 * Nat.choose (total_candidates - past_officers) (positions - 1))) = 3435 :=
by sorry

end NUMINAMATH_CALUDE_science_club_election_l2023_202325


namespace NUMINAMATH_CALUDE_systematic_sampling_theorem_l2023_202378

/-- Represents the systematic sampling problem -/
structure SystematicSampling where
  total_students : ℕ
  sample_size : ℕ
  sampling_interval : ℕ
  first_random_number : ℕ

/-- Calculates the number of selected students within a given range -/
def selected_students_in_range (s : SystematicSampling) (lower : ℕ) (upper : ℕ) : ℕ :=
  sorry

/-- The main theorem to be proved -/
theorem systematic_sampling_theorem (s : SystematicSampling) 
  (h1 : s.total_students = 1000)
  (h2 : s.sample_size = 50)
  (h3 : s.sampling_interval = 20)
  (h4 : s.first_random_number = 15) :
  selected_students_in_range s 601 785 = 9 := by
  sorry

end NUMINAMATH_CALUDE_systematic_sampling_theorem_l2023_202378


namespace NUMINAMATH_CALUDE_reciprocal_roots_condition_l2023_202336

/-- The roots of the quadratic equation 2x^2 + 5x + k = 0 are reciprocal if and only if k = 2 -/
theorem reciprocal_roots_condition (k : ℝ) : 
  (∃ x y : ℝ, x ≠ 0 ∧ y ≠ 0 ∧ x * y = 1 ∧ 2 * x^2 + 5 * x + k = 0 ∧ 2 * y^2 + 5 * y + k = 0) ↔ 
  k = 2 :=
by sorry

end NUMINAMATH_CALUDE_reciprocal_roots_condition_l2023_202336


namespace NUMINAMATH_CALUDE_special_function_properties_l2023_202317

/-- A function satisfying the given conditions -/
def SpecialFunction (f : ℝ → ℝ) : Prop :=
  (∀ x, f x + f (4 - x) = 0) ∧ (∀ x, f (x + 2) - f (x - 2) = 0)

/-- Theorem stating the properties of the special function -/
theorem special_function_properties (f : ℝ → ℝ) (h : SpecialFunction f) :
  (∀ x, f (-x) = -f x) ∧ (∀ x, f (x + 4) = f x) := by
  sorry


end NUMINAMATH_CALUDE_special_function_properties_l2023_202317


namespace NUMINAMATH_CALUDE_distance_to_cegled_l2023_202326

/-- The problem setup for calculating the distance to Cegléd -/
structure TravelProblem where
  s : ℝ  -- Total distance from home to Cegléd
  v : ℝ  -- Planned speed for both Antal and Béla
  t : ℝ  -- Planned travel time
  s₁ : ℝ  -- Béla's travel distance when alone

/-- The conditions of the problem -/
def problem_conditions (p : TravelProblem) : Prop :=
  p.t = p.s / p.v ∧  -- Planned time
  p.s₁ = 4 * p.s / 5 ∧  -- Béla's solo distance
  p.s / 5 = 48 * (1 / 6) ∧  -- Final section travel time
  (4 * p.s₁) / (3 * p.v) = (4 * (p.s₁ + 2 * p.s / 5)) / (5 * p.v)  -- Time equivalence for travel

/-- The theorem stating that the total distance is 40 km -/
theorem distance_to_cegled (p : TravelProblem) 
  (h : problem_conditions p) : p.s = 40 := by
  sorry


end NUMINAMATH_CALUDE_distance_to_cegled_l2023_202326


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l2023_202361

/-- A polynomial with real coefficients -/
def g (p q r s : ℝ) (x : ℂ) : ℂ := x^4 + p*x^3 + q*x^2 + r*x + s

/-- Theorem stating that if g(1+i) = 0 and g(3i) = 0, then p + q + r + s = 9 -/
theorem sum_of_coefficients (p q r s : ℝ) :
  g p q r s (1 + Complex.I) = 0 →
  g p q r s (3 * Complex.I) = 0 →
  p + q + r + s = 9 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l2023_202361


namespace NUMINAMATH_CALUDE_square_minus_circle_area_l2023_202301

theorem square_minus_circle_area : 
  let square_side : ℝ := 4
  let circle_diameter : ℝ := 2
  let square_area := square_side * square_side
  let circle_area := Real.pi * (circle_diameter / 2) ^ 2
  square_area - circle_area = 16 - Real.pi := by
  sorry

end NUMINAMATH_CALUDE_square_minus_circle_area_l2023_202301


namespace NUMINAMATH_CALUDE_farm_cows_l2023_202379

/-- Given a farm with cows and horses, prove the number of cows -/
theorem farm_cows (total_horses : ℕ) (ratio_cows : ℕ) (ratio_horses : ℕ) 
  (h1 : total_horses = 6)
  (h2 : ratio_cows = 7)
  (h3 : ratio_horses = 2) :
  (ratio_cows : ℚ) / ratio_horses * total_horses = 21 := by
  sorry

end NUMINAMATH_CALUDE_farm_cows_l2023_202379


namespace NUMINAMATH_CALUDE_uranus_appearance_time_l2023_202393

/-- Represents time in hours and minutes -/
structure Time where
  hours : ℕ
  minutes : ℕ
  inv : minutes < 60

/-- Adds minutes to a given time -/
def addMinutes (t : Time) (m : ℕ) : Time :=
  let totalMinutes := t.minutes + m
  let newHours := t.hours + totalMinutes / 60
  let newMinutes := totalMinutes % 60
  ⟨newHours % 24, newMinutes, by sorry⟩

/-- Calculates the difference in minutes between two times -/
def minutesBetween (t1 t2 : Time) : ℕ :=
  (t2.hours - t1.hours) * 60 + (t2.minutes - t1.minutes)

theorem uranus_appearance_time
  (marsDisappearance : Time)
  (jupiterDelay : ℕ)
  (uranusDelay : ℕ)
  (h1 : marsDisappearance = ⟨0, 10, by sorry⟩)  -- 12:10 AM
  (h2 : jupiterDelay = 161)  -- 2 hours and 41 minutes
  (h3 : uranusDelay = 196)  -- 3 hours and 16 minutes
  : minutesBetween ⟨6, 0, by sorry⟩ (addMinutes (addMinutes marsDisappearance jupiterDelay) uranusDelay) = 7 :=
by sorry

end NUMINAMATH_CALUDE_uranus_appearance_time_l2023_202393


namespace NUMINAMATH_CALUDE_unique_solution_sqrt_equation_l2023_202395

/-- The equation √(x+2) + 2√(x-1) + 3√(3x-2) = 10 has a unique solution x = 2 -/
theorem unique_solution_sqrt_equation :
  ∃! x : ℝ, (x + 2 ≥ 0) ∧ (x - 1 ≥ 0) ∧ (3*x - 2 ≥ 0) ∧
  (Real.sqrt (x + 2) + 2 * Real.sqrt (x - 1) + 3 * Real.sqrt (3*x - 2) = 10) ∧
  x = 2 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_sqrt_equation_l2023_202395


namespace NUMINAMATH_CALUDE_higher_power_of_two_divisibility_l2023_202329

theorem higher_power_of_two_divisibility (n k : ℕ) : 
  ∃ i ∈ Finset.range k, ∀ j ∈ Finset.range k, j ≠ i → 
    (∃ m : ℕ, (n + i + 1) = 2^m * (2*l + 1) ∧ 
              ∀ p : ℕ, (n + j + 1) = 2^p * (2*q + 1) → m > p) :=
by sorry

end NUMINAMATH_CALUDE_higher_power_of_two_divisibility_l2023_202329


namespace NUMINAMATH_CALUDE_probability_matches_given_l2023_202310

def total_pens : ℕ := 8
def defective_pens : ℕ := 3
def pens_bought : ℕ := 2

def probability_no_defective (total : ℕ) (defective : ℕ) (bought : ℕ) : ℚ :=
  (Nat.choose (total - defective) bought : ℚ) / (Nat.choose total bought : ℚ)

theorem probability_matches_given :
  probability_no_defective total_pens defective_pens pens_bought = 5 / 14 :=
by sorry

end NUMINAMATH_CALUDE_probability_matches_given_l2023_202310


namespace NUMINAMATH_CALUDE_computer_contracts_probability_l2023_202373

theorem computer_contracts_probability 
  (p_hardware : ℝ) 
  (p_not_software : ℝ) 
  (p_at_least_one : ℝ) 
  (h1 : p_hardware = 4/5) 
  (h2 : p_not_software = 3/5) 
  (h3 : p_at_least_one = 5/6) : 
  p_hardware + (1 - p_not_software) - p_at_least_one = 11/30 :=
by sorry

end NUMINAMATH_CALUDE_computer_contracts_probability_l2023_202373


namespace NUMINAMATH_CALUDE_abs_neg_nine_equals_nine_l2023_202331

theorem abs_neg_nine_equals_nine : abs (-9 : ℤ) = 9 := by
  sorry

end NUMINAMATH_CALUDE_abs_neg_nine_equals_nine_l2023_202331


namespace NUMINAMATH_CALUDE_complementary_angle_l2023_202371

theorem complementary_angle (A : ℝ) (h : A = 25) : 90 - A = 65 := by
  sorry

end NUMINAMATH_CALUDE_complementary_angle_l2023_202371


namespace NUMINAMATH_CALUDE_inequality_proof_l2023_202333

theorem inequality_proof (a b : ℝ) (h1 : a > b) (h2 : b > 0) : a * 2^(-b) > b * 2^(-a) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2023_202333


namespace NUMINAMATH_CALUDE_sphere_surface_area_from_cube_l2023_202380

theorem sphere_surface_area_from_cube (cube_edge_length : ℝ) (sphere_radius : ℝ) : 
  cube_edge_length = 2 →
  sphere_radius^2 = 3 →
  4 * Real.pi * sphere_radius^2 = 12 * Real.pi :=
by
  sorry

end NUMINAMATH_CALUDE_sphere_surface_area_from_cube_l2023_202380


namespace NUMINAMATH_CALUDE_product_of_roots_l2023_202340

theorem product_of_roots (x : ℝ) : 
  (3 * x^2 + 6 * x - 81 = 0) → 
  (∃ α β : ℝ, (3 * α^2 + 6 * α - 81 = 0) ∧ 
              (3 * β^2 + 6 * β - 81 = 0) ∧ 
              (α * β = -27)) := by
  sorry

end NUMINAMATH_CALUDE_product_of_roots_l2023_202340


namespace NUMINAMATH_CALUDE_percentage_multiplication_equality_l2023_202396

theorem percentage_multiplication_equality : ∃ x : ℝ, 45 * x = (45 / 100) * 900 ∧ x = 9 := by
  sorry

end NUMINAMATH_CALUDE_percentage_multiplication_equality_l2023_202396


namespace NUMINAMATH_CALUDE_cos_135_degrees_l2023_202306

theorem cos_135_degrees :
  Real.cos (135 * π / 180) = -Real.sqrt 2 / 2 := by sorry

end NUMINAMATH_CALUDE_cos_135_degrees_l2023_202306


namespace NUMINAMATH_CALUDE_special_pair_characterization_l2023_202394

/-- A pair of integers is special if it is of the form (n, n-1) or (n-1, n) for some positive integer n. -/
def IsSpecialPair (p : ℤ × ℤ) : Prop :=
  ∃ n : ℤ, n > 0 ∧ (p = (n, n - 1) ∨ p = (n - 1, n))

/-- The sum of two pairs -/
def PairSum (p q : ℤ × ℤ) : ℤ × ℤ :=
  (p.1 + q.1, p.2 + q.2)

/-- A pair can be expressed as a sum of special pairs -/
def CanExpressAsSumOfSpecialPairs (p : ℤ × ℤ) : Prop :=
  ∃ (k : ℕ) (specialPairs : Fin k → ℤ × ℤ),
    k ≥ 2 ∧
    (∀ i, IsSpecialPair (specialPairs i)) ∧
    (∀ i j, i ≠ j → specialPairs i ≠ specialPairs j) ∧
    p = Finset.sum Finset.univ (λ i => specialPairs i)

theorem special_pair_characterization (n m : ℤ) 
    (h_positive : n > 0 ∧ m > 0)
    (h_not_special : ¬IsSpecialPair (n, m)) :
    CanExpressAsSumOfSpecialPairs (n, m) ↔ n + m ≥ (n - m)^2 := by
  sorry

end NUMINAMATH_CALUDE_special_pair_characterization_l2023_202394


namespace NUMINAMATH_CALUDE_hexagon_exterior_angles_sum_l2023_202332

/-- A hexagon is a polygon with 6 sides -/
def Hexagon : Type := Unit

/-- The sum of exterior angles of a polygon -/
def sum_exterior_angles (p : Type) : ℝ := sorry

/-- Theorem: The sum of the exterior angles of a hexagon is 360 degrees -/
theorem hexagon_exterior_angles_sum :
  sum_exterior_angles Hexagon = 360 :=
sorry

end NUMINAMATH_CALUDE_hexagon_exterior_angles_sum_l2023_202332


namespace NUMINAMATH_CALUDE_duck_flying_ratio_l2023_202327

/-- Represents the flying time of a duck during different seasons -/
structure DuckFlyingTime where
  total : ℕ
  south : ℕ
  east : ℕ

/-- Calculates the ratio of north flying time to south flying time -/
def northToSouthRatio (d : DuckFlyingTime) : ℚ :=
  let north := d.total - d.south - d.east
  (north : ℚ) / d.south

/-- Theorem stating that the ratio of north to south flying time is 2:1 -/
theorem duck_flying_ratio :
  ∀ d : DuckFlyingTime,
  d.total = 180 ∧ d.south = 40 ∧ d.east = 60 →
  northToSouthRatio d = 2 := by
  sorry


end NUMINAMATH_CALUDE_duck_flying_ratio_l2023_202327


namespace NUMINAMATH_CALUDE_least_perimeter_triangle_l2023_202391

def triangle_perimeter (a b c : ℕ) : ℕ := a + b + c

def is_triangle (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

theorem least_perimeter_triangle :
  ∃ (c : ℕ), 
    is_triangle 24 51 c ∧ 
    (∀ (x : ℕ), is_triangle 24 51 x → triangle_perimeter 24 51 c ≤ triangle_perimeter 24 51 x) ∧
    triangle_perimeter 24 51 c = 103 := by
  sorry

end NUMINAMATH_CALUDE_least_perimeter_triangle_l2023_202391


namespace NUMINAMATH_CALUDE_no_integer_solutions_l2023_202338

theorem no_integer_solutions : ¬ ∃ (a b c : ℤ), a^2 + b^2 = 8*c + 6 := by
  sorry

end NUMINAMATH_CALUDE_no_integer_solutions_l2023_202338


namespace NUMINAMATH_CALUDE_sin_plus_cos_equals_sqrt_a_plus_one_l2023_202397

theorem sin_plus_cos_equals_sqrt_a_plus_one (θ : Real) (a : Real) 
  (h1 : 0 < θ ∧ θ < π / 2) -- θ is an acute angle
  (h2 : Real.sin (2 * θ) = a) : -- sin 2θ = a
  Real.sin θ + Real.cos θ = Real.sqrt (a + 1) := by sorry

end NUMINAMATH_CALUDE_sin_plus_cos_equals_sqrt_a_plus_one_l2023_202397


namespace NUMINAMATH_CALUDE_watch_sale_gain_percentage_l2023_202324

/-- Proves that for a watch with a given cost price, sold at a loss, 
    if the selling price is increased by a certain amount, 
    the resulting gain percentage is as expected. -/
theorem watch_sale_gain_percentage 
  (cost_price : ℝ) 
  (loss_percentage : ℝ) 
  (price_increase : ℝ) : 
  cost_price = 1200 →
  loss_percentage = 10 →
  price_increase = 168 →
  let loss_amount := (loss_percentage / 100) * cost_price
  let initial_selling_price := cost_price - loss_amount
  let new_selling_price := initial_selling_price + price_increase
  let gain_amount := new_selling_price - cost_price
  let gain_percentage := (gain_amount / cost_price) * 100
  gain_percentage = 4 := by
sorry


end NUMINAMATH_CALUDE_watch_sale_gain_percentage_l2023_202324


namespace NUMINAMATH_CALUDE_f_properties_l2023_202312

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (1/2) * x^2 + a * Real.log x

theorem f_properties :
  (∀ x > 0, f (-1) x ≥ 1/2) ∧ 
  (f (-1) 1 = 1/2) ∧
  (∀ x ≥ 1, f 1 x < (2/3) * x^3) := by
  sorry

end NUMINAMATH_CALUDE_f_properties_l2023_202312


namespace NUMINAMATH_CALUDE_aaron_gave_five_sweets_l2023_202365

/-- Represents the number of sweets given to a friend -/
def sweets_given_to_friend (initial_cherry initial_strawberry initial_pineapple : ℕ) 
  (remaining : ℕ) : ℕ :=
  initial_cherry / 2 + initial_strawberry / 2 + initial_pineapple / 2 - remaining

/-- Proves that Aaron gave 5 cherry sweets to his friend -/
theorem aaron_gave_five_sweets : 
  sweets_given_to_friend 30 40 50 55 = 5 := by
  sorry

#eval sweets_given_to_friend 30 40 50 55

end NUMINAMATH_CALUDE_aaron_gave_five_sweets_l2023_202365


namespace NUMINAMATH_CALUDE_cow_ratio_theorem_l2023_202344

theorem cow_ratio_theorem (big_cows small_cows : ℕ) 
  (h : big_cows * 7 = small_cows * 6) : 
  (small_cows - big_cows : ℚ) / small_cows = 1 / 7 := by
  sorry

end NUMINAMATH_CALUDE_cow_ratio_theorem_l2023_202344


namespace NUMINAMATH_CALUDE_fathers_age_multiple_l2023_202304

theorem fathers_age_multiple (son_age : ℕ) (father_age : ℕ) (k : ℕ) : 
  father_age = 27 →
  father_age = k * son_age + 3 →
  father_age + 3 = 2 * (son_age + 3) + 8 →
  k = 3 := by
  sorry

end NUMINAMATH_CALUDE_fathers_age_multiple_l2023_202304


namespace NUMINAMATH_CALUDE_solution_set_characterization_l2023_202328

/-- The set of solutions to the equation z + y² + x³ = xyz with x = gcd(y, z) -/
def SolutionSet : Set (ℕ × ℕ × ℕ) :=
  {s | s.1 > 0 ∧ s.2.1 > 0 ∧ s.2.2 > 0 ∧
       s.2.2 + s.2.1^2 + s.1^3 = s.1 * s.2.1 * s.2.2 ∧
       s.1 = Nat.gcd s.2.1 s.2.2}

theorem solution_set_characterization :
  SolutionSet = {(1, 2, 5), (1, 3, 5), (2, 2, 4), (2, 6, 4)} := by sorry

end NUMINAMATH_CALUDE_solution_set_characterization_l2023_202328


namespace NUMINAMATH_CALUDE_determinant_max_value_l2023_202339

open Real

theorem determinant_max_value : 
  let det := fun θ : ℝ => 
    Matrix.det !![1, 1, 1; 1, 1 + cos θ, 1; 1 + sin θ, 1, 1]
  ∃ (max_val : ℝ), max_val = 1/2 ∧ ∀ θ, det θ ≤ max_val :=
by sorry

end NUMINAMATH_CALUDE_determinant_max_value_l2023_202339


namespace NUMINAMATH_CALUDE_fraction_simplification_l2023_202349

theorem fraction_simplification (x : ℝ) : (3*x - 4)/4 + (5 - 2*x)/3 = (x + 8)/12 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l2023_202349


namespace NUMINAMATH_CALUDE_total_missed_pitches_l2023_202383

-- Define the constants from the problem
def pitches_per_token : ℕ := 15
def macy_tokens : ℕ := 11
def piper_tokens : ℕ := 17
def macy_hits : ℕ := 50
def piper_hits : ℕ := 55

-- Theorem statement
theorem total_missed_pitches :
  (macy_tokens * pitches_per_token + piper_tokens * pitches_per_token) - (macy_hits + piper_hits) = 315 := by
  sorry


end NUMINAMATH_CALUDE_total_missed_pitches_l2023_202383


namespace NUMINAMATH_CALUDE_min_cuts_for_100_pieces_l2023_202350

/-- Represents the number of pieces a cube is divided into after making cuts -/
def num_pieces (a b c : ℕ) : ℕ := (a + 1) * (b + 1) * (c + 1)

/-- Theorem stating that 11 is the minimum number of cuts needed to divide a cube into 100 pieces -/
theorem min_cuts_for_100_pieces :
  ∃ (a b c : ℕ), num_pieces a b c = 100 ∧ a + b + c = 11 ∧
  (∀ (x y z : ℕ), num_pieces x y z ≥ 100 → x + y + z ≥ 11) :=
sorry

end NUMINAMATH_CALUDE_min_cuts_for_100_pieces_l2023_202350


namespace NUMINAMATH_CALUDE_courtyard_width_l2023_202370

/-- The width of a rectangular courtyard given its length and paving stone requirements -/
theorem courtyard_width (length : Real) (num_stones : Nat) (stone_length stone_width : Real) 
  (h1 : length = 40)
  (h2 : num_stones = 132)
  (h3 : stone_length = 2.5)
  (h4 : stone_width = 2) :
  length * (num_stones * stone_length * stone_width / length) = 16.5 := by
  sorry

#check courtyard_width

end NUMINAMATH_CALUDE_courtyard_width_l2023_202370


namespace NUMINAMATH_CALUDE_perpendicular_bisector_and_parallel_line_l2023_202356

-- Define points A, B, and P
def A : ℝ × ℝ := (6, -6)
def B : ℝ × ℝ := (2, 2)
def P : ℝ × ℝ := (2, -3)

-- Define the perpendicular bisector equation
def perp_bisector (x y : ℝ) : Prop := x - 2*y - 8 = 0

-- Define the parallel line equation
def parallel_line (x y : ℝ) : Prop := 2*x + y - 1 = 0

-- Theorem statement
theorem perpendicular_bisector_and_parallel_line :
  -- Part 1: Perpendicular bisector
  (∀ x y, perp_bisector x y ↔ 
    -- Midpoint condition
    ((x - (A.1 + B.1)/2)^2 + (y - (A.2 + B.2)/2)^2 = 
     ((A.1 - B.1)/2)^2 + ((A.2 - B.2)/2)^2) ∧
    -- Perpendicularity condition
    ((y - A.2)*(B.1 - A.1) = -(x - A.1)*(B.2 - A.2))) ∧
  -- Part 2: Parallel line
  (∀ x y, parallel_line x y ↔
    -- Point P lies on the line
    (2*P.1 + P.2 - 1 = 0) ∧
    -- Parallel to AB
    ((y - P.2)/(x - P.1) = (B.2 - A.2)/(B.1 - A.1))) :=
sorry


end NUMINAMATH_CALUDE_perpendicular_bisector_and_parallel_line_l2023_202356


namespace NUMINAMATH_CALUDE_jamie_dives_for_pearls_l2023_202385

/-- Given that 25% of oysters have pearls, Jamie can collect 16 oysters per dive,
    and Jamie needs to collect 56 pearls, prove that Jamie needs to make 14 dives. -/
theorem jamie_dives_for_pearls (pearl_probability : ℚ) (oysters_per_dive : ℕ) (total_pearls : ℕ) :
  pearl_probability = 1/4 →
  oysters_per_dive = 16 →
  total_pearls = 56 →
  (total_pearls : ℚ) / (pearl_probability * oysters_per_dive) = 14 := by
  sorry

end NUMINAMATH_CALUDE_jamie_dives_for_pearls_l2023_202385


namespace NUMINAMATH_CALUDE_arthur_walked_four_point_five_miles_l2023_202313

/-- The distance Arthur walked in miles -/
def arthur_distance (blocks_west : ℕ) (blocks_south : ℕ) (miles_per_block : ℚ) : ℚ :=
  (blocks_west + blocks_south : ℚ) * miles_per_block

/-- Theorem stating that Arthur walked 4.5 miles -/
theorem arthur_walked_four_point_five_miles :
  arthur_distance 8 10 (1/4) = 4.5 := by
sorry

end NUMINAMATH_CALUDE_arthur_walked_four_point_five_miles_l2023_202313


namespace NUMINAMATH_CALUDE_wire_around_square_field_l2023_202388

theorem wire_around_square_field (area : ℝ) (wire_length : ℝ) : 
  area = 69696 → wire_length = 15840 → 
  (wire_length / (4 * Real.sqrt area)) = 15 := by
  sorry

end NUMINAMATH_CALUDE_wire_around_square_field_l2023_202388


namespace NUMINAMATH_CALUDE_number_sequence_properties_l2023_202368

/-- Represents the sequence formed by concatenating numbers from 1 to 999 -/
def NumberSequence : Type := List Nat

/-- Constructs the NumberSequence -/
def createSequence : NumberSequence := sorry

/-- Counts the total number of digits in the sequence -/
def countDigits (seq : NumberSequence) : Nat := sorry

/-- Counts the occurrences of a specific digit in the sequence -/
def countDigitOccurrences (seq : NumberSequence) (digit : Nat) : Nat := sorry

/-- Finds the digit at a specific position in the sequence -/
def digitAtPosition (seq : NumberSequence) (position : Nat) : Nat := sorry

theorem number_sequence_properties (seq : NumberSequence) :
  seq = createSequence →
  (countDigits seq = 2889) ∧
  (countDigitOccurrences seq 1 = 300) ∧
  (digitAtPosition seq 2016 = 8) := by
  sorry

end NUMINAMATH_CALUDE_number_sequence_properties_l2023_202368


namespace NUMINAMATH_CALUDE_camel_distribution_count_l2023_202302

def is_valid_camel_distribution (n : ℕ) : Prop :=
  n ≥ 1 ∧ n ≤ 99 ∧
  ∀ k : ℕ, k ≤ 62 → k + min (62 - k) (n - k) ≥ (100 + n) / 2

theorem camel_distribution_count :
  ∃ (S : Finset ℕ), (∀ n ∈ S, is_valid_camel_distribution n) ∧ S.card = 72 :=
sorry

end NUMINAMATH_CALUDE_camel_distribution_count_l2023_202302


namespace NUMINAMATH_CALUDE_probability_even_sum_is_5_11_l2023_202382

def ball_numbers : Finset ℕ := Finset.range 12

theorem probability_even_sum_is_5_11 :
  let total_outcomes := ball_numbers.card * (ball_numbers.card - 1)
  let favorable_outcomes := 
    (ball_numbers.filter (λ x => x % 2 = 0)).card * ((ball_numbers.filter (λ x => x % 2 = 0)).card - 1) +
    (ball_numbers.filter (λ x => x % 2 = 1)).card * ((ball_numbers.filter (λ x => x % 2 = 1)).card - 1)
  (favorable_outcomes : ℚ) / total_outcomes = 5 / 11 := by
  sorry

end NUMINAMATH_CALUDE_probability_even_sum_is_5_11_l2023_202382


namespace NUMINAMATH_CALUDE_sequence_sum_and_kth_term_l2023_202323

theorem sequence_sum_and_kth_term 
  (a : ℕ → ℤ) 
  (S : ℕ → ℤ) 
  (k : ℕ) 
  (h1 : ∀ n, S n = n^2 - 8*n) 
  (h2 : a k = 5) : 
  k = 7 := by sorry

end NUMINAMATH_CALUDE_sequence_sum_and_kth_term_l2023_202323


namespace NUMINAMATH_CALUDE_expression_evaluation_l2023_202311

theorem expression_evaluation : 5 + 15 / 3 - 2^2 * 4 = -6 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2023_202311


namespace NUMINAMATH_CALUDE_difference_of_sums_l2023_202384

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def is_multiple_of_three (n : ℕ) : Prop := ∃ k, n = 3 * k

def smallest_two_digit_multiple_of_three : ℕ := 12

def largest_two_digit_multiple_of_three : ℕ := 99

def smallest_two_digit_non_multiple_of_three : ℕ := 10

def largest_two_digit_non_multiple_of_three : ℕ := 98

theorem difference_of_sums : 
  (largest_two_digit_multiple_of_three + smallest_two_digit_multiple_of_three) -
  (largest_two_digit_non_multiple_of_three + smallest_two_digit_non_multiple_of_three) = 3 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_sums_l2023_202384


namespace NUMINAMATH_CALUDE_total_vehicles_is_282_l2023_202347

/-- The number of vehicles Kendra saw during her road trip -/
def total_vehicles : ℕ :=
  let morning_minivans := 20
  let morning_sedans := 17
  let morning_suvs := 12
  let morning_trucks := 8
  let morning_motorcycles := 5

  let afternoon_minivans := 22
  let afternoon_sedans := 13
  let afternoon_suvs := 15
  let afternoon_trucks := 10
  let afternoon_motorcycles := 7

  let evening_minivans := 15
  let evening_sedans := 19
  let evening_suvs := 18
  let evening_trucks := 14
  let evening_motorcycles := 10

  let night_minivans := 10
  let night_sedans := 12
  let night_suvs := 20
  let night_trucks := 20
  let night_motorcycles := 15

  let total_minivans := morning_minivans + afternoon_minivans + evening_minivans + night_minivans
  let total_sedans := morning_sedans + afternoon_sedans + evening_sedans + night_sedans
  let total_suvs := morning_suvs + afternoon_suvs + evening_suvs + night_suvs
  let total_trucks := morning_trucks + afternoon_trucks + evening_trucks + night_trucks
  let total_motorcycles := morning_motorcycles + afternoon_motorcycles + evening_motorcycles + night_motorcycles

  total_minivans + total_sedans + total_suvs + total_trucks + total_motorcycles

theorem total_vehicles_is_282 : total_vehicles = 282 := by
  sorry

end NUMINAMATH_CALUDE_total_vehicles_is_282_l2023_202347


namespace NUMINAMATH_CALUDE_smallest_next_divisor_after_221_l2023_202399

theorem smallest_next_divisor_after_221 (m : ℕ) (h1 : m ≥ 1000 ∧ m ≤ 9999) 
  (h2 : Even m) (h3 : m % 221 = 0) :
  ∃ (d : ℕ), d > 221 ∧ m % d = 0 ∧ d ≥ 238 ∧ 
  ∀ (d' : ℕ), d' > 221 ∧ m % d' = 0 → d' ≥ 238 :=
sorry

end NUMINAMATH_CALUDE_smallest_next_divisor_after_221_l2023_202399


namespace NUMINAMATH_CALUDE_leja_theorem_l2023_202345

/-- A set of points in a plane where any three points lie on a circle of radius r -/
def SpecialPointSet (P : Set (ℝ × ℝ)) (r : ℝ) : Prop :=
  ∀ p q s : ℝ × ℝ, p ∈ P → q ∈ P → s ∈ P → p ≠ q → q ≠ s → p ≠ s →
    ∃ c : ℝ × ℝ, dist c p = r ∧ dist c q = r ∧ dist c s = r

/-- Leja's theorem -/
theorem leja_theorem (P : Set (ℝ × ℝ)) (r : ℝ) (h : SpecialPointSet P r) :
  ∃ A : ℝ × ℝ, ∀ p ∈ P, dist A p ≤ r := by
  sorry

end NUMINAMATH_CALUDE_leja_theorem_l2023_202345


namespace NUMINAMATH_CALUDE_f_satisfies_equation_l2023_202303

noncomputable def f (x : ℝ) : ℝ := (x^3 - x^2 + 1) / (2*x*(1-x))

theorem f_satisfies_equation :
  ∀ x : ℝ, x ≠ 0 ∧ x ≠ 1 → f (1/x) + f (1-x) = x :=
by
  sorry

end NUMINAMATH_CALUDE_f_satisfies_equation_l2023_202303


namespace NUMINAMATH_CALUDE_f_composition_equals_14_l2023_202337

-- Define the functions f and g
def f (x : ℝ) : ℝ := 3 * x - 4
def g (x : ℝ) : ℝ := x + 2

-- State the theorem
theorem f_composition_equals_14 : f (1 + g 3) = 14 := by
  sorry

end NUMINAMATH_CALUDE_f_composition_equals_14_l2023_202337


namespace NUMINAMATH_CALUDE_election_votes_calculation_l2023_202363

theorem election_votes_calculation (total_votes : ℕ) : 
  (75 : ℚ) / 100 * ((100 : ℚ) - 15) / 100 * total_votes = 357000 → 
  total_votes = 560000 :=
by
  sorry

end NUMINAMATH_CALUDE_election_votes_calculation_l2023_202363


namespace NUMINAMATH_CALUDE_simplify_fraction_l2023_202359

theorem simplify_fraction (x : ℝ) (h1 : x ≠ 1) (h2 : x ≠ -1) :
  (3 / (x - 1)) + ((x - 3) / (1 - x^2)) = (2*x + 6) / (x^2 - 1) := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l2023_202359


namespace NUMINAMATH_CALUDE_jiyoon_sum_l2023_202315

theorem jiyoon_sum : 36 + 17 + 32 + 54 + 28 + 3 = 170 := by
  sorry

end NUMINAMATH_CALUDE_jiyoon_sum_l2023_202315


namespace NUMINAMATH_CALUDE_absolute_value_inequality_l2023_202375

theorem absolute_value_inequality (a b c : ℝ) (h : |a + b| < -c) :
  (∃! n : ℕ, n = 2 ∧
    (a < -b - c) ∧
    (a + b > c) ∧
    ¬(a + c < b) ∧
    ¬(|a| + c < b)) :=
by sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_l2023_202375


namespace NUMINAMATH_CALUDE_percentage_relation_l2023_202389

theorem percentage_relation (a b c : ℝ) 
  (h1 : c = 0.14 * a) 
  (h2 : c = 0.40 * b) : 
  b = 0.35 * a := by
sorry

end NUMINAMATH_CALUDE_percentage_relation_l2023_202389


namespace NUMINAMATH_CALUDE_fruit_cost_l2023_202335

/-- The cost of buying apples and oranges at given prices and quantities -/
theorem fruit_cost (apple_price : ℚ) (apple_weight : ℚ) (orange_price : ℚ) (orange_weight : ℚ)
  (apple_buy : ℚ) (orange_buy : ℚ) :
  apple_price = 3 →
  apple_weight = 4 →
  orange_price = 5 →
  orange_weight = 6 →
  apple_buy = 12 →
  orange_buy = 18 →
  (apple_price / apple_weight * apple_buy + orange_price / orange_weight * orange_buy : ℚ) = 24 :=
by sorry

end NUMINAMATH_CALUDE_fruit_cost_l2023_202335


namespace NUMINAMATH_CALUDE_amount_paid_is_fifty_l2023_202353

/-- Represents the purchase and change scenario --/
structure Purchase where
  book_cost : ℕ
  pen_cost : ℕ
  ruler_cost : ℕ
  change_received : ℕ

/-- Calculates the total cost of items --/
def total_cost (p : Purchase) : ℕ :=
  p.book_cost + p.pen_cost + p.ruler_cost

/-- Calculates the amount paid --/
def amount_paid (p : Purchase) : ℕ :=
  total_cost p + p.change_received

/-- Theorem stating that the amount paid is $50 --/
theorem amount_paid_is_fifty (p : Purchase) 
  (h1 : p.book_cost = 25)
  (h2 : p.pen_cost = 4)
  (h3 : p.ruler_cost = 1)
  (h4 : p.change_received = 20) :
  amount_paid p = 50 := by
  sorry

end NUMINAMATH_CALUDE_amount_paid_is_fifty_l2023_202353


namespace NUMINAMATH_CALUDE_rice_weight_proof_l2023_202376

/-- Given rice divided equally into 4 containers, each containing 50 ounces,
    prove that the total weight is 12.5 pounds, where 1 pound = 16 ounces. -/
theorem rice_weight_proof (containers : ℕ) (ounces_per_container : ℝ) 
    (ounces_per_pound : ℝ) : 
  containers = 4 →
  ounces_per_container = 50 →
  ounces_per_pound = 16 →
  (containers * ounces_per_container) / ounces_per_pound = 12.5 := by
  sorry

end NUMINAMATH_CALUDE_rice_weight_proof_l2023_202376


namespace NUMINAMATH_CALUDE_p_plus_q_equals_26_l2023_202316

theorem p_plus_q_equals_26 (P Q : ℝ) :
  (∀ x : ℝ, x ≠ 3 → P / (x - 3) + Q * (x + 2) = (-2 * x^2 + 8 * x + 34) / (x - 3)) →
  P + Q = 26 := by
sorry

end NUMINAMATH_CALUDE_p_plus_q_equals_26_l2023_202316


namespace NUMINAMATH_CALUDE_wire_length_proof_l2023_202358

theorem wire_length_proof (shorter_piece longer_piece total_length : ℝ) : 
  shorter_piece = 4 →
  shorter_piece = (2 / 5) * longer_piece →
  total_length = shorter_piece + longer_piece →
  total_length = 14 := by
sorry

end NUMINAMATH_CALUDE_wire_length_proof_l2023_202358


namespace NUMINAMATH_CALUDE_sanchez_problem_l2023_202390

theorem sanchez_problem (x y : ℕ+) : x - y = 3 → x * y = 56 → x + y = 17 := by sorry

end NUMINAMATH_CALUDE_sanchez_problem_l2023_202390


namespace NUMINAMATH_CALUDE_other_workers_count_l2023_202377

def total_workers : ℕ := 5
def chosen_workers : ℕ := 2
def probability_jack_and_jill : ℚ := 1/10

theorem other_workers_count :
  let other_workers := total_workers - 2
  probability_jack_and_jill = 1 / (total_workers.choose chosen_workers) →
  other_workers = 3 := by
sorry

end NUMINAMATH_CALUDE_other_workers_count_l2023_202377


namespace NUMINAMATH_CALUDE_right_triangle_angle_split_l2023_202343

theorem right_triangle_angle_split (BC AC : ℝ) (h_right : BC = 5 ∧ AC = 12) :
  let AB := Real.sqrt (BC^2 + AC^2)
  let angle_ratio := (1 : ℝ) / 3
  let smaller_segment := AB * (Real.sqrt 3 / 2)
  smaller_segment = 13 * Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_angle_split_l2023_202343


namespace NUMINAMATH_CALUDE_power_function_solution_l2023_202346

-- Define a power function type
def PowerFunction := ℝ → ℝ

-- Define the properties of our specific power function
def isPowerFunctionThroughPoint (f : PowerFunction) : Prop :=
  ∃ α : ℝ, (∀ x : ℝ, f x = x ^ α) ∧ f (-2) = -1/8

-- State the theorem
theorem power_function_solution 
  (f : PowerFunction) 
  (h : isPowerFunctionThroughPoint f) : 
  ∃ x : ℝ, f x = 27 ∧ x = 1/3 := by
sorry

end NUMINAMATH_CALUDE_power_function_solution_l2023_202346


namespace NUMINAMATH_CALUDE_negation_of_exists_is_forall_l2023_202341

theorem negation_of_exists_is_forall (f : ℝ → ℝ) :
  (¬ ∃ x : ℝ, f x < 0) ↔ (∀ x : ℝ, f x ≥ 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_exists_is_forall_l2023_202341


namespace NUMINAMATH_CALUDE_jane_egg_money_l2023_202392

/-- Calculates the money made from selling eggs over a period of weeks. -/
def money_from_eggs (num_chickens : ℕ) (eggs_per_chicken : ℕ) (price_per_dozen : ℚ) (num_weeks : ℕ) : ℚ :=
  (num_chickens * eggs_per_chicken * num_weeks : ℚ) / 12 * price_per_dozen

/-- Proves that Jane makes $20 in 2 weeks from selling eggs. -/
theorem jane_egg_money :
  money_from_eggs 10 6 2 2 = 20 := by
  sorry

end NUMINAMATH_CALUDE_jane_egg_money_l2023_202392


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l2023_202330

theorem sum_of_coefficients (a a₁ a₂ a₃ a₄ a₅ : ℝ) : 
  (∀ x y : ℝ, (x - 2*y)^5 = a*x^5 + a₁*x^4*y + a₂*x^3*y^2 + a₃*x^2*y^3 + a₄*x*y^4 + a₅*y^5) →
  a + a₁ + a₂ + a₃ + a₄ + a₅ = -1 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l2023_202330


namespace NUMINAMATH_CALUDE_not_necessary_nor_sufficient_condition_l2023_202354

theorem not_necessary_nor_sufficient_condition (a : ℝ) :
  ¬(∀ x : ℝ, a * x^2 + a * x - 1 < 0 ↔ a < 0) :=
sorry

end NUMINAMATH_CALUDE_not_necessary_nor_sufficient_condition_l2023_202354


namespace NUMINAMATH_CALUDE_stick_marking_underdetermined_l2023_202352

/-- Represents the length of a portion of the stick -/
structure Portion where
  length : ℚ
  isValid : 0 < length ∧ length ≤ 1

/-- Represents the configuration of markings on the stick -/
structure StickMarkings where
  fifthPortions : ℕ
  xPortions : ℕ
  xLength : ℚ
  totalLength : ℚ
  validTotal : fifthPortions + xPortions = 8
  validLength : fifthPortions * (1/5) + xPortions * xLength = totalLength

/-- Theorem stating that the problem is underdetermined -/
theorem stick_marking_underdetermined :
  ∀ (m : StickMarkings),
    m.totalLength = 1 →
    ∃ (m' : StickMarkings),
      m'.totalLength = 1 ∧
      m'.fifthPortions ≠ m.fifthPortions ∧
      m'.xLength ≠ m.xLength :=
sorry

end NUMINAMATH_CALUDE_stick_marking_underdetermined_l2023_202352


namespace NUMINAMATH_CALUDE_storks_birds_difference_l2023_202318

theorem storks_birds_difference : 
  let initial_birds : ℕ := 2
  let initial_storks : ℕ := 6
  let additional_birds : ℕ := 3
  let final_birds : ℕ := initial_birds + additional_birds
  initial_storks - final_birds = 1 := by sorry

end NUMINAMATH_CALUDE_storks_birds_difference_l2023_202318


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l2023_202357

def A : Set ℤ := {x : ℤ | |x| < 3}
def B : Set ℤ := {x : ℤ | |x| > 1}

theorem intersection_of_A_and_B : A ∩ B = {-2, 2} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l2023_202357


namespace NUMINAMATH_CALUDE_at_least_one_survives_one_of_each_type_survives_l2023_202372

-- Define the survival probabilities
def survival_rate_A : ℚ := 5/6
def survival_rate_B : ℚ := 4/5

-- Define the number of trees of each type
def num_trees_A : ℕ := 2
def num_trees_B : ℕ := 2

-- Define the total number of trees
def total_trees : ℕ := num_trees_A + num_trees_B

-- Theorem for the probability that at least one tree survives
theorem at_least_one_survives :
  1 - (1 - survival_rate_A)^num_trees_A * (1 - survival_rate_B)^num_trees_B = 899/900 := by
  sorry

-- Theorem for the probability that one tree of each type survives
theorem one_of_each_type_survives :
  num_trees_A * survival_rate_A * (1 - survival_rate_A) *
  num_trees_B * survival_rate_B * (1 - survival_rate_B) = 4/45 := by
  sorry

end NUMINAMATH_CALUDE_at_least_one_survives_one_of_each_type_survives_l2023_202372


namespace NUMINAMATH_CALUDE_coffee_package_size_l2023_202305

theorem coffee_package_size (total_coffee : ℕ) (large_package_size : ℕ) (large_package_count : ℕ) (small_package_count_diff : ℕ) :
  total_coffee = 55 →
  large_package_size = 10 →
  large_package_count = 3 →
  small_package_count_diff = 2 →
  ∃ (small_package_size : ℕ),
    small_package_size * (large_package_count + small_package_count_diff) +
    large_package_size * large_package_count = total_coffee ∧
    small_package_size = 5 :=
by sorry

end NUMINAMATH_CALUDE_coffee_package_size_l2023_202305


namespace NUMINAMATH_CALUDE_vector_problem_l2023_202342

/-- Given three vectors a, b, c in ℝ² -/
def a : ℝ × ℝ := (1, 2)
def b : ℝ × ℝ := (-2, 3)
def c : ℝ → ℝ × ℝ := λ m ↦ (-2, m)

/-- Dot product of two 2D vectors -/
def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

/-- Two vectors are perpendicular if their dot product is zero -/
def perpendicular (v w : ℝ × ℝ) : Prop := dot_product v w = 0

/-- Two vectors are collinear if one is a scalar multiple of the other -/
def collinear (v w : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, v.1 = k * w.1 ∧ v.2 = k * w.2

/-- Main theorem combining both parts of the problem -/
theorem vector_problem :
  (∃ m : ℝ, perpendicular a (b + c m) → m = -1) ∧
  (∃ k : ℝ, collinear (k • a + b) (2 • a - b) → k = -2) := by
  sorry

end NUMINAMATH_CALUDE_vector_problem_l2023_202342


namespace NUMINAMATH_CALUDE_bisection_method_accuracy_l2023_202334

theorem bisection_method_accuracy (initial_interval_width : ℝ) (desired_accuracy : ℝ) : 
  initial_interval_width = 2 →
  desired_accuracy = 0.1 →
  ∃ n : ℕ, (n ≥ 5 ∧ initial_interval_width / (2^n : ℝ) < desired_accuracy) ∧
           ∀ m : ℕ, m < 5 → initial_interval_width / (2^m : ℝ) ≥ desired_accuracy :=
by sorry

end NUMINAMATH_CALUDE_bisection_method_accuracy_l2023_202334


namespace NUMINAMATH_CALUDE_chef_cooked_ten_wings_l2023_202362

/-- The number of additional chicken wings cooked by the chef for a group of friends -/
def additional_wings (num_friends : ℕ) (pre_cooked_wings : ℕ) (wings_per_person : ℕ) : ℕ :=
  num_friends * wings_per_person - pre_cooked_wings

/-- Theorem: Given 3 friends, 8 pre-cooked wings, and 6 wings per person, 
    the number of additional wings cooked is 10 -/
theorem chef_cooked_ten_wings : additional_wings 3 8 6 = 10 := by
  sorry

end NUMINAMATH_CALUDE_chef_cooked_ten_wings_l2023_202362


namespace NUMINAMATH_CALUDE_relationship_abc_l2023_202320

theorem relationship_abc (a b c : ℝ) 
  (ha : a = 2^(1/5))
  (hb : b = 2^(3/10))
  (hc : c = Real.log 2 / Real.log 3) :
  c < a ∧ a < b :=
sorry

end NUMINAMATH_CALUDE_relationship_abc_l2023_202320


namespace NUMINAMATH_CALUDE_cube_sum_equals_four_l2023_202398

theorem cube_sum_equals_four (x y : ℝ) 
  (h1 : x + y = 1) 
  (h2 : x^2 + y^2 = 3) : 
  x^3 + y^3 = 4 := by
sorry

end NUMINAMATH_CALUDE_cube_sum_equals_four_l2023_202398


namespace NUMINAMATH_CALUDE_second_point_x_coordinate_l2023_202321

/-- Given two points on a line, prove the x-coordinate of the second point -/
theorem second_point_x_coordinate 
  (m n : ℝ) 
  (h1 : m = 2 * n + 5) 
  (h2 : m + 1 = 2 * (n + 0.5) + 5) : 
  m + 1 = 2 * n + 6 := by
  sorry

end NUMINAMATH_CALUDE_second_point_x_coordinate_l2023_202321


namespace NUMINAMATH_CALUDE_min_value_inequality_l2023_202309

theorem min_value_inequality (x y : ℝ) (h1 : x > -1) (h2 : y > 0) (h3 : x + 2*y = 1) :
  1 / (x + 1) + 1 / y ≥ (3 + 2 * Real.sqrt 2) / 2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_inequality_l2023_202309


namespace NUMINAMATH_CALUDE_jessica_purchases_total_cost_l2023_202351

/-- The cost of Jessica's cat toy in dollars -/
def cat_toy_cost : ℚ := 10.22

/-- The cost of Jessica's cage in dollars -/
def cage_cost : ℚ := 11.73

/-- The total cost of Jessica's purchases in dollars -/
def total_cost : ℚ := cat_toy_cost + cage_cost

theorem jessica_purchases_total_cost :
  total_cost = 21.95 := by sorry

end NUMINAMATH_CALUDE_jessica_purchases_total_cost_l2023_202351


namespace NUMINAMATH_CALUDE_rat_count_proof_l2023_202307

def total_rats (kenia_rats : ℕ) (hunter_rats : ℕ) (elodie_rats : ℕ) : ℕ :=
  kenia_rats + hunter_rats + elodie_rats

theorem rat_count_proof (kenia_rats : ℕ) (hunter_rats : ℕ) (elodie_rats : ℕ) :
  kenia_rats = 3 * (hunter_rats + elodie_rats) →
  elodie_rats = 30 →
  elodie_rats = hunter_rats + 10 →
  total_rats kenia_rats hunter_rats elodie_rats = 200 :=
by
  sorry

end NUMINAMATH_CALUDE_rat_count_proof_l2023_202307


namespace NUMINAMATH_CALUDE_abs_inequality_solution_set_l2023_202374

theorem abs_inequality_solution_set (x : ℝ) : 
  |3*x + 1| - |x - 1| < 0 ↔ -1 < x ∧ x < 1 := by sorry

end NUMINAMATH_CALUDE_abs_inequality_solution_set_l2023_202374


namespace NUMINAMATH_CALUDE_problem_statement_l2023_202364

def f (x : ℝ) := |2*x - 1|

theorem problem_statement (a b c : ℝ) 
  (h1 : a < b) (h2 : b < c) 
  (h3 : f a > f c) (h4 : f c > f b) : 
  2 - a < 2*c := by
sorry

end NUMINAMATH_CALUDE_problem_statement_l2023_202364


namespace NUMINAMATH_CALUDE_unique_root_in_interval_l2023_202355

theorem unique_root_in_interval (f : ℝ → ℝ) (m n : ℝ) :
  (∀ x, f x = -x^3 - x) →
  m ≤ n →
  f m * f n < 0 →
  ∃! x, m ≤ x ∧ x ≤ n ∧ f x = 0 :=
by sorry

end NUMINAMATH_CALUDE_unique_root_in_interval_l2023_202355


namespace NUMINAMATH_CALUDE_caps_collection_total_l2023_202387

theorem caps_collection_total (A B C : ℕ) : 
  A = (B + C) / 2 →
  B = (A + C) / 3 →
  C = 150 →
  A + B + C = 360 := by
sorry

end NUMINAMATH_CALUDE_caps_collection_total_l2023_202387


namespace NUMINAMATH_CALUDE_inequality_theorem_l2023_202386

/-- A function f: ℝ → ℝ satisfying the given condition -/
def SatisfiesCondition (f : ℝ → ℝ) : Prop :=
  ∀ x, Differentiable ℝ f ∧ (x - 1) * (deriv (deriv f) x) < 0

/-- Theorem stating the inequality for functions satisfying the condition -/
theorem inequality_theorem (f : ℝ → ℝ) (h : SatisfiesCondition f) :
  f 0 + f 2 < 2 * f 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_theorem_l2023_202386


namespace NUMINAMATH_CALUDE_second_die_has_seven_sides_l2023_202314

/-- The number of sides on the first die -/
def first_die_sides : ℕ := 6

/-- The probability of rolling a sum of 13 with both dice -/
def prob_sum_13 : ℚ := 23809523809523808 / 1000000000000000000

/-- The number of sides on the second die -/
def second_die_sides : ℕ := sorry

theorem second_die_has_seven_sides :
  (1 : ℚ) / (first_die_sides * second_die_sides) = prob_sum_13 ∧ 
  second_die_sides ≥ 7 →
  second_die_sides = 7 := by sorry

end NUMINAMATH_CALUDE_second_die_has_seven_sides_l2023_202314


namespace NUMINAMATH_CALUDE_contradiction_assumption_l2023_202367

theorem contradiction_assumption (x y z : ℝ) :
  (¬ (x > 0 ∨ y > 0 ∨ z > 0)) ↔ (x ≤ 0 ∧ y ≤ 0 ∧ z ≤ 0) := by
  sorry

end NUMINAMATH_CALUDE_contradiction_assumption_l2023_202367


namespace NUMINAMATH_CALUDE_train_distance_problem_l2023_202348

theorem train_distance_problem (v1 v2 d : ℝ) (h1 : v1 = 16) (h2 : v2 = 21) (h3 : d = 60) :
  let t := d / (v2 - v1)
  let d1 := v1 * t
  let d2 := v2 * t
  d1 + d2 = 444 := by sorry

end NUMINAMATH_CALUDE_train_distance_problem_l2023_202348


namespace NUMINAMATH_CALUDE_f_increasing_f_sum_zero_l2023_202300

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a - 2 / (2^x + 1)

theorem f_increasing (a : ℝ) : 
  ∀ x₁ x₂ : ℝ, x₁ < x₂ → f a x₁ < f a x₂ := by sorry

theorem f_sum_zero : 
  f 1 (-5) + f 1 (-3) + f 1 (-1) + f 1 1 + f 1 3 + f 1 5 = 0 := by sorry

end NUMINAMATH_CALUDE_f_increasing_f_sum_zero_l2023_202300


namespace NUMINAMATH_CALUDE_lesser_fraction_l2023_202322

theorem lesser_fraction (x y : ℝ) (h_sum : x + y = 10/11) (h_prod : x * y = 1/8) :
  min x y = (80 - 2 * Real.sqrt 632) / 176 := by sorry

end NUMINAMATH_CALUDE_lesser_fraction_l2023_202322


namespace NUMINAMATH_CALUDE_machine_worked_three_minutes_l2023_202366

/-- An industrial machine that makes shirts -/
structure ShirtMachine where
  shirts_per_minute : ℕ
  shirts_made_yesterday : ℕ

/-- The number of minutes the machine worked yesterday -/
def minutes_worked_yesterday (machine : ShirtMachine) : ℕ :=
  machine.shirts_made_yesterday / machine.shirts_per_minute

/-- Theorem stating that the machine worked for 3 minutes yesterday -/
theorem machine_worked_three_minutes (machine : ShirtMachine) 
    (h1 : machine.shirts_per_minute = 3)
    (h2 : machine.shirts_made_yesterday = 9) : 
  minutes_worked_yesterday machine = 3 := by
  sorry

end NUMINAMATH_CALUDE_machine_worked_three_minutes_l2023_202366
