import Mathlib

namespace NUMINAMATH_CALUDE_specific_rhombus_area_l926_92618

/-- Represents a rhombus with given properties -/
structure Rhombus where
  side_length : ℝ
  diagonal_difference : ℝ
  diagonals_perpendicular_bisectors : Bool

/-- Calculates the area of a rhombus with the given properties -/
def rhombus_area (r : Rhombus) : ℝ :=
  sorry

/-- Theorem stating the area of a specific rhombus -/
theorem specific_rhombus_area :
  let r : Rhombus := {
    side_length := Real.sqrt 165,
    diagonal_difference := 10,
    diagonals_perpendicular_bisectors := true
  }
  rhombus_area r = 305 / 4 := by sorry

end NUMINAMATH_CALUDE_specific_rhombus_area_l926_92618


namespace NUMINAMATH_CALUDE_total_earnings_is_228_l926_92680

/-- Calculates Zainab's total earnings for 4 weeks of passing out flyers -/
def total_earnings : ℝ :=
  let monday_hours : ℝ := 3
  let monday_rate : ℝ := 2.5
  let wednesday_hours : ℝ := 4
  let wednesday_rate : ℝ := 3
  let saturday_hours : ℝ := 5
  let saturday_rate : ℝ := 3.5
  let saturday_flyers : ℝ := 200
  let flyer_commission : ℝ := 0.1
  let weeks : ℝ := 4

  let monday_earnings := monday_hours * monday_rate
  let wednesday_earnings := wednesday_hours * wednesday_rate
  let saturday_hourly_earnings := saturday_hours * saturday_rate
  let saturday_commission := saturday_flyers * flyer_commission
  let saturday_total_earnings := saturday_hourly_earnings + saturday_commission
  let weekly_earnings := monday_earnings + wednesday_earnings + saturday_total_earnings

  weeks * weekly_earnings

theorem total_earnings_is_228 : total_earnings = 228 := by
  sorry

end NUMINAMATH_CALUDE_total_earnings_is_228_l926_92680


namespace NUMINAMATH_CALUDE_abs_diff_and_opposite_l926_92608

theorem abs_diff_and_opposite (a b : ℝ) (h : a < b) : 
  |((a - b) - (b - a))| = 2*b - 2*a := by sorry

end NUMINAMATH_CALUDE_abs_diff_and_opposite_l926_92608


namespace NUMINAMATH_CALUDE_hyperbola_foci_l926_92675

/-- The hyperbola equation -/
def hyperbola_equation (x y : ℝ) : Prop :=
  y^2 / 3 - x^2 = 1

/-- The coordinates of a focus -/
def is_focus (x y : ℝ) : Prop :=
  x = 0 ∧ (y = 2 ∨ y = -2)

/-- Theorem: The foci of the given hyperbola are at (0, ±2) -/
theorem hyperbola_foci :
  ∀ x y : ℝ, hyperbola_equation x y →
  (∃ a b : ℝ, is_focus a b ∧ 
    (x - a)^2 + (y - b)^2 = (x + a)^2 + (y + b)^2) :=
sorry

end NUMINAMATH_CALUDE_hyperbola_foci_l926_92675


namespace NUMINAMATH_CALUDE_square_minus_twelve_plus_fiftyfour_l926_92606

theorem square_minus_twelve_plus_fiftyfour (a b : ℝ) 
  (h1 : 0 ≤ a) (h2 : 0 ≤ b) 
  (h3 : a^2 + b^2 = 74) (h4 : a * b = 35) : 
  a^2 - 12 * a + 54 = 19 := by
sorry

end NUMINAMATH_CALUDE_square_minus_twelve_plus_fiftyfour_l926_92606


namespace NUMINAMATH_CALUDE_binary_10111_equals_43_base_5_l926_92634

/-- Converts a binary number represented as a list of bits to its decimal equivalent -/
def binary_to_decimal (bits : List Bool) : ℕ :=
  bits.enum.foldl (fun acc (i, b) => acc + if b then 2^i else 0) 0

/-- Converts a natural number to its base-5 representation -/
def to_base_5 (n : ℕ) : List ℕ :=
  if n = 0 then [0] else
    let rec aux (m : ℕ) (acc : List ℕ) :=
      if m = 0 then acc
      else aux (m / 5) ((m % 5) :: acc)
    aux n []

/-- The binary representation of 10111 -/
def binary_10111 : List Bool := [true, true, true, false, true]

theorem binary_10111_equals_43_base_5 :
  to_base_5 (binary_to_decimal binary_10111) = [4, 3] :=
sorry

end NUMINAMATH_CALUDE_binary_10111_equals_43_base_5_l926_92634


namespace NUMINAMATH_CALUDE_normal_probability_theorem_l926_92660

/-- The standard normal cumulative distribution function -/
noncomputable def Φ : ℝ → ℝ := sorry

/-- Normal distribution probability density function -/
def normal_pdf (μ σ : ℝ) (x : ℝ) : ℝ := sorry

theorem normal_probability_theorem (ξ : ℝ → ℝ) (μ σ : ℝ) 
  (h_normal : ∀ x, normal_pdf μ σ x = sorry)  -- ξ follows N(μ, σ²)
  (h_mean : ∫ x, x * normal_pdf μ σ x = 3)    -- E[ξ] = 3
  (h_var : ∫ x, (x - μ)^2 * normal_pdf μ σ x = 1)  -- D[ξ] = 1
  : ∫ x in Set.Ioo (-1) 1, normal_pdf μ σ x = Φ (-4) - Φ (-2) :=
sorry

end NUMINAMATH_CALUDE_normal_probability_theorem_l926_92660


namespace NUMINAMATH_CALUDE_two_sin_plus_three_cos_l926_92672

theorem two_sin_plus_three_cos (x : ℝ) : 
  2 * Real.cos x - 3 * Real.sin x = 4 → 
  (2 * Real.sin x + 3 * Real.cos x = 3) ∨ (2 * Real.sin x + 3 * Real.cos x = 1) := by
sorry

end NUMINAMATH_CALUDE_two_sin_plus_three_cos_l926_92672


namespace NUMINAMATH_CALUDE_angle_C_measure_l926_92625

-- Define the triangle ABC
variable (A B C : ℝ)

-- Define the conditions
axiom scalene : A ≠ B ∧ B ≠ C ∧ A ≠ C
axiom angle_sum : A + B + C = 180
axiom angle_relation : C = A + 40
axiom angle_B : B = 2 * A

-- Theorem to prove
theorem angle_C_measure : C = 75 := by
  sorry

end NUMINAMATH_CALUDE_angle_C_measure_l926_92625


namespace NUMINAMATH_CALUDE_A_power_50_l926_92650

def A : Matrix (Fin 2) (Fin 2) ℤ := !![5, 1; -16, -3]

theorem A_power_50 : A^50 = !![201, 50; -800, -199] := by sorry

end NUMINAMATH_CALUDE_A_power_50_l926_92650


namespace NUMINAMATH_CALUDE_coin_circumference_diameter_ratio_l926_92688

theorem coin_circumference_diameter_ratio :
  let diameter : ℝ := 100
  let circumference : ℝ := 314
  circumference / diameter = 3.14 := by sorry

end NUMINAMATH_CALUDE_coin_circumference_diameter_ratio_l926_92688


namespace NUMINAMATH_CALUDE_total_car_production_l926_92699

theorem total_car_production (north_america : ℕ) (europe : ℕ) 
  (h1 : north_america = 3884) (h2 : europe = 2871) : 
  north_america + europe = 6755 := by
  sorry

end NUMINAMATH_CALUDE_total_car_production_l926_92699


namespace NUMINAMATH_CALUDE_geometric_series_first_term_l926_92657

theorem geometric_series_first_term (a r : ℝ) (h1 : r ≠ 1) (h2 : |r| < 1) : 
  (a / (1 - r) = 20) → 
  (a^2 / (1 - r^2) = 80) → 
  a = 20/3 := by sorry

end NUMINAMATH_CALUDE_geometric_series_first_term_l926_92657


namespace NUMINAMATH_CALUDE_certain_number_problem_l926_92656

theorem certain_number_problem (N : ℚ) : 
  (5 / 6 : ℚ) * N = (5 / 16 : ℚ) * N + 200 → N = 384 := by
sorry

end NUMINAMATH_CALUDE_certain_number_problem_l926_92656


namespace NUMINAMATH_CALUDE_matrix_equation_l926_92643

def A : Matrix (Fin 2) (Fin 2) ℚ := !![2, 1; 1, 3]
def B : Matrix (Fin 2) (Fin 2) ℚ := !![1, 0; 1, -1]
def C : Matrix (Fin 2) (Fin 2) ℚ := !![3/5, 4/5; -1/5, -3/5]

theorem matrix_equation : A * C = B := by sorry

end NUMINAMATH_CALUDE_matrix_equation_l926_92643


namespace NUMINAMATH_CALUDE_line_intercepts_l926_92642

/-- Given a line with equation 5x + 3y - 15 = 0, prove that its x-intercept is 3 and y-intercept is 5 -/
theorem line_intercepts :
  let f : ℝ → ℝ := λ x => -(5/3) * x + 5
  (f 0 = 5) ∧ (f⁻¹ 0 = 3) := by sorry

end NUMINAMATH_CALUDE_line_intercepts_l926_92642


namespace NUMINAMATH_CALUDE_circle_chord_theorem_l926_92640

def circle_chord_problem (r AB SX : ℝ) : Prop :=
  let S := (0 : ℝ × ℝ)  -- Center of the circle
  let k := {p : ℝ × ℝ | (p.1 - S.1)^2 + (p.2 - S.2)^2 = r^2}  -- Circle
  ∃ (A B C D X : ℝ × ℝ),
    A ∈ k ∧ B ∈ k ∧ C ∈ k ∧ D ∈ k ∧  -- Points on the circle
    (A.1 - B.1) * (C.1 - D.1) + (A.2 - B.2) * (C.2 - D.2) = 0 ∧  -- Perpendicular chords
    (X.1 - S.1)^2 + (X.2 - S.2)^2 = SX^2 ∧  -- X is SX distance from center
    (A.1 - B.1)^2 + (A.2 - B.2)^2 = AB^2 ∧  -- Length of AB
    abs ((C.1 - D.1)^2 + (C.2 - D.2)^2 - 10000) < 1  -- Length of CD ≈ 100 mm

theorem circle_chord_theorem :
  circle_chord_problem 52 96 25 := by
  sorry

end NUMINAMATH_CALUDE_circle_chord_theorem_l926_92640


namespace NUMINAMATH_CALUDE_ways_to_walk_teaching_building_l926_92696

/-- Represents a building with a given number of floors and staircases per floor -/
structure Building where
  floors : Nat
  staircases_per_floor : Nat

/-- Calculates the number of ways to walk from the first floor to the top floor -/
def ways_to_walk (b : Building) : Nat :=
  b.staircases_per_floor ^ (b.floors - 1)

/-- The teaching building -/
def teaching_building : Building :=
  { floors := 4, staircases_per_floor := 2 }

theorem ways_to_walk_teaching_building :
  ways_to_walk teaching_building = 2^3 := by
  sorry

#eval ways_to_walk teaching_building

end NUMINAMATH_CALUDE_ways_to_walk_teaching_building_l926_92696


namespace NUMINAMATH_CALUDE_complex_simplification_l926_92698

theorem complex_simplification :
  (5 - 3*Complex.I) + (-2 + 6*Complex.I) - (7 - 2*Complex.I) = -4 + 5*Complex.I :=
by sorry

end NUMINAMATH_CALUDE_complex_simplification_l926_92698


namespace NUMINAMATH_CALUDE_unique_prime_factorization_l926_92690

theorem unique_prime_factorization : 
  ∃! (d e f : ℕ), 
    d.Prime ∧ e.Prime ∧ f.Prime ∧ 
    d ≠ e ∧ d ≠ f ∧ e ≠ f ∧
    d * e * f = 7902 ∧
    d + e + f = 1322 :=
by sorry

end NUMINAMATH_CALUDE_unique_prime_factorization_l926_92690


namespace NUMINAMATH_CALUDE_special_square_midpoint_sum_l926_92654

/-- A square in the first quadrant with specific points on its sides -/
structure SpecialSquare where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ
  in_first_quadrant : A.1 ≥ 0 ∧ A.2 ≥ 0 ∧ B.1 ≥ 0 ∧ B.2 ≥ 0 ∧ C.1 ≥ 0 ∧ C.2 ≥ 0 ∧ D.1 ≥ 0 ∧ D.2 ≥ 0
  is_square : (A.1 - B.1)^2 + (A.2 - B.2)^2 = (B.1 - C.1)^2 + (B.2 - C.2)^2 ∧
              (B.1 - C.1)^2 + (B.2 - C.2)^2 = (C.1 - D.1)^2 + (C.2 - D.2)^2 ∧
              (C.1 - D.1)^2 + (C.2 - D.2)^2 = (D.1 - A.1)^2 + (D.2 - A.2)^2
  point_on_AD : ∃ (t : ℝ), 0 ≤ t ∧ t ≤ 1 ∧ (2, 0) = (t * A.1 + (1 - t) * D.1, t * A.2 + (1 - t) * D.2)
  point_on_BC : ∃ (t : ℝ), 0 ≤ t ∧ t ≤ 1 ∧ (6, 0) = (t * B.1 + (1 - t) * C.1, t * B.2 + (1 - t) * C.2)
  point_on_AB : ∃ (t : ℝ), 0 ≤ t ∧ t ≤ 1 ∧ (10, 0) = (t * A.1 + (1 - t) * B.1, t * A.2 + (1 - t) * B.2)
  point_on_CD : ∃ (t : ℝ), 0 ≤ t ∧ t ≤ 1 ∧ (14, 0) = (t * C.1 + (1 - t) * D.1, t * C.2 + (1 - t) * D.2)

/-- The sum of coordinates of the midpoint of the special square is 10 -/
theorem special_square_midpoint_sum (sq : SpecialSquare) :
  (sq.A.1 + sq.C.1) / 2 + (sq.A.2 + sq.C.2) / 2 = 10 := by
  sorry

end NUMINAMATH_CALUDE_special_square_midpoint_sum_l926_92654


namespace NUMINAMATH_CALUDE_synesthesia_demonstrates_mutual_influence_and_restriction_l926_92679

/-- Represents a sensory perception -/
inductive Sense
  | Sight
  | Hearing
  | Taste
  | Smell
  | Touch

/-- Represents the phenomenon of synesthesia -/
def Synesthesia := Set (Sense × Sense)

/-- Represents the property of mutual influence and restriction -/
def MutualInfluenceAndRestriction (s : Synesthesia) : Prop := sorry

/-- Represents a thing and its internal elements -/
structure Thing where
  elements : Set Sense

theorem synesthesia_demonstrates_mutual_influence_and_restriction 
  (s : Synesthesia) 
  (h : s.Nonempty) : 
  MutualInfluenceAndRestriction s := by
  sorry

#check synesthesia_demonstrates_mutual_influence_and_restriction

end NUMINAMATH_CALUDE_synesthesia_demonstrates_mutual_influence_and_restriction_l926_92679


namespace NUMINAMATH_CALUDE_pace_ratio_l926_92638

/-- The ratio of a man's pace on a day he was late to his usual pace -/
theorem pace_ratio (usual_time : ℝ) (late_time : ℝ) (h1 : usual_time = 2) 
  (h2 : late_time = usual_time + 1/3) : 
  (usual_time / late_time) = 6/7 := by
  sorry

end NUMINAMATH_CALUDE_pace_ratio_l926_92638


namespace NUMINAMATH_CALUDE_cubic_root_sum_l926_92678

-- Define the cubic polynomial
def cubic_poly (a b c d : ℝ) (x : ℝ) : ℝ := a * x^3 + b * x^2 + c * x + d

-- State the theorem
theorem cubic_root_sum (a b c d : ℝ) : 
  a ≠ 0 → 
  cubic_poly a b c d 4 = 0 →
  cubic_poly a b c d (-3) = 0 →
  (b + c) / a = -13 := by
sorry

end NUMINAMATH_CALUDE_cubic_root_sum_l926_92678


namespace NUMINAMATH_CALUDE_total_distance_run_l926_92630

/-- The circumference of the circular track in meters -/
def track_length : ℝ := 50

/-- The number of pairs of children (boy-girl pairs) -/
def num_pairs : ℕ := 4

/-- Theorem: The total distance run by all children is 100 meters -/
theorem total_distance_run : 
  track_length * (num_pairs : ℝ) / 2 = 100 := by sorry

end NUMINAMATH_CALUDE_total_distance_run_l926_92630


namespace NUMINAMATH_CALUDE_series_sum_proof_l926_92610

theorem series_sum_proof : ∑' k, (k : ℝ) / (4 ^ k) = 4 / 9 := by sorry

end NUMINAMATH_CALUDE_series_sum_proof_l926_92610


namespace NUMINAMATH_CALUDE_total_water_consumption_in_week_l926_92636

/-- Represents the water consumption of a sibling -/
structure WaterConsumption where
  weekday : ℕ
  weekend : ℕ

/-- Calculates the total water consumption for a sibling in a week -/
def weeklyConsumption (wc : WaterConsumption) : ℕ :=
  wc.weekday * 5 + wc.weekend * 2

/-- Theorem: Total water consumption of siblings in a week -/
theorem total_water_consumption_in_week (theo mason roxy zara lily : WaterConsumption)
  (h_theo : theo = { weekday := 8, weekend := 10 })
  (h_mason : mason = { weekday := 7, weekend := 8 })
  (h_roxy : roxy = { weekday := 9, weekend := 11 })
  (h_zara : zara = { weekday := 10, weekend := 12 })
  (h_lily : lily = { weekday := 6, weekend := 7 }) :
  weeklyConsumption theo + weeklyConsumption mason + weeklyConsumption roxy +
  weeklyConsumption zara + weeklyConsumption lily = 296 := by
  sorry


end NUMINAMATH_CALUDE_total_water_consumption_in_week_l926_92636


namespace NUMINAMATH_CALUDE_puppy_cost_first_year_l926_92604

def adoption_fee : ℝ := 150.00
def dog_food : ℝ := 40.00
def treats : ℝ := 3 * 5.00
def toys : ℝ := 2 * 25.00
def crate : ℝ := 120.00
def bed : ℝ := 80.00
def collar_leash : ℝ := 35.00
def grooming_tools : ℝ := 45.00
def training_classes : ℝ := 55.00 + 60.00 + 60.00 + 70.00 + 70.00
def discount_rate : ℝ := 0.12
def dog_license : ℝ := 25.00
def pet_insurance_first_half : ℝ := 6 * 25.00
def pet_insurance_second_half : ℝ := 6 * 30.00

def discountable_items : ℝ := dog_food + treats + toys + crate + bed + collar_leash + grooming_tools

theorem puppy_cost_first_year :
  let total_initial := adoption_fee + dog_food + treats + toys + crate + bed + collar_leash + grooming_tools + training_classes
  let discount := discount_rate * discountable_items
  let total_after_discount := total_initial - discount
  let total_insurance := pet_insurance_first_half + pet_insurance_second_half
  total_after_discount + dog_license + total_insurance = 1158.80 := by
sorry

end NUMINAMATH_CALUDE_puppy_cost_first_year_l926_92604


namespace NUMINAMATH_CALUDE_no_snow_no_fog_probability_l926_92632

theorem no_snow_no_fog_probability
  (p_snow : ℝ)
  (p_fog_given_no_snow : ℝ)
  (h_p_snow : p_snow = 1/4)
  (h_p_fog_given_no_snow : p_fog_given_no_snow = 1/3) :
  (1 - p_snow) * (1 - p_fog_given_no_snow) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_no_snow_no_fog_probability_l926_92632


namespace NUMINAMATH_CALUDE_coins_fit_in_new_box_l926_92620

/-- Represents a rectangular box -/
structure Box where
  width : ℝ
  height : ℝ

/-- Represents a collection of coins -/
structure CoinCollection where
  maxDiameter : ℝ

/-- Check if a coin collection can fit in a box -/
def canFitIn (coins : CoinCollection) (box : Box) : Prop :=
  box.width * box.height ≥ 0 -- This is a simplification, as we don't know the exact arrangement

/-- Theorem: If coins fit in the original box, they can fit in the new box -/
theorem coins_fit_in_new_box 
  (coins : CoinCollection)
  (originalBox : Box)
  (newBox : Box)
  (h1 : coins.maxDiameter ≤ 10)
  (h2 : originalBox.width = 30 ∧ originalBox.height = 70)
  (h3 : newBox.width = 40 ∧ newBox.height = 60)
  (h4 : canFitIn coins originalBox) :
  canFitIn coins newBox :=
by
  sorry

#check coins_fit_in_new_box

end NUMINAMATH_CALUDE_coins_fit_in_new_box_l926_92620


namespace NUMINAMATH_CALUDE_count_sequences_eq_fib_21_l926_92665

/-- The number of increasing sequences satisfying the given conditions -/
def count_sequences : ℕ := sorry

/-- The 21st Fibonacci number -/
def fib_21 : ℕ := sorry

/-- Predicate for valid sequences -/
def valid_sequence (a : ℕ → ℕ) : Prop :=
  (∀ i, 1 ≤ a i ∧ a i ≤ 20) ∧
  (∀ i j, i < j → a i < a j) ∧
  (∀ i, a i % 2 = i % 2)

theorem count_sequences_eq_fib_21 : count_sequences = fib_21 := by
  sorry

end NUMINAMATH_CALUDE_count_sequences_eq_fib_21_l926_92665


namespace NUMINAMATH_CALUDE_solve_equation_l926_92652

theorem solve_equation (x : ℚ) : 5 * (x - 9) = 3 * (3 - 3 * x) + 9 → x = 63 / 14 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l926_92652


namespace NUMINAMATH_CALUDE_internal_diagonal_cubes_l926_92666

theorem internal_diagonal_cubes (a b c : ℕ) (ha : a = 200) (hb : b = 300) (hc : c = 350) :
  a + b + c - (Nat.gcd a b + Nat.gcd b c + Nat.gcd c a) + Nat.gcd a (Nat.gcd b c) = 700 := by
  sorry

end NUMINAMATH_CALUDE_internal_diagonal_cubes_l926_92666


namespace NUMINAMATH_CALUDE_company_z_employees_l926_92674

/-- The number of employees in Company Z having birthdays on Wednesday -/
def wednesday_birthdays : ℕ := 12

/-- The number of employees in Company Z having birthdays on any day other than Wednesday -/
def other_day_birthdays : ℕ := 11

/-- The number of days in a week -/
def days_in_week : ℕ := 7

theorem company_z_employees :
  let total_employees := wednesday_birthdays + (days_in_week - 1) * other_day_birthdays
  wednesday_birthdays > other_day_birthdays →
  total_employees = 78 := by
  sorry

end NUMINAMATH_CALUDE_company_z_employees_l926_92674


namespace NUMINAMATH_CALUDE_divisibility_by_1989_l926_92644

theorem divisibility_by_1989 (n : ℕ) (h : n ≥ 3) :
  ∃ k : ℤ, n^(n^(n^n)) - n^(n^n) = 1989 * k := by sorry

end NUMINAMATH_CALUDE_divisibility_by_1989_l926_92644


namespace NUMINAMATH_CALUDE_remaining_distance_l926_92609

theorem remaining_distance (total_distance driven_distance : ℕ) 
  (h1 : total_distance = 1200)
  (h2 : driven_distance = 215) :
  total_distance - driven_distance = 985 := by
sorry

end NUMINAMATH_CALUDE_remaining_distance_l926_92609


namespace NUMINAMATH_CALUDE_exists_integer_sqrt_8m_l926_92622

theorem exists_integer_sqrt_8m : ∃ m : ℕ+, ∃ k : ℕ, (8 * m.val : ℝ).sqrt = k := by
  sorry

end NUMINAMATH_CALUDE_exists_integer_sqrt_8m_l926_92622


namespace NUMINAMATH_CALUDE_quadratic_equation_unique_solution_l926_92653

theorem quadratic_equation_unique_solution (m : ℝ) : 
  (∃! x : ℝ, 4 * x^2 + m * x + 16 = 0) ↔ m = 16 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_unique_solution_l926_92653


namespace NUMINAMATH_CALUDE_fliers_left_for_next_day_l926_92645

def total_fliers : ℕ := 2500
def morning_fraction : ℚ := 1/5
def afternoon_fraction : ℚ := 1/4

theorem fliers_left_for_next_day :
  let morning_sent := total_fliers * morning_fraction
  let remaining_after_morning := total_fliers - morning_sent
  let afternoon_sent := remaining_after_morning * afternoon_fraction
  total_fliers - morning_sent - afternoon_sent = 1500 := by sorry

end NUMINAMATH_CALUDE_fliers_left_for_next_day_l926_92645


namespace NUMINAMATH_CALUDE_total_spokes_in_garage_l926_92691

/-- The number of bicycles in the garage -/
def num_bicycles : ℕ := 4

/-- The number of spokes per wheel -/
def spokes_per_wheel : ℕ := 10

/-- The number of wheels per bicycle -/
def wheels_per_bicycle : ℕ := 2

/-- Theorem: The total number of spokes in the garage is 80 -/
theorem total_spokes_in_garage : 
  num_bicycles * wheels_per_bicycle * spokes_per_wheel = 80 := by
  sorry

end NUMINAMATH_CALUDE_total_spokes_in_garage_l926_92691


namespace NUMINAMATH_CALUDE_power_calculation_l926_92687

theorem power_calculation : 16^12 * 8^8 / 2^60 = 4096 := by
  sorry

end NUMINAMATH_CALUDE_power_calculation_l926_92687


namespace NUMINAMATH_CALUDE_kids_difference_l926_92664

/-- The number of kids Julia played with on different days of the week. -/
structure KidsPlayed where
  monday : ℕ
  wednesday : ℕ

/-- Theorem stating the difference in number of kids played with between Monday and Wednesday. -/
theorem kids_difference (k : KidsPlayed) (h1 : k.monday = 6) (h2 : k.wednesday = 4) :
  k.monday - k.wednesday = 2 := by
  sorry

end NUMINAMATH_CALUDE_kids_difference_l926_92664


namespace NUMINAMATH_CALUDE_yellow_apples_count_l926_92686

theorem yellow_apples_count (green red total : ℕ) 
  (h1 : green = 2) 
  (h2 : red = 3) 
  (h3 : total = 19) : 
  total - (green + red) = 14 := by
  sorry

end NUMINAMATH_CALUDE_yellow_apples_count_l926_92686


namespace NUMINAMATH_CALUDE_quadratic_integer_root_existence_l926_92615

theorem quadratic_integer_root_existence (a b c : ℕ) (h : a + b + c = 2000) :
  ∃ (a' b' c' : ℤ), 
    (∃ (x : ℤ), a' * x^2 + b' * x + c' = 0) ∧ 
    (|a - a'| + |b - b'| + |c - c'| : ℤ) ≤ 1050 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_integer_root_existence_l926_92615


namespace NUMINAMATH_CALUDE_sequence_properties_l926_92676

theorem sequence_properties (a b c : ℝ) : 
  (a ≠ b ∧ b ≠ c ∧ a ≠ c) →
  (∃ (d : ℝ), b = a + d ∧ c = b + d) →
  (b^2 = a*c ∧ a*c > 0) →
  (∃ (x y z : ℝ), x ≠ y ∧ y ≠ z ∧ x ≠ z ∧
    (∃ (d : ℝ), y = x + d ∧ z = y + d) ∧
    y^2 = x*z ∧ x*z > 0) ∧
  (∃ (p q r : ℝ), ¬ (∃ (m n : ℚ), p = m/n) ∧
                  ¬ (∃ (m n : ℚ), q = m/n) ∧
                  ¬ (∃ (m n : ℚ), r = m/n) ∧
                  (∃ (d : ℚ), q = p + d ∧ r = q + d)) :=
by sorry

end NUMINAMATH_CALUDE_sequence_properties_l926_92676


namespace NUMINAMATH_CALUDE_kitchen_chairs_count_l926_92659

/-- Represents the number of chairs bought for different rooms and in total. -/
structure ChairPurchase where
  total : Nat
  livingRoom : Nat

/-- Calculates the number of chairs bought for the kitchen. -/
def kitchenChairs (purchase : ChairPurchase) : Nat :=
  purchase.total - purchase.livingRoom

/-- Theorem stating that for the given purchase, the number of kitchen chairs is 6. -/
theorem kitchen_chairs_count (purchase : ChairPurchase) 
  (h1 : purchase.total = 9) 
  (h2 : purchase.livingRoom = 3) : 
  kitchenChairs purchase = 6 := by
  sorry

end NUMINAMATH_CALUDE_kitchen_chairs_count_l926_92659


namespace NUMINAMATH_CALUDE_rhombus_prism_volume_l926_92667

/-- A right prism with a rhombus base -/
structure RhombusPrism where
  /-- The acute angle of the rhombus base -/
  α : ℝ
  /-- The length of the larger diagonal of the rhombus base -/
  l : ℝ
  /-- The angle between the larger diagonal and the base plane -/
  β : ℝ
  /-- The acute angle condition -/
  h_α_acute : 0 < α ∧ α < π / 2
  /-- The positive length condition -/
  h_l_pos : l > 0
  /-- The angle β condition -/
  h_β_acute : 0 < β ∧ β < π / 2

/-- The volume of a rhombus-based right prism -/
noncomputable def volume (p : RhombusPrism) : ℝ :=
  1/2 * p.l^3 * Real.sin p.β * Real.cos p.β^2 * Real.tan (p.α/2)

theorem rhombus_prism_volume (p : RhombusPrism) :
  volume p = 1/2 * p.l^3 * Real.sin p.β * Real.cos p.β^2 * Real.tan (p.α/2) := by
  sorry

end NUMINAMATH_CALUDE_rhombus_prism_volume_l926_92667


namespace NUMINAMATH_CALUDE_max_satiated_pikes_l926_92658

/-- Represents the number of pikes in the pond -/
def total_pikes : ℕ := 30

/-- Represents the minimum number of pikes a satiated pike must eat -/
def min_eaten : ℕ := 3

/-- Predicate to check if a number is a valid count of satiated pikes -/
def is_valid_satiated_count (n : ℕ) : Prop :=
  n * min_eaten < total_pikes ∧ n ≤ total_pikes

/-- Theorem stating that the maximum number of satiated pikes is 9 -/
theorem max_satiated_pikes :
  ∃ (max : ℕ), is_valid_satiated_count max ∧
  ∀ (n : ℕ), is_valid_satiated_count n → n ≤ max :=
by sorry

end NUMINAMATH_CALUDE_max_satiated_pikes_l926_92658


namespace NUMINAMATH_CALUDE_world_not_ending_l926_92649

theorem world_not_ending (n : ℕ) : ¬(∃ k : ℕ, (1 + n) = 11 * k ∧ (3 + 7 * n) = 11 * k) := by
  sorry

end NUMINAMATH_CALUDE_world_not_ending_l926_92649


namespace NUMINAMATH_CALUDE_substitution_result_l926_92651

theorem substitution_result (x y : ℝ) : 
  (y = 2 * x - 3) → 
  (x - 2 * y = 6) → 
  (x - 4 * x + 6 = 6) := by
sorry

end NUMINAMATH_CALUDE_substitution_result_l926_92651


namespace NUMINAMATH_CALUDE_orchids_planted_today_calculation_l926_92648

/-- The number of orchid bushes planted today in the park. -/
def orchids_planted_today (current : ℕ) (tomorrow : ℕ) (final : ℕ) : ℕ :=
  final - current - tomorrow

/-- Theorem stating the number of orchid bushes planted today. -/
theorem orchids_planted_today_calculation :
  orchids_planted_today 47 25 109 = 37 := by
  sorry

end NUMINAMATH_CALUDE_orchids_planted_today_calculation_l926_92648


namespace NUMINAMATH_CALUDE_tailor_buttons_total_l926_92600

theorem tailor_buttons_total (green yellow blue total : ℕ) : 
  green = 90 →
  yellow = green + 10 →
  blue = green - 5 →
  total = green + yellow + blue →
  total = 275 := by
sorry

end NUMINAMATH_CALUDE_tailor_buttons_total_l926_92600


namespace NUMINAMATH_CALUDE_youngest_child_age_l926_92671

theorem youngest_child_age (age1 age2 age3 : ℕ) : 
  age1 < age2 ∧ age2 < age3 →
  6 + (0.60 * (age1 + age2 + age3 : ℝ)) + (3 * 0.90) = 15.30 →
  age1 = 1 :=
sorry

end NUMINAMATH_CALUDE_youngest_child_age_l926_92671


namespace NUMINAMATH_CALUDE_star_commutative_iff_three_lines_l926_92616

/-- The ⋆ operation -/
def star (a b : ℝ) : ℝ := a^2 * b - 2 * a * b^2

/-- The set of points (x, y) where x ⋆ y = y ⋆ x -/
def star_commutative_set : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | star p.1 p.2 = star p.2 p.1}

/-- The union of three lines: x = 0, y = 0, and x = y -/
def three_lines : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 = 0 ∨ p.2 = 0 ∨ p.1 = p.2}

theorem star_commutative_iff_three_lines :
  star_commutative_set = three_lines := by sorry

end NUMINAMATH_CALUDE_star_commutative_iff_three_lines_l926_92616


namespace NUMINAMATH_CALUDE_phase_shift_cos_l926_92601

theorem phase_shift_cos (b c : ℝ) : 
  let phase_shift := -c / b
  b = 2 ∧ c = π / 2 → phase_shift = -π / 4 := by
sorry

end NUMINAMATH_CALUDE_phase_shift_cos_l926_92601


namespace NUMINAMATH_CALUDE_gcd_84_120_l926_92641

theorem gcd_84_120 : Nat.gcd 84 120 = 12 := by
  sorry

end NUMINAMATH_CALUDE_gcd_84_120_l926_92641


namespace NUMINAMATH_CALUDE_triangle_area_range_l926_92647

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ :=
  if 0 < x ∧ x < 1 then -Real.log x
  else if x > 1 then Real.log x
  else 0  -- undefined for x ≤ 0 and x = 1

-- Define the derivative of f(x)
noncomputable def f_deriv (x : ℝ) : ℝ :=
  if 0 < x ∧ x < 1 then -1/x
  else if x > 1 then 1/x
  else 0  -- undefined for x ≤ 0 and x = 1

-- Theorem statement
theorem triangle_area_range (x₁ x₂ : ℝ) 
  (h₁ : 0 < x₁ ∧ x₁ < 1) (h₂ : x₂ > 1) 
  (h_perp : f_deriv x₁ * f_deriv x₂ = -1) :
  let y₁ := f x₁
  let y₂ := f x₂
  let m₁ := f_deriv x₁
  let m₂ := f_deriv x₂
  let x_int := (y₂ - y₁ + m₁*x₁ - m₂*x₂) / (m₁ - m₂)
  let area := abs ((1 - Real.log x₁ - (-1 + Real.log x₂)) * x_int / 2)
  0 < area ∧ area < 1 :=
sorry

end NUMINAMATH_CALUDE_triangle_area_range_l926_92647


namespace NUMINAMATH_CALUDE_survivor_same_tribe_probability_l926_92631

/-- The probability that both quitters are from the same tribe in a Survivor-like game. -/
theorem survivor_same_tribe_probability :
  let total_contestants : ℕ := 18
  let tribe_size : ℕ := 9
  let immune_contestants : ℕ := 1
  let quitters : ℕ := 2
  let contestants_at_risk : ℕ := total_contestants - immune_contestants
  let same_tribe_quitters : ℕ := 2 * (tribe_size.choose quitters)
  let total_quitter_combinations : ℕ := contestants_at_risk.choose quitters
  (same_tribe_quitters : ℚ) / total_quitter_combinations = 9 / 17 :=
by sorry

end NUMINAMATH_CALUDE_survivor_same_tribe_probability_l926_92631


namespace NUMINAMATH_CALUDE_line_through_coefficient_points_l926_92628

/-- Given two lines that pass through a common point, prove that the line passing through
    the points defined by the coefficients of these lines has a specific equation. -/
theorem line_through_coefficient_points (a₁ a₂ b₁ b₂ : ℝ) : 
  (a₁ * 2 + b₁ * 3 + 1 = 0) →
  (a₂ * 2 + b₂ * 3 + 1 = 0) →
  ∀ (x y : ℝ), (x = a₁ ∧ y = b₁) ∨ (x = a₂ ∧ y = b₂) → 2 * x + 3 * y + 1 = 0 := by
  sorry

end NUMINAMATH_CALUDE_line_through_coefficient_points_l926_92628


namespace NUMINAMATH_CALUDE_tan_plus_3sin_30_deg_l926_92613

theorem tan_plus_3sin_30_deg :
  Real.tan (30 * π / 180) + 3 * Real.sin (30 * π / 180) = (1 + 3 * Real.sqrt 3) / 2 :=
by sorry

end NUMINAMATH_CALUDE_tan_plus_3sin_30_deg_l926_92613


namespace NUMINAMATH_CALUDE_multiply_fractions_l926_92662

theorem multiply_fractions : (12 : ℚ) * (1 / 17) * 34 = 24 := by
  sorry

end NUMINAMATH_CALUDE_multiply_fractions_l926_92662


namespace NUMINAMATH_CALUDE_mary_baseball_cards_l926_92697

theorem mary_baseball_cards (x : ℕ) : 
  x - 8 + 26 + 40 = 84 → x = 26 := by
  sorry

end NUMINAMATH_CALUDE_mary_baseball_cards_l926_92697


namespace NUMINAMATH_CALUDE_reinforcement_size_l926_92683

/-- Calculates the size of reinforcement given initial garrison size, initial provisions duration,
    time passed before reinforcement, and remaining provisions duration after reinforcement. -/
def calculate_reinforcement (initial_garrison : ℕ) (initial_duration : ℕ) 
                            (time_passed : ℕ) (remaining_duration : ℕ) : ℕ :=
  let total_provisions := initial_garrison * initial_duration
  let provisions_left := total_provisions - (initial_garrison * time_passed)
  let reinforcement := (provisions_left / remaining_duration) - initial_garrison
  reinforcement

/-- Theorem stating that given the specific conditions of the problem,
    the calculated reinforcement size is 3000. -/
theorem reinforcement_size :
  calculate_reinforcement 2000 65 15 20 = 3000 := by
  sorry

end NUMINAMATH_CALUDE_reinforcement_size_l926_92683


namespace NUMINAMATH_CALUDE_factorization_of_2m_squared_minus_8_l926_92621

theorem factorization_of_2m_squared_minus_8 (m : ℝ) : 2 * m^2 - 8 = 2 * (m + 2) * (m - 2) := by
  sorry

end NUMINAMATH_CALUDE_factorization_of_2m_squared_minus_8_l926_92621


namespace NUMINAMATH_CALUDE_complex_equation_solution_l926_92611

theorem complex_equation_solution (z : ℂ) : z * (2 - I) = 11 + 7 * I → z = 3 + 5 * I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l926_92611


namespace NUMINAMATH_CALUDE_tangent_slope_determines_a_l926_92633

/-- Given a function f(x) = (x^2 + a) / (x + 1), prove that if the slope of the tangent line
    at x = 1 is 1, then a = -1 -/
theorem tangent_slope_determines_a (a : ℝ) :
  let f : ℝ → ℝ := λ x ↦ (x^2 + a) / (x + 1)
  (deriv f 1 = 1) → a = -1 := by
  sorry

end NUMINAMATH_CALUDE_tangent_slope_determines_a_l926_92633


namespace NUMINAMATH_CALUDE_triangle_side_length_l926_92603

theorem triangle_side_length (a c : ℝ) (B : ℝ) (h1 : a = 5) (h2 : c = 8) (h3 : B = π / 3) :
  ∃ b : ℝ, b > 0 ∧ b^2 = a^2 + c^2 - 2*a*c*(Real.cos B) ∧ b = 7 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_length_l926_92603


namespace NUMINAMATH_CALUDE_zero_in_interval_l926_92673

-- Define the function f(x) = x³ + x - 3
def f (x : ℝ) : ℝ := x^3 + x - 3

-- Theorem statement
theorem zero_in_interval : ∃ x : ℝ, x > 1 ∧ x < 2 ∧ f x = 0 := by
  sorry

end NUMINAMATH_CALUDE_zero_in_interval_l926_92673


namespace NUMINAMATH_CALUDE_smallest_divisor_after_391_l926_92619

/-- Given an even 4-digit number m where 391 is a divisor,
    the smallest possible divisor of m greater than 391 is 441 -/
theorem smallest_divisor_after_391 (m : ℕ) (h1 : 1000 ≤ m) (h2 : m < 10000) 
    (h3 : Even m) (h4 : m % 391 = 0) : 
  ∃ (d : ℕ), d ∣ m ∧ d > 391 ∧ d ≥ 441 ∧ ∀ (x : ℕ), x ∣ m → x > 391 → x ≥ d :=
by sorry

end NUMINAMATH_CALUDE_smallest_divisor_after_391_l926_92619


namespace NUMINAMATH_CALUDE_grape_juice_percentage_l926_92655

/-- Calculates the percentage of grape juice in a mixture after adding pure grape juice -/
theorem grape_juice_percentage 
  (original_volume : ℝ) 
  (original_percentage : ℝ) 
  (added_volume : ℝ) : 
  original_volume = 40 →
  original_percentage = 0.1 →
  added_volume = 10 →
  (original_volume * original_percentage + added_volume) / (original_volume + added_volume) = 0.28 := by
sorry


end NUMINAMATH_CALUDE_grape_juice_percentage_l926_92655


namespace NUMINAMATH_CALUDE_unique_sequence_l926_92637

def is_valid_sequence (a : ℕ → ℕ) : Prop :=
  (∀ n m : ℕ, a (n * m) = a n * a m) ∧
  (∀ k : ℕ, ∃ n > k, Finset.range n = Finset.image a (Finset.range n))

theorem unique_sequence (a : ℕ → ℕ) (h : is_valid_sequence a) : ∀ n : ℕ, a n = n := by
  sorry

end NUMINAMATH_CALUDE_unique_sequence_l926_92637


namespace NUMINAMATH_CALUDE_det_B_equals_two_l926_92627

theorem det_B_equals_two (p q : ℝ) (B : Matrix (Fin 2) (Fin 2) ℝ) :
  B = !![p, 3; -4, q] →
  B + 2 * B⁻¹ = 0 →
  Matrix.det B = 2 := by
sorry

end NUMINAMATH_CALUDE_det_B_equals_two_l926_92627


namespace NUMINAMATH_CALUDE_optimal_z_maximizes_optimal_z_satisfies_condition_l926_92685

open Complex

/-- The complex number that maximizes the given expression -/
def optimal_z : ℂ := -4 + I

theorem optimal_z_maximizes (z : ℂ) (h : arg (z + 3) = Real.pi * (3 / 4)) :
  1 / (abs (z + 6) + abs (z - 3 * I)) ≤ 1 / (abs (optimal_z + 6) + abs (optimal_z - 3 * I)) :=
by sorry

theorem optimal_z_satisfies_condition :
  arg (optimal_z + 3) = Real.pi * (3 / 4) :=
by sorry

end NUMINAMATH_CALUDE_optimal_z_maximizes_optimal_z_satisfies_condition_l926_92685


namespace NUMINAMATH_CALUDE_complex_number_problem_l926_92695

theorem complex_number_problem (a : ℝ) :
  let z : ℂ := a + (10 * Complex.I) / (3 - Complex.I)
  (∃ b : ℝ, z = Complex.I * b) → Complex.abs (a - 2 * Complex.I) = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_complex_number_problem_l926_92695


namespace NUMINAMATH_CALUDE_z_in_fourth_quadrant_l926_92668

-- Define the complex number z
def z : ℂ := (2 - Complex.I) ^ 2

-- Theorem stating that z is in the fourth quadrant
theorem z_in_fourth_quadrant : 
  Real.sign (z.re) = 1 ∧ Real.sign (z.im) = -1 :=
sorry

end NUMINAMATH_CALUDE_z_in_fourth_quadrant_l926_92668


namespace NUMINAMATH_CALUDE_constant_term_expansion_l926_92682

theorem constant_term_expansion (n : ℕ) (h1 : 2 ≤ n) (h2 : n ≤ 10) :
  (∃ k : ℕ, ∃ r : ℕ, n = 3 * r ∧ n = 2 * k) ↔ n = 6 := by
  sorry

end NUMINAMATH_CALUDE_constant_term_expansion_l926_92682


namespace NUMINAMATH_CALUDE_triangle_perimeter_l926_92614

-- Define the lines
def line_through_origin (m : ℝ) := {(x, y) : ℝ × ℝ | y = m * x}
def vertical_line (a : ℝ) := {(x, y) : ℝ × ℝ | x = a}
def sloped_line (m : ℝ) (b : ℝ) := {(x, y) : ℝ × ℝ | y = m * x + b}

-- Define the triangle
def right_triangle (m : ℝ) := 
  (0, 0) ∈ line_through_origin m ∧
  (1, -m) ∈ line_through_origin m ∧
  (1, -m) ∈ vertical_line 1 ∧
  (1, 1.5) ∈ sloped_line (1/2) 1 ∧
  (1, 1.5) ∈ vertical_line 1

-- Theorem statement
theorem triangle_perimeter :
  ∀ m : ℝ, right_triangle m → 
  (Real.sqrt ((1:ℝ)^2 + m^2) + Real.sqrt ((1:ℝ)^2 + (1.5 + m)^2) + 0.5) = 3 + Real.sqrt 5 :=
sorry

end NUMINAMATH_CALUDE_triangle_perimeter_l926_92614


namespace NUMINAMATH_CALUDE_intersection_points_l926_92639

-- Define the functions f and g
def f (t x : ℝ) : ℝ := t * x^2 - x + 1
def g (t x : ℝ) : ℝ := 2 * t * x - 1

-- Define the discriminant function
def discriminant (t : ℝ) : ℝ := (2 * t - 1)^2

-- Theorem statement
theorem intersection_points (t : ℝ) :
  (∃ x : ℝ, f t x = g t x) ∧
  (∀ x y : ℝ, f t x = g t x ∧ f t y = g t y → x = y ∨ (∃ z : ℝ, f t z = g t z ∧ z ≠ x ∧ z ≠ y)) :=
by sorry

end NUMINAMATH_CALUDE_intersection_points_l926_92639


namespace NUMINAMATH_CALUDE_weight_difference_l926_92612

/-- Given the combined weights of Annette and Caitlin, and Caitlin and Sara,
    prove that Annette weighs 8 pounds more than Sara. -/
theorem weight_difference (a c s : ℝ) 
  (h1 : a + c = 95) 
  (h2 : c + s = 87) : 
  a - s = 8 := by
  sorry

end NUMINAMATH_CALUDE_weight_difference_l926_92612


namespace NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l926_92624

/-- An isosceles triangle with side lengths 2 and 5 has a perimeter of 12 -/
theorem isosceles_triangle_perimeter : 
  ∀ (a b c : ℝ), 
  a = 5 → b = 5 → c = 2 →  -- Two sides are 5, one side is 2
  (a = b ∨ b = c ∨ a = c) →  -- The triangle is isosceles
  a + b + c = 12 :=  -- The perimeter is 12
by
  sorry


end NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l926_92624


namespace NUMINAMATH_CALUDE_composite_shape_area_l926_92602

/-- The total area of a composite shape consisting of three rectangles -/
def composite_area (rect1_width rect1_height rect2_width rect2_height rect3_width rect3_height : ℕ) : ℕ :=
  rect1_width * rect1_height + rect2_width * rect2_height + rect3_width * rect3_height

/-- Theorem stating that the area of the given composite shape is 68 square units -/
theorem composite_shape_area : composite_area 7 6 3 2 4 5 = 68 := by
  sorry

end NUMINAMATH_CALUDE_composite_shape_area_l926_92602


namespace NUMINAMATH_CALUDE_triangle_ratio_l926_92661

theorem triangle_ratio (A B C : ℝ) (a b c : ℝ) : 
  A = π/3 → b = 1 → c = 4 → 
  (a + b + c) / (Real.sin A + Real.sin B + Real.sin C) = 2 * Real.sqrt 39 / 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_ratio_l926_92661


namespace NUMINAMATH_CALUDE_factorization_proof_l926_92681

theorem factorization_proof (x : ℝ) : 
  (x^2 - 1) * (x^4 + x^2 + 1) - (x^3 + 1)^2 = -2 * (x + 1) * (x^2 - x + 1) := by
  sorry

end NUMINAMATH_CALUDE_factorization_proof_l926_92681


namespace NUMINAMATH_CALUDE_min_distance_point_l926_92693

/-- Given a triangle ABC with vertices (x₁, y₁), (x₂, y₂), (x₃, y₃), 
    the point P that minimizes the sum of squares of distances from P to the three vertices 
    has coordinates ((x₁ + x₂ + x₃) / 3, (y₁ + y₂ + y₃) / 3) -/
theorem min_distance_point (x₁ y₁ x₂ y₂ x₃ y₃ : ℝ) :
  let dist_sum_sq (x y : ℝ) := 
    (x - x₁)^2 + (y - y₁)^2 + (x - x₂)^2 + (y - y₂)^2 + (x - x₃)^2 + (y - y₃)^2
  ∃ (x y : ℝ), (∀ (u v : ℝ), dist_sum_sq x y ≤ dist_sum_sq u v) ∧ 
    x = (x₁ + x₂ + x₃) / 3 ∧ y = (y₁ + y₂ + y₃) / 3 := by
  sorry

end NUMINAMATH_CALUDE_min_distance_point_l926_92693


namespace NUMINAMATH_CALUDE_smallest_soldier_arrangement_l926_92623

theorem smallest_soldier_arrangement : ∃ (n : ℕ), n > 0 ∧
  (∀ k ∈ ({2, 3, 4, 5, 6, 7, 8, 9, 10} : Finset ℕ), n % k = k - 1) ∧
  (∀ m : ℕ, m > 0 → (∀ k ∈ ({2, 3, 4, 5, 6, 7, 8, 9, 10} : Finset ℕ), m % k = k - 1) → m ≥ n) ∧
  n = 2519 := by
  sorry

end NUMINAMATH_CALUDE_smallest_soldier_arrangement_l926_92623


namespace NUMINAMATH_CALUDE_solution_set_of_inequality_range_of_a_l926_92684

-- Define the function f(x)
def f (x : ℝ) : ℝ := |x + 1| + |x - 5|

-- Theorem for part I
theorem solution_set_of_inequality (x : ℝ) :
  (f x ≤ x + 10) ↔ (x ∈ Set.Icc (-2) 14) :=
sorry

-- Theorem for part II
theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, f x ≥ a - (x - 2)^2) ↔ (a ∈ Set.Iic 6) :=
sorry

end NUMINAMATH_CALUDE_solution_set_of_inequality_range_of_a_l926_92684


namespace NUMINAMATH_CALUDE_union_equals_interval_l926_92607

-- Define sets A and B
def A : Set ℝ := {x : ℝ | x^2 + 2*x - 3 < 0}
def B : Set ℝ := {x : ℝ | x^2 - 4*x ≤ 0}

-- Define the interval (-3, 4]
def interval : Set ℝ := Set.Ioc (-3) 4

-- Theorem statement
theorem union_equals_interval : A ∪ B = interval := by sorry

end NUMINAMATH_CALUDE_union_equals_interval_l926_92607


namespace NUMINAMATH_CALUDE_infinite_rational_points_in_circle_l926_92605

theorem infinite_rational_points_in_circle : 
  ∀ ε > 0, ∃ (x y : ℚ), x > 0 ∧ y > 0 ∧ x^2 + y^2 ≤ 25 ∧ 
  ∀ (x' y' : ℚ), x' > 0 → y' > 0 → x'^2 + y'^2 ≤ 25 → (x - x')^2 + (y - y')^2 < ε^2 :=
sorry

end NUMINAMATH_CALUDE_infinite_rational_points_in_circle_l926_92605


namespace NUMINAMATH_CALUDE_find_a_value_l926_92669

theorem find_a_value (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a^b = b^a) (h4 : b = 4*a) : a = (4 : ℝ)^(1/3) := by
  sorry

end NUMINAMATH_CALUDE_find_a_value_l926_92669


namespace NUMINAMATH_CALUDE_smallest_four_digit_divisible_by_37_l926_92692

theorem smallest_four_digit_divisible_by_37 : ∃ n : ℕ, 
  (n ≥ 1000 ∧ n < 10000) ∧  -- four-digit number
  n % 37 = 0 ∧              -- divisible by 37
  (∀ m : ℕ, (m ≥ 1000 ∧ m < 10000) ∧ m % 37 = 0 → n ≤ m) ∧  -- smallest such number
  n = 1036 :=               -- the answer is 1036
by sorry

end NUMINAMATH_CALUDE_smallest_four_digit_divisible_by_37_l926_92692


namespace NUMINAMATH_CALUDE_stream_speed_l926_92694

/-- Proves that given a boat with a speed of 22 km/hr in still water, 
    traveling 54 km downstream in 2 hours, the speed of the stream is 5 km/hr. -/
theorem stream_speed 
  (boat_speed : ℝ) 
  (distance : ℝ) 
  (time : ℝ) 
  (h1 : boat_speed = 22)
  (h2 : distance = 54)
  (h3 : time = 2)
  : ∃ (stream_speed : ℝ), 
    distance = (boat_speed + stream_speed) * time ∧ 
    stream_speed = 5 :=
by
  sorry

end NUMINAMATH_CALUDE_stream_speed_l926_92694


namespace NUMINAMATH_CALUDE_man_walking_speed_l926_92663

/-- Calculates the walking speed of a man given the following conditions:
  * The man walks at a constant speed
  * He takes a 5-minute rest after every kilometer
  * He covers 5 kilometers in 50 minutes
-/
theorem man_walking_speed (total_time : ℝ) (total_distance : ℝ) (rest_time : ℝ) 
  (rest_frequency : ℝ) (h1 : total_time = 50) (h2 : total_distance = 5) 
  (h3 : rest_time = 5) (h4 : rest_frequency = 1) : 
  (total_distance / ((total_time - (rest_time * (total_distance - 1))) / 60)) = 10 := by
  sorry

#check man_walking_speed

end NUMINAMATH_CALUDE_man_walking_speed_l926_92663


namespace NUMINAMATH_CALUDE_mihaly_third_day_foxes_l926_92629

/-- Represents the number of animals hunted by a person on a specific day -/
structure DailyHunt where
  rabbits : ℕ
  foxes : ℕ
  pheasants : ℕ

/-- Represents the total hunt over three days for a person -/
structure ThreeDayHunt where
  day1 : DailyHunt
  day2 : DailyHunt
  day3 : DailyHunt

def Karoly : ThreeDayHunt := sorry
def Laszlo : ThreeDayHunt := sorry
def Mihaly : ThreeDayHunt := sorry

def total_animals : ℕ := 86

def first_day_foxes : ℕ := 12
def first_day_rabbits : ℕ := 14

def second_day_total : ℕ := 44

def total_pheasants : ℕ := 12

theorem mihaly_third_day_foxes :
  (∀ d : DailyHunt, d.rabbits ≥ 1 ∧ d.foxes ≥ 1 ∧ d.pheasants ≥ 1) →
  (∀ d : DailyHunt, d ≠ Laszlo.day2 → Even d.rabbits ∧ Even d.foxes ∧ Even d.pheasants) →
  Laszlo.day2.foxes = 5 →
  (Karoly.day1.foxes + Laszlo.day1.foxes + Mihaly.day1.foxes = first_day_foxes) →
  (Karoly.day1.rabbits + Laszlo.day1.rabbits + Mihaly.day1.rabbits = first_day_rabbits) →
  (Karoly.day2.rabbits + Karoly.day2.foxes + Karoly.day2.pheasants +
   Laszlo.day2.rabbits + Laszlo.day2.foxes + Laszlo.day2.pheasants +
   Mihaly.day2.rabbits + Mihaly.day2.foxes + Mihaly.day2.pheasants = second_day_total) →
  (Karoly.day1.pheasants + Karoly.day2.pheasants + Karoly.day3.pheasants +
   Laszlo.day1.pheasants + Laszlo.day2.pheasants + Laszlo.day3.pheasants +
   Mihaly.day1.pheasants + Mihaly.day2.pheasants + Mihaly.day3.pheasants = total_pheasants) →
  (Karoly.day1.rabbits + Karoly.day1.foxes + Karoly.day1.pheasants +
   Karoly.day2.rabbits + Karoly.day2.foxes + Karoly.day2.pheasants +
   Karoly.day3.rabbits + Karoly.day3.foxes + Karoly.day3.pheasants +
   Laszlo.day1.rabbits + Laszlo.day1.foxes + Laszlo.day1.pheasants +
   Laszlo.day2.rabbits + Laszlo.day2.foxes + Laszlo.day2.pheasants +
   Laszlo.day3.rabbits + Laszlo.day3.foxes + Laszlo.day3.pheasants +
   Mihaly.day1.rabbits + Mihaly.day1.foxes + Mihaly.day1.pheasants +
   Mihaly.day2.rabbits + Mihaly.day2.foxes + Mihaly.day2.pheasants +
   Mihaly.day3.rabbits + Mihaly.day3.foxes + Mihaly.day3.pheasants = total_animals) →
  Mihaly.day3.foxes = 1 := by
  sorry

end NUMINAMATH_CALUDE_mihaly_third_day_foxes_l926_92629


namespace NUMINAMATH_CALUDE_committee_selection_count_l926_92689

/-- Represents the total number of members in the class committee -/
def totalMembers : Nat := 5

/-- Represents the number of roles to be filled -/
def rolesToFill : Nat := 3

/-- Represents the number of members who cannot serve in a specific role -/
def restrictedMembers : Nat := 2

/-- Calculates the number of ways to select committee members under given constraints -/
def selectCommitteeMembers (total : Nat) (roles : Nat) (restricted : Nat) : Nat :=
  (total - restricted) * (total - 1) * (total - 2)

theorem committee_selection_count :
  selectCommitteeMembers totalMembers rolesToFill restrictedMembers = 36 := by
  sorry

#eval selectCommitteeMembers totalMembers rolesToFill restrictedMembers

end NUMINAMATH_CALUDE_committee_selection_count_l926_92689


namespace NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_l926_92617

-- Problem 1
theorem problem_1 (x : ℝ) (h : x = 2 - Real.sqrt 7) : x^2 - 4*x + 5 = 8 := by
  sorry

-- Problem 2
theorem problem_2 (x : ℝ) (h : 2*x = Real.sqrt 5 + 1) : x^3 - 2*x^2 = -1 := by
  sorry

-- Problem 3
theorem problem_3 (a : ℝ) (h : a^2 = Real.sqrt (a^2 + 10) + 3) : a^2 + 1/a^2 = Real.sqrt 53 := by
  sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_l926_92617


namespace NUMINAMATH_CALUDE_intersection_chord_length_l926_92646

/-- Circle in 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Line passing through origin in polar form -/
structure PolarLine where
  angle : ℝ

/-- Chord formed by intersection of circle and line -/
def chord_length (c : Circle) (l : PolarLine) : ℝ :=
  sorry

theorem intersection_chord_length :
  let c : Circle := { center := (0, -6), radius := 5 }
  let l : PolarLine := { angle := Real.arctan (Real.sqrt 5 / 2) }
  chord_length c l = 6 := by sorry

end NUMINAMATH_CALUDE_intersection_chord_length_l926_92646


namespace NUMINAMATH_CALUDE_rect_to_cylindrical_conversion_l926_92626

/-- Conversion from rectangular to cylindrical coordinates -/
theorem rect_to_cylindrical_conversion :
  let x : ℝ := 3
  let y : ℝ := -3 * Real.sqrt 3
  let z : ℝ := 4
  let r : ℝ := 6
  let θ : ℝ := 4 * π / 3
  r > 0 ∧ 0 ≤ θ ∧ θ < 2 * π ∧
  x = r * Real.cos θ ∧
  y = r * Real.sin θ ∧
  z = z := by sorry

end NUMINAMATH_CALUDE_rect_to_cylindrical_conversion_l926_92626


namespace NUMINAMATH_CALUDE_equal_distances_l926_92670

/-- The number of people seated at the round table. -/
def n : ℕ := 41

/-- The distance between two positions in a circular arrangement. -/
def circularDistance (a b : ℕ) : ℕ :=
  min ((a - b + n) % n) ((b - a + n) % n)

/-- Theorem stating that the distance from 31 to 7 equals the distance from 31 to 14
    in a circular arrangement of 41 people. -/
theorem equal_distances : circularDistance 31 7 = circularDistance 31 14 := by
  sorry


end NUMINAMATH_CALUDE_equal_distances_l926_92670


namespace NUMINAMATH_CALUDE_lindas_tv_cost_l926_92635

def lindas_problem (original_savings : ℝ) (furniture_fraction : ℝ) : ℝ :=
  original_savings * (1 - furniture_fraction)

theorem lindas_tv_cost :
  lindas_problem 500 (4/5) = 100 := by sorry

end NUMINAMATH_CALUDE_lindas_tv_cost_l926_92635


namespace NUMINAMATH_CALUDE_red_blood_cell_diameter_scientific_notation_l926_92677

/-- Scientific notation representation of a real number -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  is_valid : 1 ≤ coefficient ∧ coefficient < 10

/-- Convert a real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem red_blood_cell_diameter_scientific_notation :
  toScientificNotation 0.0000077 = ScientificNotation.mk 7.7 (-6) (by sorry) :=
sorry

end NUMINAMATH_CALUDE_red_blood_cell_diameter_scientific_notation_l926_92677
