import Mathlib

namespace x_fifth_minus_five_x_equals_3100_l929_92962

theorem x_fifth_minus_five_x_equals_3100 (x : ℝ) (h : x = 5) : x^5 - 5*x = 3100 := by
  sorry

end x_fifth_minus_five_x_equals_3100_l929_92962


namespace stating_transport_equation_transport_scenario_proof_l929_92902

/-- Represents the scenario of two transports moving towards each other. -/
structure TransportScenario where
  x : ℝ  -- Speed of transport A in mph
  T : ℝ  -- Time in hours after transport A's departure when they are 348 miles apart

/-- 
  Theorem stating the relationship between the speeds and time
  for the given transport scenario.
-/
theorem transport_equation (scenario : TransportScenario) :
  let x := scenario.x
  let T := scenario.T
  2 * x * T + 18 * T - x - 18 = 258 := by
  sorry

/-- 
  Proves that the equation holds for the given transport scenario
  where two transports start 90 miles apart, with one traveling at speed x mph
  and the other at (x + 18) mph, starting 1 hour later, and end up 348 miles apart.
-/
theorem transport_scenario_proof (x : ℝ) (T : ℝ) :
  let scenario : TransportScenario := { x := x, T := T }
  (2 * x * T + 18 * T - x - 18 = 258) ↔
  (x * T + (x + 18) * (T - 1) = 348 - 90) := by
  sorry

end stating_transport_equation_transport_scenario_proof_l929_92902


namespace cos_beta_value_l929_92953

theorem cos_beta_value (α β : Real) (h1 : 0 < α ∧ α < π / 2) (h2 : 0 < β ∧ β < π / 2)
  (h3 : Real.tan α = 2) (h4 : Real.sin (α + β) = Real.sqrt 2 / 2) :
  Real.cos β = Real.sqrt 10 / 10 := by
  sorry

end cos_beta_value_l929_92953


namespace sum_properties_l929_92998

theorem sum_properties (a b : ℤ) (ha : 4 ∣ a) (hb : 8 ∣ b) : 
  Even (a + b) ∧ (4 ∣ (a + b)) ∧ ¬(∀ (a b : ℤ), 4 ∣ a → 8 ∣ b → 8 ∣ (a + b)) := by
  sorry

end sum_properties_l929_92998


namespace distinct_prime_factors_count_l929_92917

theorem distinct_prime_factors_count : ∃ (p : ℕ → Prop), 
  (∀ n, p n ↔ Nat.Prime n) ∧ 
  (∃ (S : Finset ℕ), 
    (∀ x ∈ S, p x) ∧ 
    Finset.card S = 7 ∧
    (∀ q, p q → q ∣ ((87 * 89 * 91 + 1) * 93) ↔ q ∈ S)) := by
  sorry

end distinct_prime_factors_count_l929_92917


namespace polynomial_simplification_l929_92996

theorem polynomial_simplification (x : ℝ) :
  (2 * x^10 + 8 * x^9 + 3 * x^8) + (5 * x^12 - x^10 + 2 * x^9 - 5 * x^8 + 4 * x^5 + 6) =
  5 * x^12 + x^10 + 10 * x^9 - 2 * x^8 + 4 * x^5 + 6 :=
by sorry

end polynomial_simplification_l929_92996


namespace exponential_inverse_sum_l929_92904

-- Define the exponential function f
def f (x : ℝ) : ℝ := sorry

-- Define the inverse function g
def g (x : ℝ) : ℝ := sorry

-- Theorem statement
theorem exponential_inverse_sum :
  (∃ (a : ℝ), ∀ (x : ℝ), f x = a^x) →  -- f is an exponential function
  (f (1 + Real.sqrt 3) * f (1 - Real.sqrt 3) = 9) →  -- Given condition
  (∀ (x : ℝ), g (f x) = x ∧ f (g x) = x) →  -- g is the inverse of f
  (g (Real.sqrt 10 + 1) + g (Real.sqrt 10 - 1) = 2) :=
by sorry

end exponential_inverse_sum_l929_92904


namespace hyperbola_eccentricity_l929_92909

/-- The eccentricity of a hyperbola given specific conditions -/
theorem hyperbola_eccentricity (a b c p : ℝ) (x₀ y₀ : ℝ) : 
  a > 0 → b > 0 → p > 0 → x₀ > 0 → y₀ > 0 →
  x₀^2 / a^2 - y₀^2 / b^2 = 1 →  -- hyperbola equation
  y₀ = (b / a) * x₀ →  -- point on asymptote
  x₀^2 + y₀^2 = c^2 →  -- MF₁ ⊥ MF₂
  y₀^2 = 2 * p * x₀ →  -- parabola equation
  c / a = 2 + Real.sqrt 5 := by
sorry

end hyperbola_eccentricity_l929_92909


namespace base_ten_to_base_five_158_l929_92972

/-- Converts a natural number to its base 5 representation -/
def toBaseFive (n : ℕ) : List ℕ :=
  if n = 0 then [0] else
  let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
    if m = 0 then acc else aux (m / 5) ((m % 5) :: acc)
  aux n []

/-- Checks if a list of digits is a valid base 5 representation -/
def isValidBaseFive (digits : List ℕ) : Prop :=
  digits.all (· < 5) ∧ digits ≠ []

theorem base_ten_to_base_five_158 :
  let base_five_repr := toBaseFive 158
  isValidBaseFive base_five_repr ∧ base_five_repr = [1, 1, 3, 3] := by sorry

end base_ten_to_base_five_158_l929_92972


namespace ellipse_foci_l929_92938

/-- The equation of the ellipse -/
def ellipse_equation (x y : ℝ) : Prop :=
  x^2 / 64 + y^2 / 100 = 1

/-- The coordinates of a focus of the ellipse -/
def focus_coordinate : ℝ × ℝ := (0, 6)

/-- Theorem stating that the given coordinates are the foci of the ellipse -/
theorem ellipse_foci :
  (ellipse_equation (focus_coordinate.1) (focus_coordinate.2) ∧
   ellipse_equation (focus_coordinate.1) (-focus_coordinate.2)) ∧
  (∀ x y : ℝ, ellipse_equation x y →
    (x^2 + y^2 < focus_coordinate.1^2 + focus_coordinate.2^2 ∨
     x^2 + y^2 = focus_coordinate.1^2 + focus_coordinate.2^2)) :=
by sorry

end ellipse_foci_l929_92938


namespace lcm_12_20_l929_92978

theorem lcm_12_20 : Nat.lcm 12 20 = 60 := by
  sorry

end lcm_12_20_l929_92978


namespace married_fraction_l929_92982

theorem married_fraction (total : ℕ) (women_fraction : ℚ) (max_unmarried_women : ℕ) :
  total = 80 →
  women_fraction = 1/4 →
  max_unmarried_women = 20 →
  (total - max_unmarried_women : ℚ) / total = 3/4 := by
sorry

end married_fraction_l929_92982


namespace unique_positive_solution_l929_92997

theorem unique_positive_solution :
  ∃! (x : ℝ), x > 0 ∧ Real.sqrt (18 * x) * Real.sqrt (2 * x) * Real.sqrt (25 * x) * Real.sqrt (5 * x) = 50 ∧ 
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.0001 ∧ |x - 0.8632| < ε := by
sorry

end unique_positive_solution_l929_92997


namespace pizza_slices_sold_l929_92951

/-- Proves that the number of small slices sold is 2000 -/
theorem pizza_slices_sold (small_price large_price : ℕ) 
  (total_slices total_revenue : ℕ) (h1 : small_price = 150) 
  (h2 : large_price = 250) (h3 : total_slices = 5000) 
  (h4 : total_revenue = 1050000) : 
  ∃ (small_slices large_slices : ℕ),
    small_slices + large_slices = total_slices ∧
    small_price * small_slices + large_price * large_slices = total_revenue ∧
    small_slices = 2000 := by
  sorry

end pizza_slices_sold_l929_92951


namespace log_equation_solution_l929_92979

theorem log_equation_solution (x : ℝ) (h : x > 0) :
  Real.log x / Real.log 8 + Real.log (x^3) / Real.log 4 = 9 →
  x = 2^(54/11) := by
sorry

end log_equation_solution_l929_92979


namespace exists_close_vertices_l929_92926

/-- A regular polygon with 2n+1 sides inscribed in a unit circle -/
structure RegularPolygon (n : ℕ) where
  vertices : Fin (2*n+1) → ℝ × ℝ
  is_regular : ∀ i : Fin (2*n+1), norm (vertices i) = 1

/-- A point inside the polygon -/
def InsidePoint (n : ℕ) (poly : RegularPolygon n) := { p : ℝ × ℝ // norm p < 1 }

/-- Distance between two points -/
def distance (p q : ℝ × ℝ) : ℝ := norm (p.1 - q.1, p.2 - q.2)

/-- Statement of the theorem -/
theorem exists_close_vertices (n : ℕ) :
  ∃ α : ℝ, α > 0 ∧
  ∀ (poly : RegularPolygon n) (p : InsidePoint n poly),
  ∃ (i j : Fin (2*n+1)), i ≠ j ∧
  |distance p.val (poly.vertices i) - distance p.val (poly.vertices j)| < 1/n - α/n^3 :=
sorry

end exists_close_vertices_l929_92926


namespace min_value_theorem_l929_92977

theorem min_value_theorem (x : ℝ) (h : x > 0) :
  x^2 + 9*x + 81/x^4 ≥ 19 ∧
  (x^2 + 9*x + 81/x^4 = 19 ↔ x = 3) := by
sorry

end min_value_theorem_l929_92977


namespace only_piston_and_bottles_are_translations_l929_92949

/-- Represents a type of motion --/
inductive Motion
| Translation
| Rotation
| Other

/-- Represents the different phenomena described in the problem --/
inductive Phenomenon
| ChildSwinging
| PistonMovement
| PendulumSwinging
| BottlesOnConveyorBelt

/-- Determines the type of motion for a given phenomenon --/
def motionType (p : Phenomenon) : Motion :=
  match p with
  | Phenomenon.ChildSwinging => Motion.Rotation
  | Phenomenon.PistonMovement => Motion.Translation
  | Phenomenon.PendulumSwinging => Motion.Rotation
  | Phenomenon.BottlesOnConveyorBelt => Motion.Translation

/-- Theorem stating that only the piston movement and bottles on conveyor belt are translations --/
theorem only_piston_and_bottles_are_translations :
  (∀ p : Phenomenon, motionType p = Motion.Translation ↔ 
    (p = Phenomenon.PistonMovement ∨ p = Phenomenon.BottlesOnConveyorBelt)) :=
by sorry

end only_piston_and_bottles_are_translations_l929_92949


namespace wire_service_reporters_l929_92931

theorem wire_service_reporters (total_reporters : ℝ) 
  (local_politics_reporters : ℝ) (politics_reporters : ℝ) :
  local_politics_reporters = 0.12 * total_reporters →
  local_politics_reporters = 0.6 * politics_reporters →
  total_reporters - politics_reporters = 0.8 * total_reporters :=
by
  sorry

end wire_service_reporters_l929_92931


namespace perpendicular_lines_b_value_l929_92920

/-- Given two perpendicular lines with direction vectors (2, 5) and (b, -3), prove that b = 15/2 -/
theorem perpendicular_lines_b_value (b : ℝ) : 
  let v₁ : Fin 2 → ℝ := ![2, 5]
  let v₂ : Fin 2 → ℝ := ![b, -3]
  (∀ i : Fin 2, v₁ i * v₂ i = 0) → b = 15/2 := by
sorry

end perpendicular_lines_b_value_l929_92920


namespace inequality_proof_l929_92987

theorem inequality_proof (a b c : ℝ) 
  (h1 : a > b) 
  (h2 : b > c) 
  (h3 : a + b + c = 0) : 
  Real.sqrt (b^2 - a*c) < Real.sqrt (3*a) := by
sorry

end inequality_proof_l929_92987


namespace cat_cafe_ratio_l929_92957

/-- The number of cats in Cat Cafe Cool -/
def cool_cats : ℕ := 5

/-- The number of cats in Cat Cafe Paw -/
def paw_cats : ℕ := 10

/-- The number of cats in Cat Cafe Meow -/
def meow_cats : ℕ := 3 * paw_cats

/-- The total number of cats in Cat Cafe Meow and Cat Cafe Paw -/
def total_cats : ℕ := 40

/-- The theorem stating the ratio of cats between Cat Cafe Paw and Cat Cafe Cool -/
theorem cat_cafe_ratio : paw_cats / cool_cats = 2 := by
  sorry

end cat_cafe_ratio_l929_92957


namespace intersection_of_A_and_B_l929_92980

def A : Set ℝ := {x : ℝ | -1 < x ∧ x < 4}
def B : Set ℝ := {0, 2, 4, 6}

theorem intersection_of_A_and_B : A ∩ B = {0, 2} := by sorry

end intersection_of_A_and_B_l929_92980


namespace other_birds_percentage_l929_92958

/-- Represents the composition of birds in the Goshawk-Eurasian Nature Reserve -/
structure BirdReserve where
  total : ℝ
  hawk_percent : ℝ
  paddyfield_warbler_percent_of_nonhawk : ℝ
  kingfisher_to_paddyfield_warbler_ratio : ℝ

/-- Theorem stating the percentage of birds that are not hawks, paddyfield-warblers, or kingfishers -/
theorem other_birds_percentage (reserve : BirdReserve) 
  (h1 : reserve.hawk_percent = 0.3)
  (h2 : reserve.paddyfield_warbler_percent_of_nonhawk = 0.4)
  (h3 : reserve.kingfisher_to_paddyfield_warbler_ratio = 0.25)
  (h4 : reserve.total > 0) :
  let hawk_count := reserve.hawk_percent * reserve.total
  let nonhawk_count := reserve.total - hawk_count
  let paddyfield_warbler_count := reserve.paddyfield_warbler_percent_of_nonhawk * nonhawk_count
  let kingfisher_count := reserve.kingfisher_to_paddyfield_warbler_ratio * paddyfield_warbler_count
  let other_count := reserve.total - (hawk_count + paddyfield_warbler_count + kingfisher_count)
  (other_count / reserve.total) = 0.35 := by
  sorry

end other_birds_percentage_l929_92958


namespace square_area_perimeter_ratio_l929_92907

theorem square_area_perimeter_ratio : 
  ∀ (s1 s2 : ℝ), s1 > 0 ∧ s2 > 0 →
  (s1^2 : ℝ) / (s2^2 : ℝ) = 49 / 64 →
  (4 * s1) / (4 * s2) = 7 / 8 :=
by
  sorry

end square_area_perimeter_ratio_l929_92907


namespace extremum_at_negative_three_l929_92916

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^3 + a*x^2 + 3*x - 9

-- Define the derivative of f
def f' (a : ℝ) (x : ℝ) : ℝ := 3*x^2 + 2*a*x + 3

-- Theorem statement
theorem extremum_at_negative_three (a : ℝ) :
  (∃ (ε : ℝ), ε > 0 ∧ ∀ (x : ℝ), x ≠ -3 ∧ |x + 3| < ε → f a x ≤ f a (-3)) →
  a = 5 :=
sorry

end extremum_at_negative_three_l929_92916


namespace history_score_calculation_l929_92914

def geography_score : ℕ := 50
def math_score : ℕ := 70
def english_score : ℕ := 66
def total_score : ℕ := 248

theorem history_score_calculation :
  total_score - (geography_score + math_score + english_score) =
  total_score - geography_score - math_score - english_score :=
by sorry

end history_score_calculation_l929_92914


namespace q_min_at_two_l929_92903

/-- The function q in terms of x -/
def q (x : ℝ) : ℝ := (x - 5)^2 + (x + 1)^2 - 18

/-- The theorem stating that q is minimized to 0 when x = 2 -/
theorem q_min_at_two : 
  (∀ x : ℝ, q x ≥ q 2) ∧ q 2 = 0 :=
sorry

end q_min_at_two_l929_92903


namespace lcm_gcd_210_396_l929_92942

theorem lcm_gcd_210_396 :
  (Nat.lcm 210 396 = 4620) ∧ (Nat.gcd 210 396 = 6) := by
  sorry

end lcm_gcd_210_396_l929_92942


namespace lowest_price_type_a_l929_92934

/-- Calculates the final price of a pet food type given its MSRP, regular discount, additional discount, and sales tax rate -/
def finalPrice (msrp : ℝ) (regularDiscount : ℝ) (additionalDiscount : ℝ) (salesTax : ℝ) : ℝ :=
  let discountedPrice := msrp * (1 - regularDiscount)
  let furtherDiscountedPrice := discountedPrice * (1 - additionalDiscount)
  furtherDiscountedPrice * (1 + salesTax)

theorem lowest_price_type_a (msrp_a msrp_b msrp_c : ℝ) :
  msrp_a = 45 ∧ msrp_b = 55 ∧ msrp_c = 50 →
  finalPrice msrp_a 0.15 0.20 0.07 < finalPrice msrp_b 0.25 0.15 0.07 ∧
  finalPrice msrp_a 0.15 0.20 0.07 < finalPrice msrp_c 0.30 0.10 0.07 :=
by sorry

end lowest_price_type_a_l929_92934


namespace inverse_composition_result_l929_92946

-- Define the functions f and h
variable (f h : ℝ → ℝ)

-- Define the inverse functions
variable (f_inv h_inv : ℝ → ℝ)

-- State the given condition
axiom condition : ∀ x, f_inv (h x) = 6 * x - 4

-- State the theorem to be proved
theorem inverse_composition_result : h_inv (f 3) = 7/6 := by sorry

end inverse_composition_result_l929_92946


namespace function_properties_l929_92906

/-- Given function f with properties as described -/
def f (x : ℝ) : ℝ := sorry

/-- ω is a positive real number -/
def ω : ℝ := sorry

/-- φ is a real number between 0 and π -/
def φ : ℝ := sorry

theorem function_properties (x α : ℝ) :
  ω > 0 ∧
  0 ≤ φ ∧ φ ≤ π ∧
  (∀ x, f x = Real.sin (ω * x + φ)) ∧
  (∀ x, f x = f (-x)) ∧
  (∃ k : ℤ, ∀ x, f (x + π) = f x) ∧
  Real.sin α + f α = 2/3 →
  (f = Real.cos) ∧
  ((Real.sqrt 2 * Real.sin (2*α - π/4) + 1) / (1 + Real.tan α) = 5/9) :=
sorry

end function_properties_l929_92906


namespace f_is_monotonic_and_odd_l929_92936

-- Define the function f(x) = -x
def f (x : ℝ) : ℝ := -x

-- State the theorem
theorem f_is_monotonic_and_odd :
  (∀ x y : ℝ, x ≤ y → f x ≤ f y) ∧ 
  (∀ x : ℝ, f (-x) = -f x) := by
  sorry


end f_is_monotonic_and_odd_l929_92936


namespace expression_evaluation_l929_92952

theorem expression_evaluation : 2 - (-3)^2 - 4 - (-5) - 6^2 - (-7) = -35 := by
  sorry

end expression_evaluation_l929_92952


namespace factorial_sum_quotient_l929_92924

theorem factorial_sum_quotient : (Nat.factorial 8 + Nat.factorial 9) / Nat.factorial 7 = 80 := by
  sorry

end factorial_sum_quotient_l929_92924


namespace min_sum_at_6_l929_92999

/-- Arithmetic sequence with given properties -/
structure ArithmeticSequence where
  a : ℕ → ℤ
  first_term : a 1 = -11
  sum_5_6 : a 5 + a 6 = -4
  is_arithmetic : ∀ n : ℕ, a (n + 1) - a n = a 2 - a 1

/-- Sum of first n terms of the arithmetic sequence -/
def S (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  (n : ℚ) / 2 * (2 * n - 24)

/-- Theorem stating that S_n reaches its minimum value when n = 6 -/
theorem min_sum_at_6 (seq : ArithmeticSequence) :
  ∀ n : ℕ, n ≠ 0 → S seq 6 ≤ S seq n :=
sorry

end min_sum_at_6_l929_92999


namespace cosine_difference_simplification_l929_92929

theorem cosine_difference_simplification (α β γ : ℝ) :
  Real.cos (α - β) * Real.cos (β - γ) - Real.sin (α - β) * Real.sin (β - γ) = Real.cos (α - γ) := by
  sorry

end cosine_difference_simplification_l929_92929


namespace unique_solution_l929_92955

def pizza_problem (boys girls : ℕ) : Prop :=
  let day1_consumption := 7 * boys + 3 * girls
  let day2_consumption := 6 * boys + 2 * girls
  (49 ≤ day1_consumption) ∧ (day1_consumption ≤ 59) ∧
  (49 ≤ day2_consumption) ∧ (day2_consumption ≤ 59)

theorem unique_solution : ∃! (b g : ℕ), pizza_problem b g ∧ b = 8 ∧ g = 1 := by
  sorry

end unique_solution_l929_92955


namespace farmer_corn_rows_l929_92959

/-- Given a farmer's crop scenario, prove the number of corn stalk rows. -/
theorem farmer_corn_rows (C : ℕ) : 
  (C * 9 + 5 * 30 = 240) → C = 10 := by
  sorry

end farmer_corn_rows_l929_92959


namespace orchard_tree_difference_l929_92948

theorem orchard_tree_difference : 
  let ahmed_orange : ℕ := 8
  let hassan_apple : ℕ := 1
  let hassan_orange : ℕ := 2
  let ahmed_apple : ℕ := 4 * hassan_apple
  let ahmed_total : ℕ := ahmed_orange + ahmed_apple
  let hassan_total : ℕ := hassan_apple + hassan_orange
  ahmed_total - hassan_total = 9 := by
sorry

end orchard_tree_difference_l929_92948


namespace probability_at_least_one_chooses_23_l929_92989

def num_students : ℕ := 4
def num_questions : ℕ := 2

theorem probability_at_least_one_chooses_23 :
  (1 : ℚ) - (1 / num_questions) ^ num_students = 15 / 16 :=
sorry

end probability_at_least_one_chooses_23_l929_92989


namespace line_tangent_to_parabola_l929_92973

/-- The line 4x + 7y + 49 = 0 is tangent to the parabola y^2 = 16x -/
theorem line_tangent_to_parabola :
  ∃! (x y : ℝ), 4 * x + 7 * y + 49 = 0 ∧ y^2 = 16 * x := by
  sorry

end line_tangent_to_parabola_l929_92973


namespace target_hit_probability_l929_92970

theorem target_hit_probability (p_a p_b : ℝ) (h_a : p_a = 0.6) (h_b : p_b = 0.5) :
  let p_hit := 1 - (1 - p_a) * (1 - p_b)
  (p_a / p_hit) = 0.75 := by
  sorry

end target_hit_probability_l929_92970


namespace add_particular_number_to_34_l929_92983

theorem add_particular_number_to_34 (x : ℝ) (h : 96 / x = 6) : 34 + x = 50 := by
  sorry

end add_particular_number_to_34_l929_92983


namespace fish_tank_weeks_l929_92927

/-- Represents the fish tank scenario -/
structure FishTank where
  initialTotal : ℕ
  dailyKoiAdded : ℕ
  dailyGoldfishAdded : ℕ
  finalKoi : ℕ
  finalGoldfish : ℕ

/-- Calculates the number of weeks fish were added to the tank -/
def weeksAdded (tank : FishTank) : ℚ :=
  let totalAdded := tank.finalKoi + tank.finalGoldfish - tank.initialTotal
  let dailyAdded := tank.dailyKoiAdded + tank.dailyGoldfishAdded
  (totalAdded : ℚ) / (dailyAdded * 7 : ℚ)

/-- Theorem stating that for the given scenario, fish were added for 3 weeks -/
theorem fish_tank_weeks (tank : FishTank) 
  (h1 : tank.initialTotal = 280)
  (h2 : tank.dailyKoiAdded = 2)
  (h3 : tank.dailyGoldfishAdded = 5)
  (h4 : tank.finalKoi = 227)
  (h5 : tank.finalGoldfish = 200) :
  weeksAdded tank = 3 := by
  sorry

end fish_tank_weeks_l929_92927


namespace quadratic_properties_l929_92994

/-- Quadratic function definition -/
def f (b : ℝ) (x : ℝ) : ℝ := x^2 + b*x + b - 1

/-- Point definition -/
structure Point where
  x : ℝ
  y : ℝ

/-- Theorem statement -/
theorem quadratic_properties (b : ℝ) :
  (∀ x, f b x = 0 ↔ x = -1 ∨ x = 1 - b) ∧
  (b < 2 → ∀ m, ∃ xp, xp = m - b + 1 ∧ 
    ∃ yp, Point.mk xp yp ∈ {p : Point | f b p.x = p.y ∧ p.x > 0 ∧ p.y > 0}) ∧
  (b = -3 → ∃ c, ∀ m n, 
    (∃ xp yp xq yq, 
      Point.mk xp yp ∈ {p : Point | f b p.x = p.y ∧ p.x > 0 ∧ p.y > 0} ∧
      Point.mk xq yq ∈ {p : Point | f b p.x = p.y ∧ p.x > 0 ∧ p.y < 0} ∧
      (yp - 0) / (xp - (-1)) = m ∧
      (yq - 0) / (xq - (-1)) = n) →
    m * n = c) :=
sorry

end quadratic_properties_l929_92994


namespace student_D_most_stable_l929_92967

/-- Represents a student in the long jump training --/
inductive Student
| A
| B
| C
| D

/-- Returns the variance of a student's performance --/
def variance (s : Student) : ℝ :=
  match s with
  | Student.A => 2.1
  | Student.B => 3.5
  | Student.C => 9
  | Student.D => 0.7

/-- Determines if a student has the most stable performance --/
def has_most_stable_performance (s : Student) : Prop :=
  ∀ t : Student, variance s ≤ variance t

/-- Theorem stating that student D has the most stable performance --/
theorem student_D_most_stable :
  has_most_stable_performance Student.D :=
sorry

end student_D_most_stable_l929_92967


namespace complex_sum_cube_ratio_l929_92925

theorem complex_sum_cube_ratio (x y z : ℂ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0)
  (h_sum : x + y + z = 30)
  (h_sq_diff : (x - y)^2 + (x - z)^2 + (y - z)^2 = 2*x*y*z) :
  (x^3 + y^3 + z^3) / (x*y*z) = 33 := by
sorry

end complex_sum_cube_ratio_l929_92925


namespace triangle_ratio_theorem_l929_92991

/-- Given a triangle ABC with points E on BC and G on AB, and Q the intersection of AE and CG,
    if AQ:QE = 3:2 and GQ:QC = 2:3, then AG:GB = 1:2 -/
theorem triangle_ratio_theorem (A B C E G Q : ℝ × ℝ) : 
  (E.1 - B.1) / (C.1 - B.1) = (E.2 - B.2) / (C.2 - B.2) →  -- E is on BC
  (G.1 - A.1) / (B.1 - A.1) = (G.2 - A.2) / (B.2 - A.2) →  -- G is on AB
  ∃ (t : ℝ), Q = (1 - t) • A + t • E ∧                     -- Q is on AE
             Q = (1 - t) • C + t • G →                     -- Q is on CG
  (Q.1 - A.1) / (E.1 - Q.1) = 3 / 2 →                      -- AQ:QE = 3:2
  (G.1 - Q.1) / (Q.1 - C.1) = 2 / 3 →                      -- GQ:QC = 2:3
  (G.1 - A.1) / (B.1 - G.1) = 1 / 2 :=                     -- AG:GB = 1:2
by sorry


end triangle_ratio_theorem_l929_92991


namespace cd_length_l929_92969

/-- Represents a point in 2D space -/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/-- Represents a quadrilateral -/
structure Quadrilateral :=
  (A : Point)
  (B : Point)
  (N : Point)
  (M : Point)

/-- Represents the problem setup -/
structure ProblemSetup :=
  (ABNM : Quadrilateral)
  (C : Point)
  (A' : Point)
  (D : Point)
  (x : ℝ)
  (AB : ℝ)
  (AM : ℝ)
  (AC : ℝ)

/-- Main theorem: CD = AC * cos(x) -/
theorem cd_length
  (setup : ProblemSetup)
  (h1 : setup.ABNM.A.y = setup.ABNM.B.y) -- AB is initially horizontal
  (h2 : setup.ABNM.N.y = setup.ABNM.M.y) -- MN is horizontal
  (h3 : setup.C.y = setup.ABNM.M.y) -- C is on line MN
  (h4 : setup.A'.x - setup.ABNM.B.x = setup.AB * Real.cos setup.x) -- A' position after rotation
  (h5 : setup.A'.y - setup.ABNM.B.y = setup.AB * Real.sin setup.x)
  (h6 : Real.sqrt ((setup.A'.x - setup.ABNM.B.x)^2 + (setup.A'.y - setup.ABNM.B.y)^2) = setup.AB) -- A'B = AB
  : Real.sqrt ((setup.D.x - setup.C.x)^2 + (setup.D.y - setup.C.y)^2) = setup.AC * Real.cos setup.x :=
sorry

end cd_length_l929_92969


namespace reflection_across_x_axis_l929_92943

/-- Represents a point in a 2D Cartesian coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- Reflects a point across the x-axis -/
def reflectAcrossXAxis (p : Point) : Point :=
  { x := p.x, y := -p.y }

theorem reflection_across_x_axis :
  let A : Point := { x := 2, y := 3 }
  reflectAcrossXAxis A = { x := 2, y := -3 } := by
  sorry

end reflection_across_x_axis_l929_92943


namespace arithmetic_sequence_diff_l929_92918

def arithmetic_sequence (a₁ : ℤ) (d : ℤ) (n : ℕ) : ℤ := a₁ + (n - 1) * d

theorem arithmetic_sequence_diff (a₁ d : ℤ) :
  |arithmetic_sequence a₁ d 105 - arithmetic_sequence a₁ d 100| = 40 :=
by sorry

end arithmetic_sequence_diff_l929_92918


namespace C2H6_C3H8_impossible_l929_92965

-- Define the heat released by combustion of 1 mol of each hydrocarbon
def heat_CH4 : ℝ := 889.5
def heat_C2H6 : ℝ := 1558.35
def heat_C2H4 : ℝ := 1409.6
def heat_C2H2 : ℝ := 1298.35
def heat_C3H8 : ℝ := 2217.8

-- Define the total heat released by the mixture
def total_heat : ℝ := 3037.6

-- Define the number of moles in the mixture
def total_moles : ℝ := 2

-- Theorem to prove that C₂H₆ and C₃H₈ combination is impossible
theorem C2H6_C3H8_impossible : 
  ¬(∃ (x y : ℝ), x + y = total_moles ∧ 
                  x * heat_C2H6 + y * heat_C3H8 = total_heat ∧
                  x > 0 ∧ y > 0) :=
by sorry

end C2H6_C3H8_impossible_l929_92965


namespace daily_rate_proof_l929_92961

/-- The daily rental rate for Jason's carriage house -/
def daily_rate : ℝ := 40

/-- The total cost for Eric's rental -/
def total_cost : ℝ := 800

/-- The number of days Eric is renting -/
def rental_days : ℕ := 20

/-- Theorem stating that the daily rate multiplied by the number of rental days equals the total cost -/
theorem daily_rate_proof : daily_rate * (rental_days : ℝ) = total_cost := by
  sorry

end daily_rate_proof_l929_92961


namespace frog_jump_theorem_l929_92913

/-- A regular polygon with 2n sides inscribed in a circle -/
structure RegularPolygon (n : ℕ) :=
  (n_ge_two : n ≥ 2)

/-- A configuration of frogs on the vertices of a regular polygon -/
structure FrogConfiguration (n : ℕ) :=
  (polygon : RegularPolygon n)
  (frogs : Fin (2*n) → Bool)

/-- A jumping method for the frogs -/
def JumpingMethod (n : ℕ) := Fin (2*n) → Bool

/-- Check if a line segment passes through the center of the circle -/
def passes_through_center (n : ℕ) (v1 v2 : Fin (2*n)) : Prop :=
  ∃ k : ℕ, v2 = v1 + n ∨ v1 = v2 + n

/-- The main theorem -/
theorem frog_jump_theorem (n : ℕ) :
  (∃ (config : FrogConfiguration n) (jump : JumpingMethod n),
    ∀ v1 v2 : Fin (2*n),
      v1 ≠ v2 →
      config.frogs v1 = true →
      config.frogs v2 = true →
      ¬passes_through_center n v1 v2) ↔
  n % 4 = 2 :=
sorry

end frog_jump_theorem_l929_92913


namespace mary_savings_problem_l929_92944

theorem mary_savings_problem (S : ℝ) (x : ℝ) (h1 : S > 0) (h2 : 0 ≤ x ∧ x ≤ 1) : 
  12 * x * S = 7 * (1 - x) * S → (1 - x) = 12 / 19 := by
  sorry

end mary_savings_problem_l929_92944


namespace third_day_is_tuesday_or_wednesday_l929_92976

/-- Represents the days of the week -/
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- Represents a month with its properties -/
structure Month where
  days : ℕ
  startDay : DayOfWeek
  mondayCount : ℕ
  tuesdayCount : ℕ
  wednesdayCount : ℕ
  sundayCount : ℕ

/-- Given the properties of a month, determine the day of the week for the 3rd day -/
def thirdDayOfMonth (m : Month) : DayOfWeek :=
  sorry

/-- Theorem stating that the 3rd day of the month is either Tuesday or Wednesday -/
theorem third_day_is_tuesday_or_wednesday (m : Month) 
  (h1 : m.mondayCount = m.wednesdayCount + 1)
  (h2 : m.tuesdayCount = m.sundayCount) :
  (thirdDayOfMonth m = DayOfWeek.Tuesday) ∨ (thirdDayOfMonth m = DayOfWeek.Wednesday) :=
  sorry

end third_day_is_tuesday_or_wednesday_l929_92976


namespace intersection_complement_equality_l929_92993

def U : Set Int := {-4, -2, -1, 0, 2, 4, 5, 6, 7}
def A : Set Int := {-2, 0, 4, 6}
def B : Set Int := {-1, 2, 4, 6, 7}

theorem intersection_complement_equality : A ∩ (U \ B) = {-2, 0} := by sorry

end intersection_complement_equality_l929_92993


namespace power_2020_l929_92956

theorem power_2020 (m n : ℕ) (h1 : 3^m = 4) (h2 : 3^(m-4*n) = 4/81) : 
  2020^n = 2020 := by
  sorry

end power_2020_l929_92956


namespace quadratic_root_condition_l929_92921

/-- Given a quadratic equation x^2 + (m - 3)x + m = 0 where m is a real number,
    if one root is greater than 1 and the other root is less than 1,
    then m < 1 -/
theorem quadratic_root_condition (m : ℝ) : 
  (∃ r₁ r₂ : ℝ, r₁ > 1 ∧ r₂ < 1 ∧ 
    r₁^2 + (m - 3) * r₁ + m = 0 ∧ 
    r₂^2 + (m - 3) * r₂ + m = 0) → 
  m < 1 := by
sorry


end quadratic_root_condition_l929_92921


namespace compound_ratio_proof_l929_92908

theorem compound_ratio_proof (x y : ℝ) (y_nonzero : y ≠ 0) :
  (2 / 3) * (6 / 7) * (1 / 3) * (3 / 8) * (4 / 5) * (x / y) = x / (17.5 * y) :=
by sorry

end compound_ratio_proof_l929_92908


namespace xyz_sum_l929_92919

theorem xyz_sum (x y z : ℕ+) 
  (h1 : x.val * y.val + z.val = 47)
  (h2 : y.val * z.val + x.val = 47)
  (h3 : x.val * z.val + y.val = 47) :
  x.val + y.val + z.val = 48 :=
by sorry

end xyz_sum_l929_92919


namespace closest_to_fraction_l929_92935

def options : List ℝ := [0.3, 3, 30, 300, 3000]

theorem closest_to_fraction (x : ℝ) (h : x = 613 / 0.307) :
  ∃ y ∈ options, ∀ z ∈ options, |x - y| ≤ |x - z| :=
sorry

end closest_to_fraction_l929_92935


namespace cube_root_of_product_l929_92968

theorem cube_root_of_product (a : ℕ) : a^3 = 21 * 35 * 45 * 35 → a = 105 := by
  sorry

end cube_root_of_product_l929_92968


namespace max_airlines_with_both_amenities_is_zero_l929_92905

/-- Represents a type of plane -/
inductive PlaneType
| A
| B

/-- Represents whether a plane has both amenities -/
def has_both_amenities : PlaneType → Bool
| PlaneType.A => true
| PlaneType.B => false

/-- Represents a fleet composition -/
structure FleetComposition :=
  (type_a_percent : ℚ)
  (type_b_percent : ℚ)
  (sum_to_one : type_a_percent + type_b_percent = 1)
  (valid_range : 0.1 ≤ type_a_percent ∧ type_a_percent ≤ 0.9)

/-- Minimum number of planes in a fleet -/
def min_fleet_size : ℕ := 5

/-- Theorem: The maximum percentage of airlines offering both amenities on all planes is 0% -/
theorem max_airlines_with_both_amenities_is_zero :
  ∀ (fc : FleetComposition),
    ¬(∀ (plane : PlaneType), has_both_amenities plane = true) :=
by sorry

end max_airlines_with_both_amenities_is_zero_l929_92905


namespace log_equality_condition_l929_92930

theorem log_equality_condition (p q : ℝ) (hp : p > 0) (hq : q > 0) (hq2 : q ≠ 2) :
  Real.log p + Real.log q = Real.log (2 * p + 3 * q) ↔ p = (3 * q) / (q - 2) :=
sorry

end log_equality_condition_l929_92930


namespace magnitude_of_AD_is_two_l929_92945

/-- Given two plane vectors m and n, prove that the magnitude of AD is 2 -/
theorem magnitude_of_AD_is_two (m n : ℝ × ℝ) : 
  let angle := Real.pi / 6
  let norm_m := Real.sqrt 3
  let norm_n := 2
  let AB := (2 * m.1 + 2 * n.1, 2 * m.2 + 2 * n.2)
  let AC := (2 * m.1 - 6 * n.1, 2 * m.2 - 6 * n.2)
  let D := ((AB.1 + AC.1) / 2, (AB.2 + AC.2) / 2)  -- midpoint of BC
  let AD := (D.1 - m.1, D.2 - m.2)
  Real.cos angle = Real.sqrt 3 / 2 →   -- angle between m and n
  norm_m = Real.sqrt 3 →
  norm_n = 2 →
  Real.sqrt (AD.1 ^ 2 + AD.2 ^ 2) = 2 :=
by sorry

end magnitude_of_AD_is_two_l929_92945


namespace geometric_parallelism_l929_92933

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the "contained in" relation between a line and a plane
variable (contained_in : Line → Plane → Prop)

-- Define the parallel relation between planes
variable (plane_parallel : Plane → Plane → Prop)

-- Define the parallel relation between a line and a plane
variable (line_plane_parallel : Line → Plane → Prop)

-- State the theorem
theorem geometric_parallelism 
  (a : Line) (α β : Plane) (h : contained_in a α) :
  (plane_parallel α β → line_plane_parallel a β) ∧
  (¬ line_plane_parallel a β → ¬ plane_parallel α β) ∧
  ¬ (line_plane_parallel a β → plane_parallel α β) :=
sorry

end geometric_parallelism_l929_92933


namespace expression_evaluation_l929_92986

/-- Given x = -2 and y = 1/2, prove that 2(x^2y + xy^2) - 2(x^2y - 1) - 3xy^2 - 2 evaluates to 1/2 -/
theorem expression_evaluation (x y : ℝ) (hx : x = -2) (hy : y = 1/2) :
  2 * (x^2 * y + x * y^2) - 2 * (x^2 * y - 1) - 3 * x * y^2 - 2 = 1/2 := by
  sorry

end expression_evaluation_l929_92986


namespace sufficient_not_necessary_l929_92992

theorem sufficient_not_necessary (x y : ℝ) : 
  (∀ x y, x < y ∧ y < 0 → x^2 > y^2) ∧ 
  (∃ x y, x^2 > y^2 ∧ ¬(x < y ∧ y < 0)) :=
by sorry

end sufficient_not_necessary_l929_92992


namespace two_red_cards_selection_count_l929_92928

/-- Represents a deck of cards with a specific structure -/
structure Deck :=
  (total_cards : ℕ)
  (suits : ℕ)
  (cards_per_suit : ℕ)
  (red_suits : ℕ)
  (black_suits : ℕ)

/-- Calculates the number of ways to select two different cards from red suits -/
def select_two_red_cards (d : Deck) : ℕ :=
  let red_cards := d.red_suits * d.cards_per_suit
  red_cards * (red_cards - 1)

/-- The main theorem to be proved -/
theorem two_red_cards_selection_count (d : Deck) 
  (h1 : d.total_cards = 36)
  (h2 : d.suits = 3)
  (h3 : d.cards_per_suit = 12)
  (h4 : d.red_suits = 2)
  (h5 : d.black_suits = 1)
  (h6 : d.red_suits + d.black_suits = d.suits) :
  select_two_red_cards d = 552 := by
  sorry


end two_red_cards_selection_count_l929_92928


namespace square_angles_equal_l929_92975

-- Define a rectangle
structure Rectangle where
  angles : Fin 4 → ℝ

-- Define a square as a special case of rectangle
structure Square extends Rectangle

-- State that all angles in a rectangle are equal
axiom rectangle_angles_equal (r : Rectangle) : ∀ i j : Fin 4, r.angles i = r.angles j

-- State that a square is a rectangle
axiom square_is_rectangle (s : Square) : Rectangle

-- Theorem to prove
theorem square_angles_equal (s : Square) : ∀ i j : Fin 4, s.angles i = s.angles j := by
  sorry

end square_angles_equal_l929_92975


namespace decimal_to_fraction_l929_92966

theorem decimal_to_fraction :
  (2.35 : ℚ) = 47 / 20 := by sorry

end decimal_to_fraction_l929_92966


namespace condition_A_sufficient_not_necessary_l929_92985

/-- Condition A: a > 1 and b > 1 -/
def condition_A (a b : ℝ) : Prop := a > 1 ∧ b > 1

/-- Condition B: a + b > 2 and ab > 1 -/
def condition_B (a b : ℝ) : Prop := a + b > 2 ∧ a * b > 1

theorem condition_A_sufficient_not_necessary :
  (∀ a b : ℝ, condition_A a b → condition_B a b) ∧
  (∃ a b : ℝ, condition_B a b ∧ ¬condition_A a b) :=
by sorry

end condition_A_sufficient_not_necessary_l929_92985


namespace derivative_reciprocal_sum_sqrt_derivative_reciprocal_sum_sqrt_value_l929_92923

theorem derivative_reciprocal_sum_sqrt (x : ℝ) (h : x ≠ 1) :
  (fun x => 2 / (1 - x)) = (fun x => 1 / (1 - Real.sqrt x) + 1 / (1 + Real.sqrt x)) :=
by sorry

theorem derivative_reciprocal_sum_sqrt_value (x : ℝ) (h : x ≠ 1) :
  deriv (fun x => 2 / (1 - x)) x = 2 / (1 - x)^2 :=
by sorry

end derivative_reciprocal_sum_sqrt_derivative_reciprocal_sum_sqrt_value_l929_92923


namespace nine_distinct_values_of_z_l929_92900

/-- Given two integers x and y between 100 and 999 inclusive, where y is formed by swapping
    the hundreds and tens digits of x (units digit remains the same), prove that the absolute
    difference z = |x - y| can have exactly 9 distinct values. -/
theorem nine_distinct_values_of_z (x y : ℤ) (z : ℕ) :
  (100 ≤ x ∧ x ≤ 999) →
  (100 ≤ y ∧ y ≤ 999) →
  (∃ a b c : ℕ, x = 100 * a + 10 * b + c ∧ y = 10 * a + 100 * b + c ∧ 1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ 0 ≤ c ∧ c ≤ 9) →
  z = |x - y| →
  ∃ (S : Finset ℕ), S.card = 9 ∧ z ∈ S ∧ ∀ w ∈ S, ∃ k : ℕ, w = 90 * k ∧ k ≤ 8 :=
sorry

#check nine_distinct_values_of_z

end nine_distinct_values_of_z_l929_92900


namespace power_inequality_condition_l929_92941

theorem power_inequality_condition (n : ℤ) : n ∈ ({-2, -1, 0, 1, 2, 3} : Set ℤ) →
  ((-1/2 : ℚ)^n > (-1/5 : ℚ)^n ↔ n = -1 ∨ n = 2) := by sorry

end power_inequality_condition_l929_92941


namespace linear_function_iteration_l929_92901

/-- Given a linear function f and its iterations, prove that ab = 6 -/
theorem linear_function_iteration (a b : ℝ) : 
  let f : ℝ → ℝ := λ x ↦ a * x + b
  let f₁ : ℝ → ℝ := f
  let f₂ : ℝ → ℝ := λ x ↦ f (f₁ x)
  let f₃ : ℝ → ℝ := λ x ↦ f (f₂ x)
  let f₄ : ℝ → ℝ := λ x ↦ f (f₃ x)
  let f₅ : ℝ → ℝ := λ x ↦ f (f₄ x)
  (∀ x, f₅ x = 32 * x + 93) → a * b = 6 := by
sorry

end linear_function_iteration_l929_92901


namespace probability_ratio_equals_ways_ratio_l929_92910

def number_of_balls : ℕ := 20
def number_of_bins : ℕ := 5

def distribution_p : List ℕ := [3, 6, 4, 4, 4]
def distribution_q : List ℕ := [4, 4, 4, 4, 4]

def ways_to_distribute (dist : List ℕ) : ℕ :=
  sorry

theorem probability_ratio_equals_ways_ratio :
  let p := (ways_to_distribute distribution_p : ℚ) / number_of_balls ^ number_of_bins
  let q := (ways_to_distribute distribution_q : ℚ) / number_of_balls ^ number_of_bins
  p / q = (ways_to_distribute distribution_p : ℚ) / (ways_to_distribute distribution_q) :=
by sorry

end probability_ratio_equals_ways_ratio_l929_92910


namespace fruit_basket_cost_l929_92988

/-- Represents the contents of a fruit basket --/
structure FruitBasket where
  bananas : ℕ
  apples : ℕ
  oranges : ℕ
  kiwis : ℕ
  strawberries : ℕ
  avocados : ℕ
  grapes : ℕ
  melons : ℕ

/-- Represents the prices of individual fruits --/
structure FruitPrices where
  banana : ℚ
  apple : ℚ
  orange : ℚ
  kiwi : ℚ
  strawberry_dozen : ℚ
  avocado : ℚ
  grapes_half_bunch : ℚ
  melon : ℚ

/-- Calculates the total cost of the fruit basket after all discounts --/
def calculateTotalCost (basket : FruitBasket) (prices : FruitPrices) : ℚ :=
  sorry

/-- Theorem stating that the total cost of the given fruit basket is $35.43 --/
theorem fruit_basket_cost :
  let basket : FruitBasket := {
    bananas := 4,
    apples := 3,
    oranges := 4,
    kiwis := 2,
    strawberries := 24,
    avocados := 2,
    grapes := 1,
    melons := 1
  }
  let prices : FruitPrices := {
    banana := 1,
    apple := 2,
    orange := 3/2,
    kiwi := 5/4,
    strawberry_dozen := 4,
    avocado := 3,
    grapes_half_bunch := 2,
    melon := 7/2
  }
  calculateTotalCost basket prices = 3543/100 :=
sorry

end fruit_basket_cost_l929_92988


namespace half_angle_quadrant_l929_92960

def is_first_quadrant (α : Real) : Prop :=
  ∃ k : ℤ, 2 * k * Real.pi < α ∧ α < (Real.pi / 2) + 2 * k * Real.pi

def is_first_or_third_quadrant (α : Real) : Prop :=
  ∃ k : ℤ, (2 * k * Real.pi < α ∧ α < (Real.pi / 2) + 2 * k * Real.pi) ∨
           ((Real.pi + 2 * k * Real.pi < α) ∧ (α < (3 * Real.pi / 2) + 2 * k * Real.pi))

theorem half_angle_quadrant (α : Real) :
  is_first_quadrant α → is_first_or_third_quadrant (α / 2) := by
  sorry

end half_angle_quadrant_l929_92960


namespace find_x_value_l929_92954

theorem find_x_value (x : ℝ) : (15 : ℝ)^x * 8^3 / 256 = 450 → x = 2 := by
  sorry

end find_x_value_l929_92954


namespace set_equality_l929_92974

def P : Set ℕ := {1, 2}
def Q : Set ℕ := {2, 3}
def M : Set ℕ := {x | x ∈ P ∧ x ∉ Q}

theorem set_equality : M = {1} := by sorry

end set_equality_l929_92974


namespace cash_drawer_value_l929_92939

/-- Calculates the total value of bills in a cash drawer given the total number of bills,
    the number of 5-dollar bills, and assuming the rest are 20-dollar bills. -/
def total_value (total_bills : ℕ) (five_dollar_bills : ℕ) : ℕ :=
  let twenty_dollar_bills := total_bills - five_dollar_bills
  5 * five_dollar_bills + 20 * twenty_dollar_bills

/-- Theorem stating that given 54 bills in total with 20 5-dollar bills,
    the total value is $780. -/
theorem cash_drawer_value :
  total_value 54 20 = 780 := by
  sorry

#eval total_value 54 20  -- Should output 780

end cash_drawer_value_l929_92939


namespace storage_unit_paint_area_l929_92911

/-- Represents a rectangular storage unit with windows --/
structure StorageUnit where
  length : ℝ
  width : ℝ
  height : ℝ
  windowCount : ℕ
  windowLength : ℝ
  windowWidth : ℝ

/-- Calculates the total area to be painted in the storage unit --/
def totalPaintArea (unit : StorageUnit) : ℝ :=
  let wallArea := 2 * (unit.length * unit.height + unit.width * unit.height)
  let ceilingArea := unit.length * unit.width
  let windowArea := unit.windowCount * (unit.windowLength * unit.windowWidth)
  wallArea + ceilingArea - windowArea

/-- Theorem stating that the total paint area for the given storage unit is 1020 square yards --/
theorem storage_unit_paint_area :
  let unit : StorageUnit := {
    length := 15,
    width := 12,
    height := 8,
    windowCount := 2,
    windowLength := 3,
    windowWidth := 4
  }
  totalPaintArea unit = 1020 := by sorry

end storage_unit_paint_area_l929_92911


namespace total_frogs_caught_l929_92922

def initial_frogs : ℕ := 5
def additional_frogs : ℕ := 2

theorem total_frogs_caught :
  initial_frogs + additional_frogs = 7 := by sorry

end total_frogs_caught_l929_92922


namespace amethyst_bead_count_l929_92964

/-- Proves the number of amethyst beads in a specific necklace configuration -/
theorem amethyst_bead_count (total : ℕ) (turquoise : ℕ) (amethyst : ℕ) : 
  total = 40 → 
  turquoise = 19 → 
  total = amethyst + 2 * amethyst + turquoise → 
  amethyst = 7 := by
  sorry

end amethyst_bead_count_l929_92964


namespace fraction_equality_implies_c_geq_one_l929_92915

theorem fraction_equality_implies_c_geq_one
  (a b : ℕ+) (c : ℝ)
  (h_c_pos : c > 0)
  (h_eq : (a + 1 : ℝ) / (b + c) = (b : ℝ) / a) :
  c ≥ 1 :=
sorry

end fraction_equality_implies_c_geq_one_l929_92915


namespace ceiling_floor_sum_l929_92940

theorem ceiling_floor_sum : ⌈(7:ℚ)/3⌉ + ⌊-(7:ℚ)/3⌋ = 0 := by sorry

end ceiling_floor_sum_l929_92940


namespace intersection_of_M_and_N_l929_92981

def M : Set ℕ := {1, 2, 3}
def N : Set ℕ := {2, 3, 4}

theorem intersection_of_M_and_N : M ∩ N = {2, 3} := by sorry

end intersection_of_M_and_N_l929_92981


namespace gcd_sum_and_count_even_integers_l929_92990

def sum_even_integers (a b : ℕ) : ℕ :=
  let first_even := if a % 2 = 0 then a else a + 1
  let last_even := if b % 2 = 0 then b else b - 1
  let n := (last_even - first_even) / 2 + 1
  n * (first_even + last_even) / 2

def count_even_integers (a b : ℕ) : ℕ :=
  let first_even := if a % 2 = 0 then a else a + 1
  let last_even := if b % 2 = 0 then b else b - 1
  (last_even - first_even) / 2 + 1

theorem gcd_sum_and_count_even_integers :
  Nat.gcd (sum_even_integers 13 63) (count_even_integers 13 63) = 25 := by
  sorry

end gcd_sum_and_count_even_integers_l929_92990


namespace dream_car_gas_consumption_l929_92971

/-- Calculates the total gas consumption for a car over two days -/
def total_gas_consumption (consumption_rate : ℝ) (miles_today : ℝ) (miles_tomorrow : ℝ) : ℝ :=
  consumption_rate * (miles_today + miles_tomorrow)

theorem dream_car_gas_consumption :
  let consumption_rate : ℝ := 4
  let miles_today : ℝ := 400
  let miles_tomorrow : ℝ := miles_today + 200
  total_gas_consumption consumption_rate miles_today miles_tomorrow = 4000 := by
sorry

end dream_car_gas_consumption_l929_92971


namespace cos_two_pi_thirds_minus_two_alpha_l929_92984

theorem cos_two_pi_thirds_minus_two_alpha (α : Real) 
  (h : Real.sin (α + π / 6) = 1 / 3) : 
  Real.cos ((2 * π) / 3 - 2 * α) = -7 / 9 := by
  sorry

end cos_two_pi_thirds_minus_two_alpha_l929_92984


namespace circle_intersection_existence_l929_92995

theorem circle_intersection_existence :
  ∃ n : ℝ, 0 < n ∧ n < 2 ∧
  (∃ x y : ℝ, x^2 + y^2 - 2*n*x + 2*n*y + 2*n^2 - 8 = 0 ∧
              (x+1)^2 + (y-1)^2 = 2) ∧
  (∃ x' y' : ℝ, x' ≠ x ∨ y' ≠ y ∧
              x'^2 + y'^2 - 2*n*x' + 2*n*y' + 2*n^2 - 8 = 0 ∧
              (x'+1)^2 + (y'-1)^2 = 2) :=
by sorry

end circle_intersection_existence_l929_92995


namespace sequence_properties_l929_92950

def sequence_a (n : ℕ+) : ℚ := sorry

def S (n : ℕ+) : ℚ := sorry

def T (n : ℕ+) : ℚ := sorry

theorem sequence_properties :
  (∀ n : ℕ+, 3 * S n = (n + 2) * sequence_a n) ∧
  sequence_a 1 = 2 →
  (∀ n : ℕ+, sequence_a n = n + 1) ∧
  ∃ M : Set ℕ+, Set.Infinite M ∧ ∀ n ∈ M, |T n - 1| < (1 : ℚ) / 10 := by
  sorry

end sequence_properties_l929_92950


namespace cross_area_is_two_l929_92932

/-- Represents a point in a 2D grid --/
structure GridPoint where
  x : ℚ
  y : ℚ

/-- Represents a triangle in the grid --/
structure Triangle where
  v1 : GridPoint
  v2 : GridPoint
  v3 : GridPoint

/-- The center point of the 4x4 grid --/
def gridCenter : GridPoint := { x := 2, y := 2 }

/-- A function to create a midpoint on the grid edge --/
def gridEdgeMidpoint (x y : ℚ) : GridPoint := { x := x, y := y }

/-- The four triangles forming the cross shape --/
def crossTriangles : List Triangle := [
  { v1 := gridCenter, v2 := gridEdgeMidpoint 0 2, v3 := gridEdgeMidpoint 2 0 },
  { v1 := gridCenter, v2 := gridEdgeMidpoint 2 4, v3 := gridEdgeMidpoint 4 2 },
  { v1 := gridCenter, v2 := gridEdgeMidpoint 0 2, v3 := gridEdgeMidpoint 2 4 },
  { v1 := gridCenter, v2 := gridEdgeMidpoint 2 0, v3 := gridEdgeMidpoint 4 2 }
]

/-- Calculate the area of a single triangle --/
def triangleArea (t : Triangle) : ℚ := 0.5

/-- Calculate the total area of the cross shape --/
def crossArea : ℚ := (crossTriangles.map triangleArea).sum

/-- The theorem stating that the area of the cross shape is 2 --/
theorem cross_area_is_two : crossArea = 2 := by sorry

end cross_area_is_two_l929_92932


namespace equation_solution_l929_92937

theorem equation_solution : 
  ∃ x : ℚ, (x + 10) / (x - 4) = (x - 3) / (x + 6) ↔ x = -48 / 23 :=
by
  sorry

end equation_solution_l929_92937


namespace max_sum_given_constraints_l929_92963

theorem max_sum_given_constraints (x y : ℝ) (h1 : x^2 + y^2 = 100) (h2 : x * y = 40) :
  x + y ≤ 6 * Real.sqrt 5 := by
  sorry

end max_sum_given_constraints_l929_92963


namespace three_digit_number_proof_l929_92947

theorem three_digit_number_proof :
  ∃! n : ℕ, 100 ≤ n ∧ n < 1000 ∧ n / 100 = 1 ∧
  (n % 100 * 10 + 1) - n = 9 * (10 : ℝ) ∧ n = 121 := by
  sorry

end three_digit_number_proof_l929_92947


namespace count_is_nine_l929_92912

/-- A function that returns the count of valid 4-digit numbers greater than 1000 
    that can be formed using the digits of 2012 -/
def count_valid_numbers : ℕ :=
  -- Define the function here
  sorry

/-- Theorem stating that the count of valid numbers is 9 -/
theorem count_is_nine : count_valid_numbers = 9 := by
  sorry

end count_is_nine_l929_92912
