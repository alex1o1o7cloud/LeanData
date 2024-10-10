import Mathlib

namespace marble_ratio_l289_28998

def dans_marbles : ℕ := 5
def marys_marbles : ℕ := 10

theorem marble_ratio : 
  (marys_marbles : ℚ) / (dans_marbles : ℚ) = 2 := by
  sorry

end marble_ratio_l289_28998


namespace collinear_opposite_vectors_l289_28937

/-- Two vectors are collinear if one is a scalar multiple of the other -/
def collinear (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, a = k • b

/-- Two vectors have opposite directions if their scalar multiple is negative -/
def opposite_directions (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, k < 0 ∧ a = k • b

theorem collinear_opposite_vectors (m : ℝ) :
  let a : ℝ × ℝ := (m, 1)
  let b : ℝ × ℝ := (1, m)
  collinear a b ∧ opposite_directions a b → m = -1 :=
by sorry

end collinear_opposite_vectors_l289_28937


namespace apple_pear_equivalence_l289_28929

theorem apple_pear_equivalence (apple_value pear_value : ℚ) :
  (3 / 4 : ℚ) * 12 * apple_value = 10 * pear_value →
  (3 / 5 : ℚ) * 15 * apple_value = 10 * pear_value :=
by
  sorry

end apple_pear_equivalence_l289_28929


namespace weight_of_doubled_cube_l289_28997

/-- Given two cubes of the same material, if the second cube has sides twice as long
    as the first cube, and the first cube weighs 4 pounds, then the second cube weighs 32 pounds. -/
theorem weight_of_doubled_cube (s : ℝ) (weight_first : ℝ) (volume_first : ℝ) (weight_second : ℝ) :
  s > 0 →
  weight_first = 4 →
  volume_first = s^3 →
  weight_first / volume_first = weight_second / ((2*s)^3) →
  weight_second = 32 := by
  sorry

end weight_of_doubled_cube_l289_28997


namespace combination_equality_implies_seven_l289_28953

theorem combination_equality_implies_seven (n : ℕ) : 
  (n.choose 3) = ((n-1).choose 3) + ((n-1).choose 4) → n = 7 := by
  sorry

end combination_equality_implies_seven_l289_28953


namespace green_marbles_taken_l289_28996

theorem green_marbles_taken (initial_green : ℝ) (remaining_green : ℝ) 
  (h1 : initial_green = 32.0)
  (h2 : remaining_green = 9.0) :
  initial_green - remaining_green = 23.0 := by
  sorry

end green_marbles_taken_l289_28996


namespace democrat_count_l289_28968

theorem democrat_count (total : ℕ) (difference : ℕ) (h1 : total = 434) (h2 : difference = 30) :
  let democrats := (total - difference) / 2
  democrats = 202 := by
sorry

end democrat_count_l289_28968


namespace smoking_rate_estimate_l289_28958

/-- Represents the survey results and conditions -/
structure SurveyData where
  total_students : ℕ
  yes_answers : ℕ
  die_prob : ℚ

/-- Calculates the estimated smoking rate based on survey data -/
def estimate_smoking_rate (data : SurveyData) : ℚ :=
  let estimated_smokers := data.yes_answers / 2
  (estimated_smokers : ℚ) / data.total_students

/-- Theorem stating the estimated smoking rate for the given survey data -/
theorem smoking_rate_estimate (data : SurveyData) 
  (h1 : data.total_students = 300)
  (h2 : data.yes_answers = 80)
  (h3 : data.die_prob = 1/2) :
  ∃ (ε : ℚ), abs (estimate_smoking_rate data - 40/300) < ε ∧ ε < 1/1000 := by
  sorry

end smoking_rate_estimate_l289_28958


namespace sum_of_integers_l289_28955

theorem sum_of_integers (a b : ℕ+) : a^2 - b^2 = 52 → a * b = 168 → a + b = 26 := by
  sorry

end sum_of_integers_l289_28955


namespace orange_harvest_difference_l289_28950

theorem orange_harvest_difference (ripe_sacks unripe_sacks : ℕ) 
  (h1 : ripe_sacks = 44) 
  (h2 : unripe_sacks = 25) : 
  ripe_sacks - unripe_sacks = 19 := by
  sorry

end orange_harvest_difference_l289_28950


namespace boatman_distance_against_current_l289_28976

/-- Represents the speed of a boat in different water conditions -/
structure BoatSpeed where
  stillWater : ℝ
  current : ℝ

/-- Calculates the distance traveled given speed and time -/
def distanceTraveled (speed time : ℝ) : ℝ := speed * time

/-- Represents the problem of a boatman traveling in a stream -/
theorem boatman_distance_against_current 
  (boat : BoatSpeed)
  (h1 : distanceTraveled (boat.stillWater + boat.current) (1/3) = 1)
  (h2 : distanceTraveled boat.stillWater 3 = 6)
  (h3 : boat.stillWater > boat.current)
  (h4 : boat.current > 0) :
  distanceTraveled (boat.stillWater - boat.current) 4 = 4 := by
  sorry

end boatman_distance_against_current_l289_28976


namespace red_area_after_four_changes_l289_28926

/-- Represents the fraction of red area remaining after one execution of the process -/
def remaining_fraction : ℚ := 8 / 9

/-- The number of times the process is executed -/
def process_iterations : ℕ := 4

/-- Calculates the fraction of the original area that remains red after n iterations -/
def red_area_fraction (n : ℕ) : ℚ := remaining_fraction ^ n

theorem red_area_after_four_changes :
  red_area_fraction process_iterations = 4096 / 6561 := by
  sorry

end red_area_after_four_changes_l289_28926


namespace line_intersection_l289_28927

theorem line_intersection :
  ∀ (x y : ℚ),
  (12 * x - 3 * y = 33) →
  (8 * x + 2 * y = 18) →
  (x = 29/12 ∧ y = -2/3) :=
by
  sorry

end line_intersection_l289_28927


namespace basketball_handshakes_l289_28972

theorem basketball_handshakes :
  let team_size : ℕ := 6
  let num_teams : ℕ := 2
  let num_referees : ℕ := 3
  let player_handshakes := team_size * team_size
  let referee_handshakes := (team_size * num_teams) * num_referees
  player_handshakes + referee_handshakes = 72 := by
sorry

end basketball_handshakes_l289_28972


namespace inequality_solution_set_l289_28946

theorem inequality_solution_set (a : ℝ) :
  let S := {x : ℝ | (x - 1) * (x - 2*a + 1) < 0}
  (a = 1 → S = ∅) ∧
  (a > 1 → S = {x : ℝ | 1 < x ∧ x < 2*a - 1}) ∧
  (a < 1 → S = {x : ℝ | 2*a - 1 < x ∧ x < 1}) :=
by sorry

end inequality_solution_set_l289_28946


namespace pi_estimation_l289_28944

theorem pi_estimation (n : ℕ) (m : ℕ) (h1 : n = 200) (h2 : m = 56) :
  let π_estimate : ℚ := 4 * (m : ℚ) / (n : ℚ) + 2
  π_estimate = 78 / 25 := by
  sorry

end pi_estimation_l289_28944


namespace min_triangle_area_l289_28923

/-- Given a triangle ABC with side AB = 2 and 2/sin(A) + 1/tan(B) = 2√3, 
    its area is greater than or equal to 2√3/3 -/
theorem min_triangle_area (A B C : ℝ) (h1 : 0 < A ∧ A < π) (h2 : 0 < B ∧ B < π) 
  (h3 : 0 < C ∧ C < π) (h4 : A + B + C = π) (h5 : Real.sin A ≠ 0) (h6 : Real.tan B ≠ 0)
  (h7 : 2 / Real.sin A + 1 / Real.tan B = 2 * Real.sqrt 3) :
  1 / 2 * 2 * Real.sin C ≥ 2 * Real.sqrt 3 / 3 := by
  sorry


end min_triangle_area_l289_28923


namespace function_is_linear_l289_28971

/-- A function satisfying the given functional equation -/
def SatisfiesEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x^2 - y^2) = (x - y) * (f x + f y)

/-- The main theorem stating that any function satisfying the equation is linear -/
theorem function_is_linear (f : ℝ → ℝ) (h : SatisfiesEquation f) :
  ∃ k : ℝ, ∀ x : ℝ, f x = k * x := by
  sorry

end function_is_linear_l289_28971


namespace max_value_of_four_numbers_l289_28973

theorem max_value_of_four_numbers
  (a b c d : ℝ)
  (h_positive : 0 < d ∧ 0 < c ∧ 0 < b ∧ 0 < a)
  (h_order : d ≤ c ∧ c ≤ b ∧ b ≤ a)
  (h_sum : a + b + c + d = 4)
  (h_sum_squares : a^2 + b^2 + c^2 + d^2 = 8) :
  a ≤ 1 + Real.sqrt 3 ∧ ∃ (a₀ b₀ c₀ d₀ : ℝ),
    0 < d₀ ∧ d₀ ≤ c₀ ∧ c₀ ≤ b₀ ∧ b₀ ≤ a₀ ∧
    a₀ + b₀ + c₀ + d₀ = 4 ∧
    a₀^2 + b₀^2 + c₀^2 + d₀^2 = 8 ∧
    a₀ = 1 + Real.sqrt 3 :=
by
  sorry

end max_value_of_four_numbers_l289_28973


namespace beef_pack_weight_l289_28995

/-- Given the conditions of James' beef purchase, prove the weight of each pack. -/
theorem beef_pack_weight (num_packs : ℕ) (price_per_pound : ℚ) (total_paid : ℚ) 
  (h1 : num_packs = 5)
  (h2 : price_per_pound = 5.5)
  (h3 : total_paid = 110) :
  (total_paid / price_per_pound) / num_packs = 4 := by
  sorry

#check beef_pack_weight

end beef_pack_weight_l289_28995


namespace gcd_176_88_l289_28985

theorem gcd_176_88 : Nat.gcd 176 88 = 88 := by
  sorry

end gcd_176_88_l289_28985


namespace y_squared_times_three_l289_28945

theorem y_squared_times_three (x y : ℤ) 
  (eq1 : 3 * x + y = 40) 
  (eq2 : 2 * x - y = 20) : 
  3 * y^2 = 48 := by
  sorry

end y_squared_times_three_l289_28945


namespace x_squared_minus_y_squared_l289_28934

theorem x_squared_minus_y_squared (x y : ℝ) 
  (eq1 : x + y = 4) 
  (eq2 : 2 * x - 2 * y = 1) : 
  x^2 - y^2 = 2 := by
  sorry

end x_squared_minus_y_squared_l289_28934


namespace two_activities_count_l289_28990

/-- Represents a school club with three activities -/
structure Club where
  total_members : ℕ
  cannot_paint : ℕ
  cannot_sculpt : ℕ
  cannot_draw : ℕ

/-- Calculates the number of members involved in exactly two activities -/
def members_in_two_activities (c : Club) : ℕ :=
  let can_paint := c.total_members - c.cannot_paint
  let can_sculpt := c.total_members - c.cannot_sculpt
  let can_draw := c.total_members - c.cannot_draw
  can_paint + can_sculpt + can_draw - c.total_members

/-- Theorem stating the number of members involved in exactly two activities -/
theorem two_activities_count (c : Club) 
  (h1 : c.total_members = 150)
  (h2 : c.cannot_paint = 55)
  (h3 : c.cannot_sculpt = 90)
  (h4 : c.cannot_draw = 40) :
  members_in_two_activities c = 115 := by
  sorry

#eval members_in_two_activities ⟨150, 55, 90, 40⟩

end two_activities_count_l289_28990


namespace divisibility_of_n_squared_plus_n_plus_two_l289_28942

theorem divisibility_of_n_squared_plus_n_plus_two (n : ℕ) :
  (∃ k : ℕ, n^2 + n + 2 = 2 * k) ∧ (∃ m : ℕ, ¬(∃ l : ℕ, m^2 + m + 2 = 5 * l)) := by
  sorry

end divisibility_of_n_squared_plus_n_plus_two_l289_28942


namespace custard_pies_sold_is_five_l289_28947

/-- Represents the bakery sales problem --/
structure BakerySales where
  pumpkin_slices_per_pie : ℕ
  custard_slices_per_pie : ℕ
  pumpkin_price_per_slice : ℕ
  custard_price_per_slice : ℕ
  pumpkin_pies_sold : ℕ
  total_revenue : ℕ

/-- Calculates the number of custard pies sold --/
def custard_pies_sold (bs : BakerySales) : ℕ :=
  sorry

/-- Theorem stating that the number of custard pies sold is 5 --/
theorem custard_pies_sold_is_five (bs : BakerySales)
  (h1 : bs.pumpkin_slices_per_pie = 8)
  (h2 : bs.custard_slices_per_pie = 6)
  (h3 : bs.pumpkin_price_per_slice = 5)
  (h4 : bs.custard_price_per_slice = 6)
  (h5 : bs.pumpkin_pies_sold = 4)
  (h6 : bs.total_revenue = 340) :
  custard_pies_sold bs = 5 :=
sorry

end custard_pies_sold_is_five_l289_28947


namespace quadratic_form_sum_l289_28969

theorem quadratic_form_sum (a h k : ℝ) : 
  (∀ x, 5 * x^2 - 20 * x + 8 = a * (x - h)^2 + k) → 
  a + h + k = -5 := by
sorry

end quadratic_form_sum_l289_28969


namespace asparagus_per_plate_l289_28992

theorem asparagus_per_plate
  (bridgette_guests : ℕ)
  (alex_guests : ℕ)
  (extra_plates : ℕ)
  (total_asparagus : ℕ)
  (h1 : bridgette_guests = 84)
  (h2 : alex_guests = 2 * bridgette_guests / 3)
  (h3 : extra_plates = 10)
  (h4 : total_asparagus = 1200) :
  total_asparagus / (bridgette_guests + alex_guests + extra_plates) = 8 :=
by
  sorry

end asparagus_per_plate_l289_28992


namespace quadratic_part_of_equation_l289_28911

-- Define the quadratic equation
def quadratic_equation (x : ℝ) : Prop :=
  x^2 - 8*x + 21 = |x - 5| + 4

-- Define the sum of solutions
def sum_of_solutions : ℝ := 20

-- Theorem to prove
theorem quadratic_part_of_equation :
  ∃ (a b c : ℝ), 
    (∀ x, quadratic_equation x → a*x^2 + b*x + c = |x - 5| + 4) ∧
    (a = 1 ∧ b = -8 ∧ c = 21) :=
sorry

end quadratic_part_of_equation_l289_28911


namespace tangent_and_zeros_theorem_l289_28915

noncomputable section

-- Define the functions
def f (x : ℝ) := 2 * x^3 - 3 * x^2 + 1
def g (k x : ℝ) := k * x + 1 - Real.log x
def h (k x : ℝ) := min (f x) (g k x)

-- Define the theorem
theorem tangent_and_zeros_theorem :
  -- Part 1: Tangent condition
  (∀ a : ℝ, (∃! t₁ t₂ : ℝ, t₁ ≠ t₂ ∧ 
    (f t₁ - (-4) = (6 * t₁^2 - 6 * t₁) * (t₁ - a)) ∧
    (f t₂ - (-4) = (6 * t₂^2 - 6 * t₂) * (t₂ - a)))
   ↔ (a = -1 ∨ a = 7/2)) ∧
  -- Part 2: Zeros condition
  (∀ k : ℝ, (∃! x₁ x₂ x₃ : ℝ, 0 < x₁ ∧ 0 < x₂ ∧ 0 < x₃ ∧ x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₁ ≠ x₃ ∧
    h k x₁ = 0 ∧ h k x₂ = 0 ∧ h k x₃ = 0)
   ↔ (0 < k ∧ k < Real.exp (-2))) := by
  sorry


end tangent_and_zeros_theorem_l289_28915


namespace intersection_of_A_and_B_l289_28974

-- Define the function f
def f (x : ℝ) : ℝ := x^2

-- Define the set B
def B : Set ℝ := {1, 2}

-- Theorem statement
theorem intersection_of_A_and_B 
  (A : Set ℝ) 
  (h1 : ∀ y ∈ B, ∃ x ∈ A, f x = y) :
  (A ∩ B = ∅) ∨ (A ∩ B = {1}) :=
sorry

end intersection_of_A_and_B_l289_28974


namespace even_sum_not_both_odd_l289_28964

theorem even_sum_not_both_odd (n m : ℤ) (h : Even (n^2 + m + n * m)) :
  ¬(Odd n ∧ Odd m) := by
  sorry

end even_sum_not_both_odd_l289_28964


namespace quadratic_roots_property_l289_28994

theorem quadratic_roots_property (c : ℝ) (x₁ x₂ : ℝ) : 
  (x₁^2 + x₁ + c = 0) → 
  (x₂^2 + x₂ + c = 0) → 
  (x₁^2 * x₂ + x₂^2 * x₁ = 3) → 
  c = -3 := by
sorry

end quadratic_roots_property_l289_28994


namespace chord_length_in_isosceles_trapezoid_l289_28961

/-- An isosceles trapezoid with an inscribed circle -/
structure IsoscelesTrapezoidWithInscribedCircle where
  /-- The radius of the inscribed circle -/
  r : ℝ
  /-- The area of the trapezoid -/
  area : ℝ
  /-- The trapezoid is isosceles -/
  isIsosceles : Bool
  /-- The circle is inscribed in the trapezoid -/
  isInscribed : Bool

/-- The theorem stating the length of the chord connecting the tangent points -/
theorem chord_length_in_isosceles_trapezoid 
  (t : IsoscelesTrapezoidWithInscribedCircle) 
  (h1 : t.r = 3)
  (h2 : t.area = 48)
  (h3 : t.isIsosceles = true)
  (h4 : t.isInscribed = true) :
  ∃ (chord_length : ℝ), chord_length = 4.5 := by
  sorry

end chord_length_in_isosceles_trapezoid_l289_28961


namespace flax_acreage_is_80_l289_28928

/-- Represents the acreage of a farm with sunflowers and flax -/
structure FarmAcreage where
  total : ℕ
  flax : ℕ
  sunflowers : ℕ
  total_eq : total = flax + sunflowers
  sunflower_excess : sunflowers = flax + 80

/-- The theorem stating that for a 240-acre farm with the given conditions, 
    the flax acreage is 80 acres -/
theorem flax_acreage_is_80 (farm : FarmAcreage) 
    (h : farm.total = 240) : farm.flax = 80 := by
  sorry

end flax_acreage_is_80_l289_28928


namespace pages_left_to_read_is_1000_l289_28987

/-- Calculates the number of pages left to read in a book series -/
def pagesLeftToRead (totalBooks : ℕ) (pagesPerBook : ℕ) (readFirstMonth : ℕ) : ℕ :=
  let remainingAfterFirstMonth := totalBooks - readFirstMonth
  let readSecondMonth := remainingAfterFirstMonth / 2
  let totalRead := readFirstMonth + readSecondMonth
  let pagesLeft := (totalBooks - totalRead) * pagesPerBook
  pagesLeft

/-- Theorem: Given the specified reading pattern, 1000 pages are left to read -/
theorem pages_left_to_read_is_1000 :
  pagesLeftToRead 14 200 4 = 1000 := by
  sorry

#eval pagesLeftToRead 14 200 4

end pages_left_to_read_is_1000_l289_28987


namespace combined_return_percentage_l289_28922

def investment1 : ℝ := 500
def investment2 : ℝ := 1500
def return1 : ℝ := 0.07
def return2 : ℝ := 0.23

def total_investment : ℝ := investment1 + investment2
def total_return : ℝ := investment1 * return1 + investment2 * return2

theorem combined_return_percentage :
  (total_return / total_investment) * 100 = 19 := by sorry

end combined_return_percentage_l289_28922


namespace dana_weekend_earnings_l289_28924

def dana_earnings (hourly_rate : ℝ) (commission_rate : ℝ) 
  (friday_hours : ℝ) (friday_sales : ℝ)
  (saturday_hours : ℝ) (saturday_sales : ℝ)
  (sunday_hours : ℝ) (sunday_sales : ℝ) : ℝ :=
  let total_hours := friday_hours + saturday_hours + sunday_hours
  let total_sales := friday_sales + saturday_sales + sunday_sales
  let hourly_earnings := hourly_rate * total_hours
  let commission_earnings := commission_rate * total_sales
  hourly_earnings + commission_earnings

theorem dana_weekend_earnings :
  dana_earnings 13 0.05 9 800 10 1000 3 300 = 391 := by
  sorry

end dana_weekend_earnings_l289_28924


namespace solution_set_equality_l289_28993

-- Define the set S
def S : Set ℝ := {x : ℝ | |x + 3| - |x - 2| ≥ 3}

-- State the theorem
theorem solution_set_equality : S = Set.Ici 1 := by sorry

end solution_set_equality_l289_28993


namespace solve_equation_one_solve_equation_two_l289_28983

-- Equation 1
theorem solve_equation_one (x : ℚ) : 2 * (x - 3) = 1 - 3 * (x + 1) ↔ x = 4 / 5 := by sorry

-- Equation 2
theorem solve_equation_two (x : ℚ) : 3 * x + (x - 1) / 2 = 3 - (x - 1) / 3 ↔ x = 1 := by sorry

end solve_equation_one_solve_equation_two_l289_28983


namespace car_production_total_l289_28963

theorem car_production_total (north_america europe asia south_america : ℕ) 
  (h1 : north_america = 3884)
  (h2 : europe = 2871)
  (h3 : asia = 5273)
  (h4 : south_america = 1945) :
  north_america + europe + asia + south_america = 13973 :=
by sorry

end car_production_total_l289_28963


namespace min_socks_for_15_pairs_l289_28957

/-- Represents the number of socks of each color in the box -/
structure SockBox where
  white : Nat
  red : Nat
  blue : Nat
  green : Nat
  yellow : Nat

/-- Calculates the minimum number of socks needed to guarantee at least n pairs -/
def minSocksForPairs (box : SockBox) (n : Nat) : Nat :=
  (box.white.min n + box.red.min n + box.blue.min n + box.green.min n + box.yellow.min n) * 2 - 1

/-- The theorem stating the minimum number of socks needed for 15 pairs -/
theorem min_socks_for_15_pairs (box : SockBox) 
    (h_white : box.white = 150)
    (h_red : box.red = 120)
    (h_blue : box.blue = 90)
    (h_green : box.green = 60)
    (h_yellow : box.yellow = 30) :
    minSocksForPairs box 15 = 146 := by
  sorry

end min_socks_for_15_pairs_l289_28957


namespace matrix_equality_proof_l289_28979

open Matrix

-- Define the condition for matrix congruence modulo 3
def congruent_mod_3 (X Y : Matrix (Fin 6) (Fin 6) ℤ) : Prop :=
  ∀ i j, (X i j - Y i j) % 3 = 0

-- Main theorem statement
theorem matrix_equality_proof (A B : Matrix (Fin 6) (Fin 6) ℤ)
  (h1 : congruent_mod_3 A (1 : Matrix (Fin 6) (Fin 6) ℤ))
  (h2 : congruent_mod_3 B (1 : Matrix (Fin 6) (Fin 6) ℤ))
  (h3 : A ^ 3 * B ^ 3 * A ^ 3 = B ^ 3) :
  A = 1 := by
  sorry

end matrix_equality_proof_l289_28979


namespace shift_increasing_interval_l289_28909

-- Define a function f
variable (f : ℝ → ℝ)

-- Define what it means for f to be increasing on an interval
def IncreasingOn (f : ℝ → ℝ) (a b : ℝ) :=
  ∀ x y, a < x ∧ x < y ∧ y < b → f x < f y

-- State the theorem
theorem shift_increasing_interval
  (h : IncreasingOn f (-2) 3) :
  IncreasingOn (fun x ↦ f (x + 5)) (-7) (-2) :=
sorry

end shift_increasing_interval_l289_28909


namespace cookie_price_is_two_l289_28966

/-- The price of each cookie in dollars, given the baking and sales conditions -/
def cookie_price (clementine_cookies jake_cookies tory_cookies total_revenue : ℕ) : ℚ :=
  total_revenue / (clementine_cookies + jake_cookies + tory_cookies)

theorem cookie_price_is_two :
  let clementine_cookies : ℕ := 72
  let jake_cookies : ℕ := 2 * clementine_cookies
  let tory_cookies : ℕ := (clementine_cookies + jake_cookies) / 2
  let total_revenue : ℕ := 648
  cookie_price clementine_cookies jake_cookies tory_cookies total_revenue = 2 := by
sorry

#eval cookie_price 72 144 108 648

end cookie_price_is_two_l289_28966


namespace set_equality_implies_m_zero_l289_28913

theorem set_equality_implies_m_zero (m : ℝ) : 
  ({3, m} : Set ℝ) = ({3*m, 3} : Set ℝ) → m = 0 := by
  sorry

end set_equality_implies_m_zero_l289_28913


namespace parallelogram_vertex_d_l289_28965

/-- Represents a 2D point -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a parallelogram -/
structure Parallelogram where
  A : Point
  B : Point
  C : Point
  D : Point

/-- Theorem: Given a parallelogram ABCD with vertices A(-1,-2), B(3,-1), and C(5,6), 
    the coordinates of vertex D are (1,5) -/
theorem parallelogram_vertex_d (ABCD : Parallelogram) 
    (h1 : ABCD.A = ⟨-1, -2⟩) 
    (h2 : ABCD.B = ⟨3, -1⟩) 
    (h3 : ABCD.C = ⟨5, 6⟩) : 
    ABCD.D = ⟨1, 5⟩ := by
  sorry


end parallelogram_vertex_d_l289_28965


namespace problem_statement_l289_28932

-- Define the function f and its derivative g
variable (f : ℝ → ℝ)
variable (g : ℝ → ℝ)

-- Define the conditions
axiom f_diff : ∀ x, HasDerivAt f (g x) x
axiom f_even : ∀ x, f (3/2 - 2*x) = f (3/2 + 2*x)
axiom g_even : ∀ x, g (2 + x) = g (2 - x)

-- State the theorem to be proved
theorem problem_statement :
  (f (-1) = f 4) ∧ (g (-1/2) = 0) := by
  sorry

end problem_statement_l289_28932


namespace total_sales_amount_theorem_l289_28933

def weight_deviations : List ℤ := [-4, -1, -2, 2, 3, 4, 7, 1]
def qualification_criterion : ℤ := 4
def price_per_bag : ℚ := 86/10

def is_qualified (deviation : ℤ) : Bool :=
  deviation.natAbs ≤ qualification_criterion

theorem total_sales_amount_theorem :
  (weight_deviations.filter is_qualified).length * price_per_bag = 602/10 := by
  sorry

end total_sales_amount_theorem_l289_28933


namespace john_purchase_profit_l289_28943

/-- Represents the purchase and sale of items with profit or loss -/
theorem john_purchase_profit (x : ℝ) : 
  let grinder_purchase := 15000
  let grinder_loss_percent := 0.04
  let mobile_profit_percent := 0.15
  let total_profit := 600
  let grinder_sale := grinder_purchase * (1 - grinder_loss_percent)
  let mobile_sale := x * (1 + mobile_profit_percent)
  (mobile_sale - x) - (grinder_purchase - grinder_sale) = total_profit →
  x = 8000 := by
sorry

end john_purchase_profit_l289_28943


namespace max_y_over_x_l289_28959

-- Define the feasible region
def FeasibleRegion (x y : ℝ) : Prop :=
  x + y ≥ 3 ∧ x - y ≥ -1 ∧ 2*x - y ≤ 3

-- State the theorem
theorem max_y_over_x :
  ∃ (max : ℝ), max = 2 ∧
  ∀ (x y : ℝ), FeasibleRegion x y → y / x ≤ max :=
sorry

end max_y_over_x_l289_28959


namespace definite_integral_x_squared_plus_sin_l289_28936

open Real MeasureTheory

theorem definite_integral_x_squared_plus_sin : 
  ∫ x in (-1)..1, (x^2 + Real.sin x) = 2/3 := by sorry

end definite_integral_x_squared_plus_sin_l289_28936


namespace theater_ticket_sales_l289_28962

theorem theater_ticket_sales (total_tickets : ℕ) (advanced_price door_price : ℚ) (total_revenue : ℚ) 
  (h1 : total_tickets = 800)
  (h2 : advanced_price = 14.5)
  (h3 : door_price = 22)
  (h4 : total_revenue = 16640) :
  ∃ (door_tickets : ℕ), 
    door_tickets = 672 ∧ 
    (total_tickets - door_tickets) * advanced_price + door_tickets * door_price = total_revenue :=
by sorry

end theater_ticket_sales_l289_28962


namespace line_with_definite_slope_line_equation_through_two_points_l289_28916

-- Statement B
theorem line_with_definite_slope (m : ℝ) :
  ∃ (k : ℝ), ∀ (x y : ℝ), m * x + y - 2 = 0 → y = k * x + (2 : ℝ) :=
sorry

-- Statement D
theorem line_equation_through_two_points (x₁ y₁ x₂ y₂ : ℝ) (h : x₁ ≠ x₂) :
  ∀ (x y : ℝ), y - y₁ = ((y₂ - y₁) / (x₂ - x₁)) * (x - x₁) →
  ∃ (m b : ℝ), y = m * x + b :=
sorry

end line_with_definite_slope_line_equation_through_two_points_l289_28916


namespace cars_to_sell_l289_28951

/-- The number of cars each client selected -/
def cars_per_client : ℕ := 3

/-- The number of times each car was selected -/
def selections_per_car : ℕ := 3

/-- The number of clients who visited the garage -/
def num_clients : ℕ := 15

/-- The number of cars the seller has to sell -/
def num_cars : ℕ := 15

theorem cars_to_sell :
  num_cars * selections_per_car = num_clients * cars_per_client :=
by sorry

end cars_to_sell_l289_28951


namespace intersection_of_A_and_B_l289_28988

-- Define set A
def A : Set ℝ := {x | x^2 < 4}

-- Define set B
def B : Set ℝ := {x | -3 < x ∧ x ≤ 1}

-- Theorem statement
theorem intersection_of_A_and_B :
  A ∩ B = {x : ℝ | -2 < x ∧ x ≤ 1} := by sorry

end intersection_of_A_and_B_l289_28988


namespace mixture_composition_l289_28919

theorem mixture_composition (x y : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) (hsum : x + y = 100) :
  0.4 * x + 0.5 * y = 47 → x = 30 :=
by sorry

end mixture_composition_l289_28919


namespace a_can_be_any_real_l289_28938

theorem a_can_be_any_real (a b c d : ℝ) 
  (h1 : (a / b) ^ 2 < (c / d) ^ 2)
  (h2 : b ≠ 0)
  (h3 : d ≠ 0)
  (h4 : c = -d) :
  ∃ (x : ℝ), x = a ∧ (x < 0 ∨ x = 0 ∨ x > 0) :=
by sorry

end a_can_be_any_real_l289_28938


namespace four_integers_with_average_five_l289_28905

theorem four_integers_with_average_five (a b c d : ℕ+) : 
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d →
  (a + b + c + d : ℚ) / 4 = 5 →
  (max a (max b (max c d)) - min a (min b (min c d)) : ℕ) = 
    max a (max b (max c d)) - min a (min b (min c d)) →
  ((a + b + c + d) - max a (max b (max c d)) - min a (min b (min c d)) : ℚ) / 2 = 5/2 :=
by sorry

end four_integers_with_average_five_l289_28905


namespace rowing_coach_votes_l289_28939

theorem rowing_coach_votes (num_coaches : ℕ) (votes_per_coach : ℕ) (coaches_per_voter : ℕ) : 
  num_coaches = 36 → 
  votes_per_coach = 5 → 
  coaches_per_voter = 3 → 
  (num_coaches * votes_per_coach) / coaches_per_voter = 60 := by
  sorry

end rowing_coach_votes_l289_28939


namespace chromatid_non_separation_can_result_in_XXY_l289_28991

/- Define the basic types and structures -/
inductive Chromosome
| X
| Y

structure Sperm :=
(chromosomes : List Chromosome)

structure Egg :=
(chromosomes : List Chromosome)

structure Offspring :=
(chromosomes : List Chromosome)

/- Define the process of sperm formation with non-separation of chromatids -/
def spermFormationWithNonSeparation : List Sperm :=
[{chromosomes := [Chromosome.X, Chromosome.X]}, {chromosomes := [Chromosome.Y, Chromosome.Y]}]

/- Define a normal egg -/
def normalEgg : Egg :=
{chromosomes := [Chromosome.X]}

/- Define the fertilization process -/
def fertilize (sperm : Sperm) (egg : Egg) : Offspring :=
{chromosomes := sperm.chromosomes ++ egg.chromosomes}

/- The theorem to be proved -/
theorem chromatid_non_separation_can_result_in_XXY :
  ∃ (sperm : Sperm) (egg : Egg),
    sperm ∈ spermFormationWithNonSeparation ∧
    egg = normalEgg ∧
    (fertilize sperm egg).chromosomes = [Chromosome.X, Chromosome.X, Chromosome.Y] :=
sorry

end chromatid_non_separation_can_result_in_XXY_l289_28991


namespace correct_stratified_sample_teaching_l289_28902

/-- Represents the composition of staff in a school -/
structure SchoolStaff where
  total : ℕ
  administrative : ℕ
  teaching : ℕ
  support : ℕ

/-- Calculates the number of teaching staff to be included in a stratified sample -/
def stratifiedSampleTeaching (staff : SchoolStaff) (sampleSize : ℕ) : ℕ :=
  (staff.teaching * sampleSize) / staff.total

/-- Theorem stating the correct number of teaching staff in the stratified sample -/
theorem correct_stratified_sample_teaching (staff : SchoolStaff) (sampleSize : ℕ) :
  staff.total = 200 ∧ 
  staff.administrative = 24 ∧ 
  staff.teaching = 10 * staff.support ∧
  staff.teaching + staff.support + staff.administrative = staff.total ∧
  sampleSize = 50 →
  stratifiedSampleTeaching staff sampleSize = 40 := by
  sorry

end correct_stratified_sample_teaching_l289_28902


namespace circle_coverage_fraction_l289_28981

/-- The fraction of a smaller circle's area not covered by a larger circle when placed inside it -/
theorem circle_coverage_fraction (dX dY : ℝ) (h_dX : dX = 16) (h_dY : dY = 18) (h_inside : dX < dY) :
  (π * (dY / 2)^2 - π * (dX / 2)^2) / (π * (dX / 2)^2) = 17 / 64 := by
  sorry

end circle_coverage_fraction_l289_28981


namespace magician_trick_min_digits_l289_28935

/-- The minimum number of digits required for the magician's trick -/
def min_digits : ℕ := 101

/-- The number of possible two-digit combinations -/
def two_digit_combinations (n : ℕ) : ℕ := (n - 1) * (10^(n - 2))

/-- The total number of possible arrangements -/
def total_arrangements (n : ℕ) : ℕ := 10^n

/-- Theorem stating that 101 is the minimum number of digits required for the magician's trick -/
theorem magician_trick_min_digits :
  (∀ n : ℕ, n ≥ min_digits → two_digit_combinations n ≥ total_arrangements n) ∧
  (∀ n : ℕ, n < min_digits → two_digit_combinations n < total_arrangements n) :=
sorry

end magician_trick_min_digits_l289_28935


namespace function_value_proof_l289_28941

theorem function_value_proof : 
  ∀ f : ℝ → ℝ, 
  (∀ x, f x = (x - 3) * (x + 4)) → 
  f 29 = 170 → 
  ∃ x, f x = 170 ∧ x = 13 := by
sorry

end function_value_proof_l289_28941


namespace tan_alpha_values_l289_28917

theorem tan_alpha_values (α : ℝ) :
  5 * Real.sin (2 * α) + 5 * Real.cos (2 * α) + 1 = 0 →
  Real.tan α = 3 ∨ Real.tan α = -1/2 := by
  sorry

end tan_alpha_values_l289_28917


namespace multiples_of_4_and_5_between_100_and_350_l289_28921

theorem multiples_of_4_and_5_between_100_and_350 : 
  (Finset.filter (fun n => n % 4 = 0 ∧ n % 5 = 0 ∧ 100 < n ∧ n < 350) (Finset.range 350)).card = 12 :=
by sorry

end multiples_of_4_and_5_between_100_and_350_l289_28921


namespace translation_coordinates_l289_28948

/-- Given a point A(-1, 2) in the Cartesian coordinate system,
    translated 4 units to the right and 2 units down to obtain point A₁,
    the coordinates of A₁ are (3, 0). -/
theorem translation_coordinates :
  let A : ℝ × ℝ := (-1, 2)
  let right_translation : ℝ := 4
  let down_translation : ℝ := 2
  let A₁ : ℝ × ℝ := (A.1 + right_translation, A.2 - down_translation)
  A₁ = (3, 0) := by
sorry

end translation_coordinates_l289_28948


namespace min_value_perpendicular_vectors_l289_28912

theorem min_value_perpendicular_vectors (x y : ℝ) :
  (x - 1) * 4 + 2 * y = 0 →
  ∃ (min : ℝ), min = 6 ∧ ∀ (z : ℝ), z = 9^x + 3^y → z ≥ min :=
by sorry

end min_value_perpendicular_vectors_l289_28912


namespace moses_esther_difference_l289_28954

theorem moses_esther_difference (total : ℝ) (moses_percentage : ℝ) : 
  total = 50 ∧ moses_percentage = 0.4 → 
  let moses_share := moses_percentage * total
  let remainder := total - moses_share
  let esther_share := remainder / 2
  moses_share - esther_share = 5 := by
sorry

end moses_esther_difference_l289_28954


namespace point_on_graph_l289_28900

/-- A point (x, y) lies on the graph of y = 2x - 1 -/
def lies_on_graph (x y : ℝ) : Prop := y = 2 * x - 1

/-- The point (2, 3) lies on the graph of y = 2x - 1 -/
theorem point_on_graph : lies_on_graph 2 3 := by
  sorry

end point_on_graph_l289_28900


namespace jensen_family_mileage_l289_28920

/-- Represents the mileage problem for the Jensen family's road trip -/
theorem jensen_family_mileage
  (total_highway_miles : ℝ)
  (total_city_miles : ℝ)
  (highway_mpg : ℝ)
  (total_gallons : ℝ)
  (h1 : total_highway_miles = 210)
  (h2 : total_city_miles = 54)
  (h3 : highway_mpg = 35)
  (h4 : total_gallons = 9) :
  (total_city_miles / (total_gallons - total_highway_miles / highway_mpg)) = 18 :=
by sorry

end jensen_family_mileage_l289_28920


namespace percentage_of_students_with_birds_l289_28910

/-- Given a school with 500 students where 75 students own birds,
    prove that 15% of the students own birds. -/
theorem percentage_of_students_with_birds :
  ∀ (total_students : ℕ) (students_with_birds : ℕ),
    total_students = 500 →
    students_with_birds = 75 →
    (students_with_birds : ℚ) / total_students * 100 = 15 := by
  sorry

end percentage_of_students_with_birds_l289_28910


namespace division_theorem_l289_28904

theorem division_theorem (dividend : ℕ) (divisor : ℕ) (remainder : ℕ) (quotient : ℕ) : 
  dividend = divisor * quotient + remainder →
  dividend = 162 →
  divisor = 17 →
  remainder = 9 →
  quotient = 9 := by
sorry

end division_theorem_l289_28904


namespace davids_biology_mark_l289_28906

def marks_english : ℕ := 45
def marks_mathematics : ℕ := 35
def marks_physics : ℕ := 52
def marks_chemistry : ℕ := 47
def average_marks : ℚ := 46.8

theorem davids_biology_mark (marks_biology : ℕ) :
  (marks_english + marks_mathematics + marks_physics + marks_chemistry + marks_biology : ℚ) / 5 = average_marks →
  marks_biology = 55 := by
sorry

end davids_biology_mark_l289_28906


namespace jimmy_win_probability_remainder_mod_1000_l289_28925

/-- Probability of rolling an odd number on a single die -/
def prob_odd_single : ℚ := 3/4

/-- Probability of Jimmy winning a single game -/
def prob_jimmy_win : ℚ := 1 - prob_odd_single^2

/-- Probability of Jimmy winning exactly k out of n games -/
def prob_jimmy_win_k_of_n (k n : ℕ) : ℚ :=
  Nat.choose n k * prob_jimmy_win^k * (1 - prob_jimmy_win)^(n - k)

/-- Probability of Jimmy winning 3 games before Jacob wins 3 games -/
def prob_jimmy_wins_3_first : ℚ :=
  prob_jimmy_win_k_of_n 3 3 +
  prob_jimmy_win_k_of_n 3 4 +
  prob_jimmy_win_k_of_n 3 5

theorem jimmy_win_probability :
  prob_jimmy_wins_3_first = 201341 / 2^19 :=
sorry

theorem remainder_mod_1000 :
  (201341 : ℤ) + 19 ≡ 360 [ZMOD 1000] :=
sorry

end jimmy_win_probability_remainder_mod_1000_l289_28925


namespace unique_solution_cubic_equation_l289_28930

theorem unique_solution_cubic_equation :
  ∃! x : ℝ, x^3 + (x + 2)^3 + (x + 4)^3 = (x + 6)^3 ∧ x = 6 := by
  sorry

end unique_solution_cubic_equation_l289_28930


namespace tan_difference_pi_4_minus_theta_l289_28952

theorem tan_difference_pi_4_minus_theta (θ : Real) (h : Real.tan θ = 1/2) :
  Real.tan (π/4 - θ) = 1/3 := by
  sorry

end tan_difference_pi_4_minus_theta_l289_28952


namespace car_trip_duration_l289_28999

/-- Represents the duration of a car trip with varying speeds -/
def car_trip (first_speed second_speed average_speed : ℝ) (first_duration : ℝ) : ℝ → Prop :=
  λ total_duration : ℝ =>
    let second_duration := total_duration - first_duration
    let total_distance := first_speed * first_duration + second_speed * second_duration
    (total_distance / total_duration = average_speed) ∧
    (total_duration > first_duration) ∧
    (first_duration > 0) ∧
    (second_duration > 0)

/-- Theorem stating that the car trip with given parameters lasts 7.5 hours -/
theorem car_trip_duration :
  car_trip 30 42 34 5 7.5 := by
  sorry

end car_trip_duration_l289_28999


namespace sector_central_angle_l289_28984

/-- Given a circular sector with area 6 cm² and radius 2 cm, prove its central angle is 3 radians. -/
theorem sector_central_angle (area : ℝ) (radius : ℝ) (h1 : area = 6) (h2 : radius = 2) :
  (2 * area) / (radius ^ 2) = 3 := by
  sorry

end sector_central_angle_l289_28984


namespace sum_of_a_and_b_l289_28931

theorem sum_of_a_and_b (a b : ℝ) (h : Real.sqrt (a + 2) + (b - 3)^2 = 0) : a + b = 1 := by
  sorry

end sum_of_a_and_b_l289_28931


namespace gcd_128_144_256_l289_28986

theorem gcd_128_144_256 : Nat.gcd 128 (Nat.gcd 144 256) = 128 := by sorry

end gcd_128_144_256_l289_28986


namespace complex_number_magnitude_l289_28918

theorem complex_number_magnitude (z w : ℂ) 
  (h1 : Complex.abs (3 * z - w) = 15)
  (h2 : Complex.abs (z + 3 * w) = 3)
  (h3 : Complex.abs (z - w) = 6) :
  Complex.abs w ^ 2 = 3.375 := by sorry

end complex_number_magnitude_l289_28918


namespace trapezoid_halving_line_iff_condition_l289_28907

/-- A trapezoid with bases a and b, and legs c and d. -/
structure Trapezoid where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ
  a_pos : 0 < a
  b_pos : 0 < b
  c_pos : 0 < c
  d_pos : 0 < d
  parallel_bases : a ≠ b → a < b

/-- The condition for a line to halve both perimeter and area of a trapezoid. -/
def halvingLineCondition (t : Trapezoid) : Prop :=
  (t.c + t.d) / 2 = (t.a + t.b) / 2 + Real.sqrt ((t.a^2 + t.b^2) / 2) ∨ t.a = t.b

/-- Theorem: A line parallel to the bases halves both perimeter and area of a trapezoid
    if and only if the halving line condition is satisfied. -/
theorem trapezoid_halving_line_iff_condition (t : Trapezoid) :
  ∃ (x : ℝ), 0 < x ∧ x < t.c ∧ x < t.d ∧
    (x + x + t.a + t.b = (t.a + t.b + t.c + t.d) / 2) ∧
    (x * (t.a + t.b) = (t.a + t.b) * t.c / 2) ↔
  halvingLineCondition t :=
sorry

end trapezoid_halving_line_iff_condition_l289_28907


namespace minimum_spotted_blueeyed_rabbits_l289_28908

theorem minimum_spotted_blueeyed_rabbits 
  (total : ℕ) (spotted : ℕ) (blueeyed : ℕ) 
  (h_total : total = 100)
  (h_spotted : spotted = 53)
  (h_blueeyed : blueeyed = 73) :
  ∃ (both : ℕ), both ≥ 26 ∧ 
    ∀ (x : ℕ), x < 26 → spotted + blueeyed - x > total :=
by sorry

end minimum_spotted_blueeyed_rabbits_l289_28908


namespace arithmetic_sequence_sum_l289_28982

theorem arithmetic_sequence_sum (x y z : ℤ) : 
  (x + y + z = 72) →
  (∃ (x y z : ℤ), x + y + z = 72 ∧ y - x = 1) ∧
  (∃ (x y z : ℤ), x + y + z = 72 ∧ y - x = 2) ∧
  (¬ ∃ (x y z : ℤ), x + y + z = 72 ∧ y - x = 2 ∧ Odd x) :=
by sorry

end arithmetic_sequence_sum_l289_28982


namespace binomial_p_value_l289_28978

/-- A random variable following a binomial distribution -/
structure BinomialRV where
  n : ℕ
  p : ℝ
  h1 : 0 ≤ p ∧ p ≤ 1

/-- Expected value of a binomial random variable -/
def expected_value (X : BinomialRV) : ℝ := X.n * X.p

/-- Variance of a binomial random variable -/
def variance (X : BinomialRV) : ℝ := X.n * X.p * (1 - X.p)

theorem binomial_p_value (X : BinomialRV) 
  (h2 : expected_value X = 30)
  (h3 : variance X = 20) : 
  X.p = 1/3 := by
  sorry

end binomial_p_value_l289_28978


namespace marching_band_members_l289_28970

theorem marching_band_members :
  ∃! n : ℕ, 100 < n ∧ n < 200 ∧
  n % 4 = 1 ∧ n % 5 = 2 ∧ n % 7 = 3 ∧
  n = 157 := by sorry

end marching_band_members_l289_28970


namespace solution_set_properties_inequality_properties_l289_28989

/-- The function f(x) = x² - ax + 3 -/
def f (a : ℝ) (x : ℝ) : ℝ := x^2 - a*x + 3

/-- Part I: Solution set properties -/
theorem solution_set_properties (a b : ℝ) :
  (∀ x, f a x ≤ -3 ↔ b ≤ x ∧ x ≤ 3) →
  a = 5 ∧ b = 2 :=
sorry

/-- Part II: Inequality properties -/
theorem inequality_properties (a : ℝ) :
  (∀ x, x ≥ 1/2 → f a x ≥ 1 - x^2) →
  a ≤ 4 :=
sorry

end solution_set_properties_inequality_properties_l289_28989


namespace theater_ticket_difference_l289_28949

theorem theater_ticket_difference :
  ∀ (orchestra_tickets balcony_tickets : ℕ),
    orchestra_tickets + balcony_tickets = 370 →
    12 * orchestra_tickets + 8 * balcony_tickets = 3320 →
    balcony_tickets - orchestra_tickets = 190 :=
by
  sorry

end theater_ticket_difference_l289_28949


namespace quadratic_roots_condition_l289_28960

/-- A quadratic function f(x) = x^2 + (k+2)x + k + 5 -/
def f (k : ℝ) (x : ℝ) : ℝ := x^2 + (k+2)*x + k + 5

theorem quadratic_roots_condition (k : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁ > 0 ∧ x₂ > 0 ∧ f k x₁ = 0 ∧ f k x₂ = 0) →
  -5 < k ∧ k < -4 :=
by sorry

end quadratic_roots_condition_l289_28960


namespace sin_2alpha_value_l289_28980

theorem sin_2alpha_value (α : Real) 
  (h : (1 - Real.tan α) / (1 + Real.tan α) = 3 - 2 * Real.sqrt 2) : 
  Real.sin (2 * α) = (2 * Real.sqrt 2) / 3 := by
  sorry

end sin_2alpha_value_l289_28980


namespace glucose_solution_volume_l289_28967

/-- Given a glucose solution where 500 cubic centimeters contain 10 grams of glucose,
    this theorem proves that the volume of solution containing 20 grams of glucose
    is 1000 cubic centimeters. -/
theorem glucose_solution_volume :
  let volume_500cc : ℝ := 500
  let glucose_500cc : ℝ := 10
  let glucose_target : ℝ := 20
  let volume_target : ℝ := (glucose_target * volume_500cc) / glucose_500cc
  volume_target = 1000 := by
  sorry

end glucose_solution_volume_l289_28967


namespace smallest_n_with_eight_and_terminating_l289_28903

/-- A function that checks if a positive integer contains the digit 8 -/
def containsEight (n : ℕ+) : Prop := sorry

/-- A function that checks if the reciprocal of a positive integer is a terminating decimal -/
def isTerminatingDecimal (n : ℕ+) : Prop := ∃ (a b : ℕ), n = 2^a * 5^b

theorem smallest_n_with_eight_and_terminating : 
  (∀ m : ℕ+, m < 8 → ¬(containsEight m ∧ isTerminatingDecimal m)) ∧ 
  (containsEight 8 ∧ isTerminatingDecimal 8) := by sorry

end smallest_n_with_eight_and_terminating_l289_28903


namespace small_tile_position_l289_28956

/-- Represents a tile on the grid -/
inductive Tile
| Small : Tile  -- 1x1 tile
| Large : Tile  -- 1x3 tile

/-- Represents a position on the 7x7 grid -/
structure Position :=
  (row : Fin 7)
  (col : Fin 7)

/-- Checks if a position is on the border of the grid -/
def is_border (p : Position) : Prop :=
  p.row = 0 ∨ p.row = 6 ∨ p.col = 0 ∨ p.col = 6

/-- Checks if a position is in the center of the grid -/
def is_center (p : Position) : Prop :=
  p.row = 3 ∧ p.col = 3

/-- Represents the state of the grid -/
structure GridState :=
  (tiles : List (Tile × Position))
  (small_tile_count : Nat)
  (large_tile_count : Nat)

/-- Checks if a GridState is valid according to the problem conditions -/
def is_valid_state (state : GridState) : Prop :=
  state.small_tile_count = 1 ∧
  state.large_tile_count = 16 ∧
  state.tiles.length = 17

/-- The main theorem to prove -/
theorem small_tile_position (state : GridState) :
  is_valid_state state →
  ∃ (p : Position), (Tile.Small, p) ∈ state.tiles ∧ (is_border p ∨ is_center p) :=
sorry

end small_tile_position_l289_28956


namespace upper_limit_of_b_l289_28975

theorem upper_limit_of_b (a b : ℤ) (h1 : 9 ≤ a ∧ a ≤ 14) (h2 : b ≥ 7) 
  (h3 : (14 : ℚ) / 7 - (9 : ℚ) / b = 1.55) : b ≤ 19 := by
  sorry

end upper_limit_of_b_l289_28975


namespace age_difference_l289_28940

theorem age_difference (A B C : ℕ) : A + B = B + C + 14 → A = C + 14 := by
  sorry

end age_difference_l289_28940


namespace carrot_count_l289_28901

/-- The number of carrots initially on the scale -/
def initial_carrots : ℕ := 20

/-- The total weight of carrots in grams -/
def total_weight : ℕ := 3640

/-- The average weight of remaining carrots in grams -/
def avg_weight_remaining : ℕ := 180

/-- The average weight of removed carrots in grams -/
def avg_weight_removed : ℕ := 190

/-- The number of removed carrots -/
def removed_carrots : ℕ := 4

theorem carrot_count : 
  total_weight = (initial_carrots - removed_carrots) * avg_weight_remaining + 
                 removed_carrots * avg_weight_removed := by
  sorry

end carrot_count_l289_28901


namespace street_lamp_combinations_l289_28914

/-- The number of lamps in the row -/
def total_lamps : ℕ := 12

/-- The number of lamps that can be turned off -/
def lamps_to_turn_off : ℕ := 3

/-- The number of valid positions to insert turned-off lamps -/
def valid_positions : ℕ := total_lamps - lamps_to_turn_off - 1

theorem street_lamp_combinations : 
  (valid_positions.choose lamps_to_turn_off) = 56 := by
  sorry

#eval valid_positions.choose lamps_to_turn_off

end street_lamp_combinations_l289_28914


namespace two_a_minus_a_equals_a_l289_28977

theorem two_a_minus_a_equals_a (a : ℝ) : 2 * a - a = a := by
  sorry

end two_a_minus_a_equals_a_l289_28977
