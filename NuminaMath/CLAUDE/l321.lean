import Mathlib

namespace NUMINAMATH_CALUDE_dress_price_difference_l321_32151

/-- Given a dress with an original price that was discounted by 15% to $85, 
    and then increased by 25%, prove that the difference between the original 
    price and the final price is $6.25. -/
theorem dress_price_difference (original_price : ℝ) : 
  original_price * (1 - 0.15) = 85 →
  original_price - (85 * (1 + 0.25)) = -6.25 := by
sorry

end NUMINAMATH_CALUDE_dress_price_difference_l321_32151


namespace NUMINAMATH_CALUDE_sum_of_powers_of_two_l321_32128

theorem sum_of_powers_of_two : 2^4 + 2^4 + 2^4 + 2^4 = 2^6 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_powers_of_two_l321_32128


namespace NUMINAMATH_CALUDE_gcd_of_powers_of_101_l321_32125

theorem gcd_of_powers_of_101 : 
  Nat.Prime 101 → Nat.gcd (101^11 + 1) (101^11 + 101^3 + 1) = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_powers_of_101_l321_32125


namespace NUMINAMATH_CALUDE_lori_earnings_l321_32170

/-- Represents the earnings from Lori's carsharing company -/
def carsharing_earnings (num_red_cars num_white_cars : ℕ) 
  (red_car_rate white_car_rate : ℚ) (rental_hours : ℕ) : ℚ :=
  let total_minutes := rental_hours * 60
  let red_car_earnings := num_red_cars * red_car_rate * total_minutes
  let white_car_earnings := num_white_cars * white_car_rate * total_minutes
  red_car_earnings + white_car_earnings

/-- Theorem stating that Lori's earnings are $2340 given the problem conditions -/
theorem lori_earnings : 
  carsharing_earnings 3 2 3 2 3 = 2340 := by
  sorry

#eval carsharing_earnings 3 2 3 2 3

end NUMINAMATH_CALUDE_lori_earnings_l321_32170


namespace NUMINAMATH_CALUDE_exercise_book_distribution_l321_32113

theorem exercise_book_distribution (students : ℕ) (total_books : ℕ) : 
  (3 * students + 7 = total_books) ∧ (5 * students = total_books + 9) →
  students = 8 ∧ total_books = 31 := by
sorry

end NUMINAMATH_CALUDE_exercise_book_distribution_l321_32113


namespace NUMINAMATH_CALUDE_sqrt_of_four_equals_two_l321_32155

theorem sqrt_of_four_equals_two : Real.sqrt 4 = 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_of_four_equals_two_l321_32155


namespace NUMINAMATH_CALUDE_factor_expression_l321_32101

theorem factor_expression (x : ℝ) : 72 * x^5 - 90 * x^9 = 18 * x^5 * (4 - 5 * x^4) := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_l321_32101


namespace NUMINAMATH_CALUDE_triangle_side_less_than_semiperimeter_l321_32118

theorem triangle_side_less_than_semiperimeter (a b c : ℝ) (h : 0 < a ∧ 0 < b ∧ 0 < c) :
  a < (a + b + c) / 2 ∧ b < (a + b + c) / 2 ∧ c < (a + b + c) / 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_less_than_semiperimeter_l321_32118


namespace NUMINAMATH_CALUDE_intersection_A_B_union_A_complement_B_l321_32190

-- Define the sets A and B
def A : Set ℝ := {x | x^2 + 3*x - 10 < 0}
def B : Set ℝ := {x | x^2 - 2*x - 3 ≥ 0}

-- Define the universal set R (real numbers)
def R : Type := ℝ

-- Theorem for A ∩ B
theorem intersection_A_B : A ∩ B = {x : ℝ | -5 < x ∧ x ≤ -1} := by sorry

-- Theorem for A ∪ (∁ₖ B)
theorem union_A_complement_B : A ∪ (Set.univ \ B) = {x : ℝ | -5 < x ∧ x < 3} := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_union_A_complement_B_l321_32190


namespace NUMINAMATH_CALUDE_tens_digit_of_expression_l321_32114

theorem tens_digit_of_expression : ∃ n : ℕ, (2023^2024 - 2025 + 6) % 100 = 10 + 100 * n := by
  sorry

end NUMINAMATH_CALUDE_tens_digit_of_expression_l321_32114


namespace NUMINAMATH_CALUDE_billy_hike_distance_l321_32178

theorem billy_hike_distance :
  let east_distance : ℝ := 7
  let north_distance : ℝ := 3 * Real.sqrt 3
  let total_distance : ℝ := Real.sqrt (east_distance^2 + north_distance^2)
  total_distance = 2 * Real.sqrt 19 := by
  sorry

end NUMINAMATH_CALUDE_billy_hike_distance_l321_32178


namespace NUMINAMATH_CALUDE_prob_4_largest_l321_32154

def card_set : Finset ℕ := {1, 2, 3, 4, 5}

def draw_size : ℕ := 3

def prob_not_select_5 : ℚ := 2 / 5

def prob_not_select_4_and_5 : ℚ := 1 / 10

theorem prob_4_largest (s : Finset ℕ) (n : ℕ) 
  (h1 : s = card_set) 
  (h2 : n = draw_size) 
  (h3 : prob_not_select_5 = 2 / 5) 
  (h4 : prob_not_select_4_and_5 = 1 / 10) : 
  (prob_not_select_5 - prob_not_select_4_and_5 : ℚ) = 3 / 10 := by
  sorry

end NUMINAMATH_CALUDE_prob_4_largest_l321_32154


namespace NUMINAMATH_CALUDE_sector_triangle_area_equality_l321_32139

/-- Given a circle with center C and radius r, and an angle φ where 0 < φ < π/2,
    prove that the area of the circular sector formed by φ is equal to 
    the area of the triangle formed by the tangent line and the radius 
    if and only if tan φ = φ. -/
theorem sector_triangle_area_equality (φ : Real) (h1 : 0 < φ) (h2 : φ < π/2) :
  let r : Real := 1  -- Assuming unit circle for simplicity
  let sector_area : Real := (φ * r^2) / 2
  let triangle_area : Real := (r^2 * Real.tan φ) / 2
  sector_area = triangle_area ↔ Real.tan φ = φ := by
  sorry

end NUMINAMATH_CALUDE_sector_triangle_area_equality_l321_32139


namespace NUMINAMATH_CALUDE_largest_even_digit_multiple_of_5_proof_l321_32157

def has_only_even_digits (n : ℕ) : Prop :=
  ∀ d, d ∈ n.digits 10 → d % 2 = 0

def largest_even_digit_multiple_of_5 : ℕ := 86880

theorem largest_even_digit_multiple_of_5_proof :
  (has_only_even_digits largest_even_digit_multiple_of_5) ∧
  (largest_even_digit_multiple_of_5 < 100000) ∧
  (largest_even_digit_multiple_of_5 % 5 = 0) ∧
  (∀ m : ℕ, m > largest_even_digit_multiple_of_5 →
    ¬(has_only_even_digits m ∧ m < 100000 ∧ m % 5 = 0)) :=
by sorry

end NUMINAMATH_CALUDE_largest_even_digit_multiple_of_5_proof_l321_32157


namespace NUMINAMATH_CALUDE_smallest_class_size_exists_class_size_l321_32166

theorem smallest_class_size (n : ℕ) : 
  (n % 3 = 2) ∧ (n % 5 = 3) ∧ (n % 8 = 5) → n ≥ 53 :=
by sorry

theorem exists_class_size : 
  ∃ n : ℕ, (n % 3 = 2) ∧ (n % 5 = 3) ∧ (n % 8 = 5) ∧ n = 53 :=
by sorry

end NUMINAMATH_CALUDE_smallest_class_size_exists_class_size_l321_32166


namespace NUMINAMATH_CALUDE_football_team_analysis_l321_32175

/-- Represents a football team's performance in a season -/
structure FootballTeam where
  total_matches : ℕ
  played_matches : ℕ
  lost_matches : ℕ
  current_points : ℕ

/-- Calculates the number of wins given the team's performance -/
def wins (team : FootballTeam) : ℕ :=
  (team.current_points - (team.played_matches - team.lost_matches)) / 2

/-- Calculates the maximum possible points after all matches -/
def max_points (team : FootballTeam) : ℕ :=
  team.current_points + (team.total_matches - team.played_matches) * 3

/-- Calculates the minimum number of wins needed to reach a goal -/
def min_wins_needed (team : FootballTeam) (goal : ℕ) : ℕ :=
  ((goal - team.current_points) + 2) / 3

theorem football_team_analysis (team : FootballTeam)
  (h1 : team.total_matches = 16)
  (h2 : team.played_matches = 9)
  (h3 : team.lost_matches = 2)
  (h4 : team.current_points = 19) :
  wins team = 6 ∧
  max_points team = 40 ∧
  min_wins_needed team 34 = 4 := by
  sorry

#eval wins { total_matches := 16, played_matches := 9, lost_matches := 2, current_points := 19 }
#eval max_points { total_matches := 16, played_matches := 9, lost_matches := 2, current_points := 19 }
#eval min_wins_needed { total_matches := 16, played_matches := 9, lost_matches := 2, current_points := 19 } 34

end NUMINAMATH_CALUDE_football_team_analysis_l321_32175


namespace NUMINAMATH_CALUDE_mean_temperature_l321_32129

def temperatures : List ℤ := [-8, -3, -7, -6, 0, 4, 6, 5, -1, 2]

theorem mean_temperature :
  (List.sum temperatures : ℚ) / temperatures.length = -4/5 := by
  sorry

end NUMINAMATH_CALUDE_mean_temperature_l321_32129


namespace NUMINAMATH_CALUDE_solve_otimes_equation_l321_32126

-- Define the ⊗ operation
def otimes (a b : ℝ) : ℝ := a - 3 * b

-- State the theorem
theorem solve_otimes_equation :
  ∃! x : ℝ, otimes x 1 + otimes 2 x = 1 ∧ x = -1 := by
  sorry

end NUMINAMATH_CALUDE_solve_otimes_equation_l321_32126


namespace NUMINAMATH_CALUDE_mike_picked_52_peaches_l321_32193

/-- The number of peaches Mike picked -/
def peaches_picked (initial final : ℕ) : ℕ := final - initial

/-- Theorem stating that Mike picked 52 peaches -/
theorem mike_picked_52_peaches : peaches_picked 34 86 = 52 := by
  sorry

end NUMINAMATH_CALUDE_mike_picked_52_peaches_l321_32193


namespace NUMINAMATH_CALUDE_sum_remainder_theorem_l321_32160

def arithmetic_sequence (a₁ : ℕ) (d : ℕ) (aₙ : ℕ) : List ℕ :=
  let n := (aₙ - a₁) / d + 1
  List.range n |>.map (fun i => a₁ + i * d)

theorem sum_remainder_theorem (a₁ d aₙ : ℕ) (h₁ : a₁ = 3) (h₂ : d = 8) (h₃ : aₙ = 283) :
  (arithmetic_sequence a₁ d aₙ).sum % 8 = 4 := by
  sorry

end NUMINAMATH_CALUDE_sum_remainder_theorem_l321_32160


namespace NUMINAMATH_CALUDE_angies_age_l321_32167

theorem angies_age : ∃ (age : ℕ), 2 * age + 4 = 20 ∧ age = 8 := by
  sorry

end NUMINAMATH_CALUDE_angies_age_l321_32167


namespace NUMINAMATH_CALUDE_combined_value_of_a_and_b_l321_32162

/-- Given that 0.5% of a equals 95 paise and b is three times a minus 50,
    prove that the combined value of a and b is 710 rupees. -/
theorem combined_value_of_a_and_b (a b : ℝ) 
  (h1 : 0.005 * a = 95 / 100)  -- 0.5% of a equals 95 paise
  (h2 : b = 3 * a - 50)        -- b is three times a minus 50
  : a + b = 710 := by sorry

end NUMINAMATH_CALUDE_combined_value_of_a_and_b_l321_32162


namespace NUMINAMATH_CALUDE_C_is_hyperbola_l321_32194

/-- The curve C is defined by the equation 3y^2 - 4(x+1)y + 12(x-2) = 0 -/
def C : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | 3 * p.2^2 - 4 * (p.1 + 1) * p.2 + 12 * (p.1 - 2) = 0}

/-- The discriminant of the quadratic equation in y -/
def discriminant (x : ℝ) : ℝ :=
  16 * x^2 - 112 * x + 304

/-- Theorem: The curve C is a hyperbola -/
theorem C_is_hyperbola : ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧
  ∀ (p : ℝ × ℝ), p ∈ C ↔ (p.1^2 / a^2) - (p.2^2 / b^2) = 1 :=
sorry

end NUMINAMATH_CALUDE_C_is_hyperbola_l321_32194


namespace NUMINAMATH_CALUDE_french_toast_slices_l321_32174

/- Define the problem parameters -/
def weeks_per_year : ℕ := 52
def days_per_week : ℕ := 2
def loaves_used : ℕ := 26
def slices_per_loaf : ℕ := 12
def slices_for_daughters : ℕ := 1

/- Define the function to calculate slices per person -/
def slices_per_person : ℚ :=
  let total_slices := loaves_used * slices_per_loaf
  let total_days := weeks_per_year * days_per_week
  let slices_per_day := total_slices / total_days
  let slices_for_parents := slices_per_day - slices_for_daughters
  slices_for_parents / 2

/- State the theorem -/
theorem french_toast_slices :
  slices_per_person = 1 := by sorry

end NUMINAMATH_CALUDE_french_toast_slices_l321_32174


namespace NUMINAMATH_CALUDE_isosceles_trapezoid_circle_tangent_l321_32119

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a circle -/
structure Circle where
  center : Point
  radius : ℝ

/-- Represents an isosceles trapezoid -/
structure IsoscelesTrapezoid where
  A : Point
  B : Point
  C : Point
  D : Point

/-- Checks if a circle is tangent to a line segment -/
def isTangent (c : Circle) (p1 p2 : Point) : Prop := sorry

/-- Checks if a point lies on a line segment -/
def liesBetween (p : Point) (p1 p2 : Point) : Prop := sorry

/-- Calculates the distance between two points -/
def distance (p1 p2 : Point) : ℝ := sorry

/-- Main theorem -/
theorem isosceles_trapezoid_circle_tangent 
  (ABCD : IsoscelesTrapezoid) 
  (c : Circle) 
  (M N : Point) :
  isTangent c ABCD.A ABCD.B →
  isTangent c ABCD.B ABCD.C →
  liesBetween M ABCD.A ABCD.D →
  liesBetween N ABCD.C ABCD.D →
  distance ABCD.A M / distance ABCD.D M = 1 / 3 →
  distance ABCD.C N / distance ABCD.D N = 4 / 3 →
  distance ABCD.A ABCD.B = 7 →
  distance ABCD.A ABCD.D = 6 →
  distance ABCD.B ABCD.C = 4 + 2 * Real.sqrt 7 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_trapezoid_circle_tangent_l321_32119


namespace NUMINAMATH_CALUDE_sin_cos_equation_solvability_l321_32173

theorem sin_cos_equation_solvability (a : ℝ) :
  (∃ x : ℝ, Real.sin x ^ 2 + Real.cos x + a = 0) ↔ -5/4 ≤ a ∧ a ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_equation_solvability_l321_32173


namespace NUMINAMATH_CALUDE_f_sum_positive_l321_32188

def f (x : ℝ) := x^3 + x

theorem f_sum_positive (a b : ℝ) (h : a + b > 0) : f a + f b > 0 := by
  sorry

end NUMINAMATH_CALUDE_f_sum_positive_l321_32188


namespace NUMINAMATH_CALUDE_no_infinite_sequence_with_sqrt_property_l321_32100

theorem no_infinite_sequence_with_sqrt_property :
  ¬ (∃ (a : ℕ → ℕ), ∀ (n : ℕ), a (n + 2) = a (n + 1) + Real.sqrt (a (n + 1) + a n)) :=
by sorry

end NUMINAMATH_CALUDE_no_infinite_sequence_with_sqrt_property_l321_32100


namespace NUMINAMATH_CALUDE_min_even_integers_l321_32117

theorem min_even_integers (a b c d e f : ℤ) : 
  a + b = 24 →
  a + b + c + d = 39 →
  a + b + c + d + e + f = 58 →
  ∃ (count : ℕ), count ≥ 2 ∧ 
    count = (if Even a then 1 else 0) + 
            (if Even b then 1 else 0) + 
            (if Even c then 1 else 0) + 
            (if Even d then 1 else 0) + 
            (if Even e then 1 else 0) + 
            (if Even f then 1 else 0) ∧
    ∀ (other_count : ℕ), 
      other_count = (if Even a then 1 else 0) + 
                    (if Even b then 1 else 0) + 
                    (if Even c then 1 else 0) + 
                    (if Even d then 1 else 0) + 
                    (if Even e then 1 else 0) + 
                    (if Even f then 1 else 0) →
      other_count ≥ count := by
sorry

end NUMINAMATH_CALUDE_min_even_integers_l321_32117


namespace NUMINAMATH_CALUDE_quadratic_inequality_solutions_quadratic_inequality_always_negative_l321_32111

-- Define the quadratic function
def f (k : ℝ) (x : ℝ) : ℝ := k * x^2 - 2 * x + 6 * k

-- Define the solution set for the first case
def solution_set_1 (x : ℝ) : Prop := x < -3 ∨ x > -2

-- Define the solution set for the second case
def solution_set_2 : Set ℝ := Set.univ

theorem quadratic_inequality_solutions (k : ℝ) (h : k ≠ 0) :
  (∀ x, f k x < 0 ↔ solution_set_1 x) → k = -2/5 :=
sorry

theorem quadratic_inequality_always_negative (k : ℝ) (h : k ≠ 0) :
  (∀ x, f k x < 0) → k < -Real.sqrt 6 / 6 :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solutions_quadratic_inequality_always_negative_l321_32111


namespace NUMINAMATH_CALUDE_x_squared_plus_2xy_range_l321_32150

theorem x_squared_plus_2xy_range :
  ∀ x y : ℝ, x^2 + y^2 = 1 →
  (∃ (z : ℝ), z = x^2 + 2*x*y ∧ 1/2 - Real.sqrt 5 / 2 ≤ z ∧ z ≤ 1/2 + Real.sqrt 5 / 2) ∧
  (∃ (a b : ℝ), a = x^2 + 2*x*y ∧ b = x^2 + 2*x*y ∧ 
   a = 1/2 - Real.sqrt 5 / 2 ∧ b = 1/2 + Real.sqrt 5 / 2) :=
by sorry

end NUMINAMATH_CALUDE_x_squared_plus_2xy_range_l321_32150


namespace NUMINAMATH_CALUDE_boat_speed_in_still_water_l321_32164

/-- The speed of a boat in still water, given that the time taken to row upstream
    is twice the time taken to row downstream, and the speed of the stream is 12 kmph. -/
theorem boat_speed_in_still_water : ∃ (V_b : ℝ),
  (∀ (t : ℝ), t > 0 → (V_b + 12) * t = (V_b - 12) * (2 * t)) ∧ V_b = 36 := by
  sorry

#check boat_speed_in_still_water

end NUMINAMATH_CALUDE_boat_speed_in_still_water_l321_32164


namespace NUMINAMATH_CALUDE_intersection_equals_result_l321_32140

-- Define the sets M and N
def M : Set ℝ := {x | (x - 3) * Real.sqrt (x - 1) ≥ 0}
def N : Set ℝ := {x | (x - 3) * (x - 1) ≥ 0}

-- Define the intersection of M and N
def M_intersect_N : Set ℝ := M ∩ N

-- Theorem statement
theorem intersection_equals_result : M_intersect_N = {x | x ≥ 3 ∨ x = 1} := by
  sorry

end NUMINAMATH_CALUDE_intersection_equals_result_l321_32140


namespace NUMINAMATH_CALUDE_equation_solution_l321_32134

theorem equation_solution : ∃ x : ℝ, 45 - (x - (37 - (15 - 16))) = 55 ∧ x = 28 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l321_32134


namespace NUMINAMATH_CALUDE_fair_coin_five_flips_probability_l321_32102

/-- Represents the probability of a specific outcome when flipping a fair coin n times -/
def coin_flip_probability (n : ℕ) (heads : Finset ℕ) : ℚ :=
  (1 / 2) ^ n

theorem fair_coin_five_flips_probability :
  coin_flip_probability 5 {0, 1} = 1 / 32 := by
  sorry

end NUMINAMATH_CALUDE_fair_coin_five_flips_probability_l321_32102


namespace NUMINAMATH_CALUDE_solve_equation_1_solve_equation_2_l321_32130

-- Equation 1: x^2 - 6x + 1 = 0
theorem solve_equation_1 : 
  ∃ x₁ x₂ : ℝ, x₁ = 3 + 2 * Real.sqrt 2 ∧ 
             x₂ = 3 - 2 * Real.sqrt 2 ∧ 
             x₁^2 - 6*x₁ + 1 = 0 ∧ 
             x₂^2 - 6*x₂ + 1 = 0 := by
  sorry

-- Equation 2: 2x^2 + 3x - 5 = 0
theorem solve_equation_2 : 
  ∃ x₁ x₂ : ℝ, x₁ = 1 ∧ 
             x₂ = -5/2 ∧ 
             2*x₁^2 + 3*x₁ - 5 = 0 ∧ 
             2*x₂^2 + 3*x₂ - 5 = 0 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_1_solve_equation_2_l321_32130


namespace NUMINAMATH_CALUDE_f_monotone_and_inequality_l321_32147

noncomputable def f (x : ℝ) : ℝ := 2 * Real.log x - Real.log (x + 1) + Real.log (x - 1)

theorem f_monotone_and_inequality (k : ℝ) (h₁ : -1 ≤ k) (h₂ : k ≤ 0) :
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ x₁ > 1 ∧ x₂ > 1 ∧
  (∀ x > 1, ∀ y > 1, x < y → f x < f y) ∧
  (∀ x > 1, x * (f x₁ + f x₂) ≥ (x + 1) * (f x + 2 - 2*x)) :=
by sorry

end NUMINAMATH_CALUDE_f_monotone_and_inequality_l321_32147


namespace NUMINAMATH_CALUDE_unique_prime_sum_difference_l321_32107

theorem unique_prime_sum_difference : ∃! p : ℕ, 
  Prime p ∧ 
  (∃ x y z w : ℕ, Prime x ∧ Prime y ∧ Prime z ∧ Prime w ∧ 
    p = x + y ∧ p = z - w) :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_CALUDE_unique_prime_sum_difference_l321_32107


namespace NUMINAMATH_CALUDE_doubled_number_excess_l321_32172

theorem doubled_number_excess (x : ℝ) : x^2 = 25 → 2*x - x/5 = 9 := by
  sorry

end NUMINAMATH_CALUDE_doubled_number_excess_l321_32172


namespace NUMINAMATH_CALUDE_solve_equation_l321_32176

theorem solve_equation (x y : ℝ) (h1 : x = 12) (h2 : ((17.28 / x) / (y * 0.2)) = 2) : y = 3.6 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l321_32176


namespace NUMINAMATH_CALUDE_prove_a_equals_two_l321_32123

/-- Given two differentiable functions f and g on ℝ, prove that a = 2 -/
theorem prove_a_equals_two
  (f g : ℝ → ℝ)
  (hf : Differentiable ℝ f)
  (hg : Differentiable ℝ g)
  (h_g_nonzero : ∀ x, g x ≠ 0)
  (h_f_def : ∃ a : ℝ, a > 0 ∧ a ≠ 1 ∧ ∀ x, f x = 2 * a^x * g x)
  (h_inequality : ∀ x, f x * (deriv g x) < (deriv f x) * g x)
  (h_sum : f 1 / g 1 + f (-1) / g (-1) = 5) :
  ∃ a : ℝ, a = 2 ∧ a > 0 ∧ a ≠ 1 ∧ ∀ x, f x = 2 * a^x * g x :=
sorry

end NUMINAMATH_CALUDE_prove_a_equals_two_l321_32123


namespace NUMINAMATH_CALUDE_cafeteria_milk_stacks_l321_32165

/-- Given a total number of cartons and the number of cartons per stack, 
    calculate the maximum number of full stacks that can be made. -/
def maxFullStacks (totalCartons : ℕ) (cartonsPerStack : ℕ) : ℕ :=
  totalCartons / cartonsPerStack

theorem cafeteria_milk_stacks : maxFullStacks 799 6 = 133 := by
  sorry

end NUMINAMATH_CALUDE_cafeteria_milk_stacks_l321_32165


namespace NUMINAMATH_CALUDE_complex_perpendicular_l321_32195

theorem complex_perpendicular (z₁ z₂ : ℂ) (hz₁ : z₁ ≠ 0) (hz₂ : z₂ ≠ 0) :
  Complex.abs (z₁ + z₂) = Complex.abs (z₁ - z₂) → z₁.re * z₂.re + z₁.im * z₂.im = 0 :=
by sorry

end NUMINAMATH_CALUDE_complex_perpendicular_l321_32195


namespace NUMINAMATH_CALUDE_shirt_sales_theorem_l321_32192

/-- Represents the sales and profit data for a shirt selling business -/
structure ShirtSales where
  initial_sales : ℕ
  initial_profit : ℝ
  sales_increase : ℝ
  profit_decrease : ℝ

/-- Calculates the new sales quantity after a price reduction -/
def new_sales (data : ShirtSales) (reduction : ℝ) : ℝ :=
  data.initial_sales + data.sales_increase * reduction

/-- Calculates the new profit per piece after a price reduction -/
def new_profit_per_piece (data : ShirtSales) (reduction : ℝ) : ℝ :=
  data.initial_profit - reduction

/-- Calculates the total daily profit after a price reduction -/
def total_daily_profit (data : ShirtSales) (reduction : ℝ) : ℝ :=
  new_sales data reduction * new_profit_per_piece data reduction

/-- The main theorem about shirt sales and profit -/
theorem shirt_sales_theorem (data : ShirtSales) 
    (h1 : data.initial_sales = 20)
    (h2 : data.initial_profit = 40)
    (h3 : data.sales_increase = 2)
    (h4 : data.profit_decrease = 1) : 
    new_sales data 3 = 26 ∧ 
    ∃ x : ℝ, x = 20 ∧ total_daily_profit data x = 1200 := by
  sorry

end NUMINAMATH_CALUDE_shirt_sales_theorem_l321_32192


namespace NUMINAMATH_CALUDE_triangle_properties_l321_32189

structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

def ABC : Triangle := { A := (8, 5), B := (4, -2), C := (-6, 3) }

-- Equation of a line: ax + by + c = 0
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

def median_to_AC (t : Triangle) : Line := sorry

def altitude_to_AB (t : Triangle) : Line := sorry

def perpendicular_bisector_BC (t : Triangle) : Line := sorry

theorem triangle_properties :
  let m := median_to_AC ABC
  let h := altitude_to_AB ABC
  let p := perpendicular_bisector_BC ABC
  m.a = 2 ∧ m.b = 1 ∧ m.c = -6 ∧
  h.a = 4 ∧ h.b = 7 ∧ h.c = 3 ∧
  p.a = 2 ∧ p.b = -1 ∧ p.c = 5/2 := by sorry

end NUMINAMATH_CALUDE_triangle_properties_l321_32189


namespace NUMINAMATH_CALUDE_cross_section_area_theorem_l321_32116

/-- Regular quadrilateral prism with given dimensions -/
structure RegularQuadrilateralPrism where
  a : ℝ
  base_edge : ℝ
  height : ℝ
  h_base_edge : base_edge = a
  h_height : height = 2 * a

/-- Plane passing through diagonal B₁D₁ and midpoint of edge DC -/
structure CuttingPlane (prism : RegularQuadrilateralPrism) where
  diagonal : ℝ × ℝ × ℝ
  midpoint : ℝ × ℝ × ℝ
  h_diagonal : diagonal = (prism.a, prism.a, prism.height)
  h_midpoint : midpoint = (prism.a / 2, prism.a, 0)

/-- Area of cross-section created by cutting plane -/
noncomputable def cross_section_area (prism : RegularQuadrilateralPrism) (plane : CuttingPlane prism) : ℝ :=
  (3 * prism.a^2 * Real.sqrt 33) / 8

/-- Theorem stating the area of the cross-section -/
theorem cross_section_area_theorem (prism : RegularQuadrilateralPrism) (plane : CuttingPlane prism) :
  cross_section_area prism plane = (3 * prism.a^2 * Real.sqrt 33) / 8 := by
  sorry

end NUMINAMATH_CALUDE_cross_section_area_theorem_l321_32116


namespace NUMINAMATH_CALUDE_miquel_point_existence_l321_32196

-- Define the basic geometric objects
variable (A B C D H M N S T : Point)

-- Define the quadrilateral ABCD
def is_convex_cyclic_quadrilateral (A B C D : Point) : Prop := sorry

-- Define that ABCD is not a kite
def is_not_kite (A B C D : Point) : Prop := sorry

-- Define perpendicular diagonals
def perpendicular_diagonals (A B C D H : Point) : Prop := sorry

-- Define midpoints
def is_midpoint (M : Point) (B C : Point) : Prop := sorry

-- Define ray intersection
def ray_intersects (M H S A D : Point) : Prop := sorry

-- Define point outside quadrilateral
def point_outside_quadrilateral (E A B C D : Point) : Prop := sorry

-- Define angle bisector
def is_angle_bisector (E H B S : Point) : Prop := sorry

-- Define equal angles
def equal_angles (B E N M D : Point) : Prop := sorry

-- Main theorem
theorem miquel_point_existence 
  (h1 : is_convex_cyclic_quadrilateral A B C D)
  (h2 : is_not_kite A B C D)
  (h3 : perpendicular_diagonals A B C D H)
  (h4 : is_midpoint M B C)
  (h5 : is_midpoint N C D)
  (h6 : ray_intersects M H S A D)
  (h7 : ray_intersects N H T A B) :
  ∃ E : Point,
    point_outside_quadrilateral E A B C D ∧
    is_angle_bisector E H B S ∧
    is_angle_bisector E H T D ∧
    equal_angles B E N M D :=
sorry

end NUMINAMATH_CALUDE_miquel_point_existence_l321_32196


namespace NUMINAMATH_CALUDE_initial_cow_count_l321_32181

theorem initial_cow_count (x : ℕ) 
  (h1 : x - 31 + 75 = 83) : x = 39 := by
  sorry

end NUMINAMATH_CALUDE_initial_cow_count_l321_32181


namespace NUMINAMATH_CALUDE_barbaras_total_cost_l321_32141

/-- The cost of Barbara's purchase at the butcher's --/
def barbaras_purchase_cost (steak_weight : Real) (steak_price : Real) 
  (chicken_weight : Real) (chicken_price : Real) : Real :=
  steak_weight * steak_price + chicken_weight * chicken_price

/-- Theorem stating the total cost of Barbara's purchase --/
theorem barbaras_total_cost : 
  barbaras_purchase_cost 2 15 1.5 8 = 42 := by
  sorry

end NUMINAMATH_CALUDE_barbaras_total_cost_l321_32141


namespace NUMINAMATH_CALUDE_simple_interest_problem_l321_32148

/-- Given a principal sum and an interest rate, if increasing the rate by 5% over 10 years
    results in Rs. 600 more interest, then the principal sum must be Rs. 1200. -/
theorem simple_interest_problem (P R : ℝ) (h : P > 0) (r : R > 0) :
  (P * (R + 5) * 10) / 100 - (P * R * 10) / 100 = 600 →
  P = 1200 := by
sorry

end NUMINAMATH_CALUDE_simple_interest_problem_l321_32148


namespace NUMINAMATH_CALUDE_chess_tournament_participants_l321_32161

/-- Represents a chess tournament with the given property --/
structure ChessTournament where
  n : ℕ  -- Total number of players
  half_points_from_last_three : Prop  -- Property that each player scored half their points against the last three

/-- Theorem stating that a chess tournament satisfying the given condition has 9 participants --/
theorem chess_tournament_participants (t : ChessTournament) : t.n = 9 := by
  sorry

end NUMINAMATH_CALUDE_chess_tournament_participants_l321_32161


namespace NUMINAMATH_CALUDE_largest_c_value_l321_32159

theorem largest_c_value : ∃ (c : ℝ), (3 * c + 4) * (c - 2) = 9 * c ∧
  ∀ (x : ℝ), (3 * x + 4) * (x - 2) = 9 * x → x ≤ c ∧ c = 4 := by
  sorry

end NUMINAMATH_CALUDE_largest_c_value_l321_32159


namespace NUMINAMATH_CALUDE_quadratic_root_difference_l321_32122

theorem quadratic_root_difference (p q : ℝ) (hp : p > 0) (hq : q > 0) :
  (∃ x y : ℝ, x^2 + p*x + q = 0 ∧ y^2 + p*y + q = 0 ∧ (x - y)^2 = 9) →
  p = Real.sqrt (4*q + 9) :=
sorry

end NUMINAMATH_CALUDE_quadratic_root_difference_l321_32122


namespace NUMINAMATH_CALUDE_angles_on_y_axis_correct_l321_32142

/-- The set of angles whose terminal sides fall on the y-axis -/
def angles_on_y_axis : Set ℝ :=
  { α | ∃ k : ℤ, α = k * Real.pi + Real.pi / 2 }

/-- Theorem stating that angles_on_y_axis correctly represents
    the set of angles whose terminal sides fall on the y-axis -/
theorem angles_on_y_axis_correct :
  ∀ α : ℝ, α ∈ angles_on_y_axis ↔ 
    (∃ k : ℤ, α = k * Real.pi + Real.pi / 2) :=
by sorry

end NUMINAMATH_CALUDE_angles_on_y_axis_correct_l321_32142


namespace NUMINAMATH_CALUDE_sum_of_sequences_is_435_l321_32152

def sequence1 : List ℕ := [2, 14, 26, 38, 50]
def sequence2 : List ℕ := [12, 24, 36, 48, 60]
def sequence3 : List ℕ := [5, 15, 25, 35, 45]

theorem sum_of_sequences_is_435 :
  (sequence1.sum + sequence2.sum + sequence3.sum) = 435 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_sequences_is_435_l321_32152


namespace NUMINAMATH_CALUDE_largest_number_l321_32104

theorem largest_number (a b c d e : ℝ) 
  (ha : a = 0.998) 
  (hb : b = 0.989) 
  (hc : c = 0.999) 
  (hd : d = 0.990) 
  (he : e = 0.980) : 
  c ≥ a ∧ c ≥ b ∧ c ≥ d ∧ c ≥ e := by
  sorry

end NUMINAMATH_CALUDE_largest_number_l321_32104


namespace NUMINAMATH_CALUDE_complex_modulus_sqrt_two_l321_32146

theorem complex_modulus_sqrt_two (x y : ℝ) (h : (1 + Complex.I) * x = 1 + y * Complex.I) : 
  Complex.abs (x + y * Complex.I) = Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_complex_modulus_sqrt_two_l321_32146


namespace NUMINAMATH_CALUDE_q_of_q_of_q_2000_pow_2000_l321_32144

/-- Sum of digits of a natural number -/
def q (n : ℕ) : ℕ := sorry

/-- Theorem stating that q(q(q(2000^2000))) = 4 -/
theorem q_of_q_of_q_2000_pow_2000 : q (q (q (2000^2000))) = 4 := by sorry

end NUMINAMATH_CALUDE_q_of_q_of_q_2000_pow_2000_l321_32144


namespace NUMINAMATH_CALUDE_closest_to_quotient_l321_32184

def options : List ℝ := [500, 1500, 2500, 5000, 7500]

theorem closest_to_quotient (x : ℝ) (h : x ∈ options \ {2500}) :
  |503 / 0.198 - 2500| < |503 / 0.198 - x| :=
by sorry

end NUMINAMATH_CALUDE_closest_to_quotient_l321_32184


namespace NUMINAMATH_CALUDE_triangle_area_circumradius_l321_32145

theorem triangle_area_circumradius (a b c R : ℝ) (α β γ : ℝ) (S : ℝ) :
  a > 0 → b > 0 → c > 0 → R > 0 →
  α > 0 → β > 0 → γ > 0 →
  α + β + γ = π →
  a / Real.sin α = b / Real.sin β →
  b / Real.sin β = c / Real.sin γ →
  c / Real.sin γ = 2 * R →
  S = 1/2 * a * b * Real.sin γ →
  S = a * b * c / (4 * R) := by
sorry

end NUMINAMATH_CALUDE_triangle_area_circumradius_l321_32145


namespace NUMINAMATH_CALUDE_gcf_of_40_and_56_l321_32149

theorem gcf_of_40_and_56 : Nat.gcd 40 56 = 8 := by
  sorry

end NUMINAMATH_CALUDE_gcf_of_40_and_56_l321_32149


namespace NUMINAMATH_CALUDE_stripe_area_cylindrical_tower_l321_32120

/-- The area of a horizontal stripe wrapping twice around a cylindrical tower -/
theorem stripe_area_cylindrical_tower (d h w : ℝ) (hd : d = 25) (hh : h = 60) (hw : w = 2) :
  let circumference := π * d
  let stripe_length := 2 * circumference
  let stripe_area := stripe_length * w
  stripe_area = 100 * π :=
sorry

end NUMINAMATH_CALUDE_stripe_area_cylindrical_tower_l321_32120


namespace NUMINAMATH_CALUDE_sports_club_membership_l321_32185

theorem sports_club_membership (total : ℕ) (badminton : ℕ) (tennis : ℕ) (both : ℕ) : 
  total = 30 → badminton = 17 → tennis = 19 → both = 9 →
  total - (badminton + tennis - both) = 3 := by
  sorry

end NUMINAMATH_CALUDE_sports_club_membership_l321_32185


namespace NUMINAMATH_CALUDE_num_sam_sandwiches_l321_32171

/-- Represents the number of different types of bread available. -/
def num_breads : ℕ := 5

/-- Represents the number of different types of meat available. -/
def num_meats : ℕ := 7

/-- Represents the number of different types of cheese available. -/
def num_cheeses : ℕ := 6

/-- Represents the number of restricted sandwich combinations due to roast beef and swiss cheese. -/
def roast_beef_swiss_restrictions : ℕ := num_breads

/-- Represents the number of restricted sandwich combinations due to rye bread and turkey. -/
def rye_turkey_restrictions : ℕ := num_cheeses

/-- Represents the number of restricted sandwich combinations due to roast beef and rye bread. -/
def roast_beef_rye_restrictions : ℕ := num_cheeses

/-- The total number of possible sandwich combinations without restrictions. -/
def total_combinations : ℕ := num_breads * num_meats * num_cheeses

/-- The number of restricted sandwich combinations. -/
def total_restrictions : ℕ := roast_beef_swiss_restrictions + rye_turkey_restrictions + roast_beef_rye_restrictions

/-- Theorem stating the number of sandwiches Sam can order. -/
theorem num_sam_sandwiches : total_combinations - total_restrictions = 193 := by
  sorry

end NUMINAMATH_CALUDE_num_sam_sandwiches_l321_32171


namespace NUMINAMATH_CALUDE_car_speed_problem_l321_32179

theorem car_speed_problem (S : ℝ) : 
  (S * 1.3 + 10 = 205) → S = 150 := by
  sorry

end NUMINAMATH_CALUDE_car_speed_problem_l321_32179


namespace NUMINAMATH_CALUDE_parallel_vectors_x_value_l321_32112

/-- Given two parallel vectors a and b in R², prove that if a = (x, 3) and b = (4, 6), then x = 2 -/
theorem parallel_vectors_x_value (x : ℝ) :
  let a : ℝ × ℝ := (x, 3)
  let b : ℝ × ℝ := (4, 6)
  (∃ (k : ℝ), a = k • b) → x = 2 := by
sorry

end NUMINAMATH_CALUDE_parallel_vectors_x_value_l321_32112


namespace NUMINAMATH_CALUDE_three_solutions_iff_a_gt_two_l321_32177

/-- The equation x · |x-a| = 1 has exactly three distinct solutions if and only if a > 2 -/
theorem three_solutions_iff_a_gt_two (a : ℝ) :
  (∃ (x₁ x₂ x₃ : ℝ), x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃ ∧
    (∀ (x : ℝ), x * |x - a| = 1 ↔ x = x₁ ∨ x = x₂ ∨ x = x₃)) ↔
  a > 2 := by
  sorry

end NUMINAMATH_CALUDE_three_solutions_iff_a_gt_two_l321_32177


namespace NUMINAMATH_CALUDE_greenwood_school_quiz_l321_32180

theorem greenwood_school_quiz (f s : ℕ) (h1 : f > 0) (h2 : s > 0) :
  (3 * f : ℚ) / 4 = (s : ℚ) / 3 → s = 3 * f := by
  sorry

end NUMINAMATH_CALUDE_greenwood_school_quiz_l321_32180


namespace NUMINAMATH_CALUDE_remainder_problem_l321_32183

theorem remainder_problem : (245 * 15 - 20 * 8 + 5) % 17 = 1 := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l321_32183


namespace NUMINAMATH_CALUDE_log_equation_implies_non_square_non_cube_integer_l321_32191

-- Define the logarithm equation
def log_equation (x : ℝ) : Prop :=
  Real.log (343 : ℝ) / Real.log (3 * x + 1) = x

-- Define what it means to be a non-square, non-cube integer
def is_non_square_non_cube_integer (x : ℝ) : Prop :=
  ∃ n : ℤ, (x : ℝ) = n ∧ ¬∃ m : ℤ, n = m^2 ∧ ¬∃ k : ℤ, n = k^3

-- The theorem statement
theorem log_equation_implies_non_square_non_cube_integer :
  ∀ x : ℝ, log_equation x → is_non_square_non_cube_integer x :=
by sorry

end NUMINAMATH_CALUDE_log_equation_implies_non_square_non_cube_integer_l321_32191


namespace NUMINAMATH_CALUDE_random_walk_properties_l321_32197

/-- Represents a random walk on a line -/
structure RandomWalk where
  a : ℕ  -- number of steps to the right
  b : ℕ  -- number of steps to the left
  h : a > b

/-- The maximum possible range of the random walk -/
def max_range (w : RandomWalk) : ℕ := w.a

/-- The minimum possible range of the random walk -/
def min_range (w : RandomWalk) : ℕ := w.a - w.b

/-- The number of sequences that achieve the maximum range -/
def max_range_sequences (w : RandomWalk) : ℕ := w.b + 1

/-- Theorem stating the properties of the random walk -/
theorem random_walk_properties (w : RandomWalk) : 
  (max_range w = w.a) ∧ 
  (min_range w = w.a - w.b) ∧ 
  (max_range_sequences w = w.b + 1) := by
  sorry

end NUMINAMATH_CALUDE_random_walk_properties_l321_32197


namespace NUMINAMATH_CALUDE_triangle_middle_side_bound_l321_32121

theorem triangle_middle_side_bound (a b c : ℝ) (h_area : 1 = (1/2) * b * c * Real.sin α) 
  (h_order : a ≥ b ∧ b ≥ c) (h_positive : a > 0 ∧ b > 0 ∧ c > 0) (h_triangle : a < b + c) :
  b ≥ Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_middle_side_bound_l321_32121


namespace NUMINAMATH_CALUDE_ellipse_focus_coincides_with_center_l321_32132

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents an ellipse -/
structure Ellipse where
  center : Point
  semi_major_axis : ℝ
  semi_minor_axis : ℝ

/-- Returns the focus with larger x-coordinate for an ellipse -/
def focus_with_larger_x (e : Ellipse) : Point :=
  e.center

theorem ellipse_focus_coincides_with_center (e : Ellipse) 
    (h1 : e.center = ⟨3, -2⟩)
    (h2 : e.semi_major_axis = 3)
    (h3 : e.semi_minor_axis = 3) :
  focus_with_larger_x e = ⟨3, -2⟩ := by
  sorry

#check ellipse_focus_coincides_with_center

end NUMINAMATH_CALUDE_ellipse_focus_coincides_with_center_l321_32132


namespace NUMINAMATH_CALUDE_meeting_point_equation_correct_l321_32168

/-- Represents the time taken for two travelers to meet given their journey durations and a head start for one traveler. -/
def meeting_equation (x : ℚ) : Prop :=
  (x + 2) / 7 + x / 5 = 1

/-- The total journey time for the first traveler -/
def journey_time_A : ℚ := 5

/-- The total journey time for the second traveler -/
def journey_time_B : ℚ := 7

/-- The head start time for the second traveler -/
def head_start : ℚ := 2

/-- Theorem stating that the meeting equation correctly represents the meeting point of two travelers given the conditions -/
theorem meeting_point_equation_correct :
  ∃ x : ℚ, 
    x > 0 ∧ 
    x < journey_time_A ∧
    x + head_start < journey_time_B ∧
    meeting_equation x :=
sorry

end NUMINAMATH_CALUDE_meeting_point_equation_correct_l321_32168


namespace NUMINAMATH_CALUDE_percentage_increase_proof_l321_32198

theorem percentage_increase_proof (original_earnings new_earnings : ℝ) 
  (h1 : original_earnings = 60)
  (h2 : new_earnings = 84) :
  ((new_earnings - original_earnings) / original_earnings) * 100 = 40 := by
sorry

end NUMINAMATH_CALUDE_percentage_increase_proof_l321_32198


namespace NUMINAMATH_CALUDE_max_projection_area_parallelepiped_l321_32127

/-- The maximum area of the orthogonal projection of a rectangular parallelepiped -/
theorem max_projection_area_parallelepiped (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  ∃ (S : ℝ), S = a * Real.sqrt (a^2 + b^2) ∧
  ∀ (S' : ℝ), S' ≤ S :=
sorry

end NUMINAMATH_CALUDE_max_projection_area_parallelepiped_l321_32127


namespace NUMINAMATH_CALUDE_product_lcm_hcf_relation_l321_32103

theorem product_lcm_hcf_relation (a b : ℕ+) 
  (h_product : a * b = 571536)
  (h_lcm : Nat.lcm a b = 31096) :
  Nat.gcd a b = 18 := by
  sorry

end NUMINAMATH_CALUDE_product_lcm_hcf_relation_l321_32103


namespace NUMINAMATH_CALUDE_baking_on_thursday_l321_32135

/-- The number of days between Amrita's cake baking -/
def baking_cycle : ℕ := 5

/-- The number of days between Thursdays -/
def thursday_cycle : ℕ := 7

/-- The number of days until Amrita bakes a cake on a Thursday again -/
def days_until_thursday_baking : ℕ := 35

theorem baking_on_thursday :
  Nat.lcm baking_cycle thursday_cycle = days_until_thursday_baking := by
  sorry

end NUMINAMATH_CALUDE_baking_on_thursday_l321_32135


namespace NUMINAMATH_CALUDE_square_perimeter_l321_32153

/-- Given a square with area 400 square meters, its perimeter is 80 meters. -/
theorem square_perimeter (s : ℝ) (h : s^2 = 400) : 4 * s = 80 := by
  sorry

end NUMINAMATH_CALUDE_square_perimeter_l321_32153


namespace NUMINAMATH_CALUDE_roses_in_vase_l321_32143

/-- The number of roses in a vase after adding more roses -/
def total_roses (initial : ℕ) (added : ℕ) : ℕ :=
  initial + added

/-- Theorem stating that the total number of roses is 18 given the initial and added amounts -/
theorem roses_in_vase : total_roses 10 8 = 18 := by
  sorry

end NUMINAMATH_CALUDE_roses_in_vase_l321_32143


namespace NUMINAMATH_CALUDE_complex_power_2009_l321_32115

theorem complex_power_2009 (i : ℂ) (h : i^2 = -1) : i^2009 = i := by sorry

end NUMINAMATH_CALUDE_complex_power_2009_l321_32115


namespace NUMINAMATH_CALUDE_sequence_sum_l321_32187

theorem sequence_sum (x₁ x₂ x₃ x₄ x₅ x₆ x₇ : ℝ) :
  x₁ + 4*x₂ + 9*x₃ + 16*x₄ + 25*x₅ + 36*x₆ + 49*x₇ = 1 →
  4*x₁ + 9*x₂ + 16*x₃ + 25*x₄ + 36*x₅ + 49*x₆ + 64*x₇ = 12 →
  9*x₁ + 16*x₂ + 25*x₃ + 36*x₄ + 49*x₅ + 64*x₆ + 81*x₇ = 123 →
  16*x₁ + 25*x₂ + 36*x₃ + 49*x₄ + 64*x₅ + 81*x₆ + 100*x₇ = 334 := by
  sorry

end NUMINAMATH_CALUDE_sequence_sum_l321_32187


namespace NUMINAMATH_CALUDE_f_has_three_zeros_l321_32108

/-- The function f(x) = x^3 - bx^2 - 4 -/
def f (b : ℝ) (x : ℝ) : ℝ := x^3 - b*x^2 - 4

/-- The theorem stating that f has three distinct real zeros iff b < -3 -/
theorem f_has_three_zeros (b : ℝ) : 
  (∃ (x y z : ℝ), x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ f b x = 0 ∧ f b y = 0 ∧ f b z = 0) ↔ 
  b < -3 := by sorry

end NUMINAMATH_CALUDE_f_has_three_zeros_l321_32108


namespace NUMINAMATH_CALUDE_octal_sum_451_167_l321_32169

/-- Converts a base-8 number to base-10 --/
def octal_to_decimal (n : ℕ) : ℕ := sorry

/-- Converts a base-10 number to base-8 --/
def decimal_to_octal (n : ℕ) : ℕ := sorry

/-- The sum of two octal numbers in base 8 --/
def octal_sum (a b : ℕ) : ℕ := decimal_to_octal (octal_to_decimal a + octal_to_decimal b)

theorem octal_sum_451_167 : octal_sum 451 167 = 640 := by sorry

end NUMINAMATH_CALUDE_octal_sum_451_167_l321_32169


namespace NUMINAMATH_CALUDE_folded_square_perimeter_ratio_l321_32136

theorem folded_square_perimeter_ratio :
  let square_side : ℝ := 10
  let folded_width : ℝ := square_side / 2
  let folded_height : ℝ := square_side
  let triangle_perimeter : ℝ := folded_width + folded_height + Real.sqrt (folded_width ^ 2 + folded_height ^ 2)
  let pentagon_perimeter : ℝ := 2 * folded_height + folded_width + Real.sqrt (folded_width ^ 2 + folded_height ^ 2) + folded_width
  triangle_perimeter / pentagon_perimeter = (3 + Real.sqrt 5) / (6 + Real.sqrt 5) := by
  sorry

end NUMINAMATH_CALUDE_folded_square_perimeter_ratio_l321_32136


namespace NUMINAMATH_CALUDE_roots_negative_reciprocals_implies_a_eq_neg_c_l321_32182

-- Define the quadratic equation
def quadratic_equation (a b c x : ℝ) : Prop := a * x^2 + b * x + c = 0

-- Define the concept of roots
def is_root (r : ℝ) (a b c : ℝ) : Prop := quadratic_equation a b c r

-- Define negative reciprocals
def negative_reciprocals (r s : ℝ) : Prop := r = -1/s ∧ s = -1/r

-- Theorem statement
theorem roots_negative_reciprocals_implies_a_eq_neg_c 
  (a b c r s : ℝ) (h1 : is_root r a b c) (h2 : is_root s a b c) 
  (h3 : negative_reciprocals r s) : a = -c :=
sorry

end NUMINAMATH_CALUDE_roots_negative_reciprocals_implies_a_eq_neg_c_l321_32182


namespace NUMINAMATH_CALUDE_probability_theorem_l321_32158

def num_questions : ℕ := 5

def valid_sum (a b : ℕ) : Prop :=
  4 ≤ a + b ∧ a + b < 8

def num_valid_combinations : ℕ := 7

def total_combinations : ℕ := num_questions * (num_questions - 1) / 2

theorem probability_theorem :
  (num_valid_combinations : ℚ) / (total_combinations : ℚ) = 7 / 10 := by sorry

end NUMINAMATH_CALUDE_probability_theorem_l321_32158


namespace NUMINAMATH_CALUDE_remaining_oranges_l321_32106

def initial_oranges : ℝ := 77.0
def eaten_oranges : ℝ := 2.0

theorem remaining_oranges : initial_oranges - eaten_oranges = 75.0 := by
  sorry

end NUMINAMATH_CALUDE_remaining_oranges_l321_32106


namespace NUMINAMATH_CALUDE_find_A_l321_32199

theorem find_A : ∀ A B : ℕ,
  (A ≥ 1 ∧ A ≤ 9) →
  (B ≥ 0 ∧ B ≤ 9) →
  (10 * A + 3 ≥ 10 ∧ 10 * A + 3 ≤ 99) →
  (610 + B ≥ 100 ∧ 610 + B ≤ 999) →
  (10 * A + 3) + (610 + B) = 695 →
  A = 8 := by
sorry

end NUMINAMATH_CALUDE_find_A_l321_32199


namespace NUMINAMATH_CALUDE_necessary_condition_for_A_l321_32110

-- Define the set A
def A : Set ℝ := {x | (x - 2) / (x + 1) ≤ 0}

-- State the theorem
theorem necessary_condition_for_A (a : ℝ) :
  (∀ x ∈ A, x ≥ a) → a ≤ -1 := by
  sorry

end NUMINAMATH_CALUDE_necessary_condition_for_A_l321_32110


namespace NUMINAMATH_CALUDE_common_area_rectangle_circle_l321_32133

/-- The area of the region common to a 10 by 4 rectangle and a circle with radius 3, sharing the same center, is equal to 9π. -/
theorem common_area_rectangle_circle :
  let rectangle_width : ℝ := 10
  let rectangle_height : ℝ := 4
  let circle_radius : ℝ := 3
  let circle_area : ℝ := π * circle_radius^2
  (∀ x y, x^2 / (rectangle_width/2)^2 + y^2 / (rectangle_height/2)^2 ≤ 1 → x^2 + y^2 ≤ circle_radius^2) →
  circle_area = 9 * π :=
by sorry

end NUMINAMATH_CALUDE_common_area_rectangle_circle_l321_32133


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l321_32109

theorem quadratic_inequality_solution (a c : ℝ) : 
  (∀ x : ℝ, ax^2 + 2*x + c < 0 ↔ -1/3 < x ∧ x < 1/2) → 
  a = 12 ∧ c = -2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l321_32109


namespace NUMINAMATH_CALUDE_point_positions_l321_32105

/-- Define a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Check if a point is in the first octant -/
def isInFirstOctant (p : Point3D) : Prop :=
  p.x > 0 ∧ p.y > 0 ∧ p.z > 0

/-- Check if a point is in the second octant -/
def isInSecondOctant (p : Point3D) : Prop :=
  p.x < 0 ∧ p.y > 0 ∧ p.z > 0

/-- Check if a point is in the eighth octant -/
def isInEighthOctant (p : Point3D) : Prop :=
  p.x > 0 ∧ p.y < 0 ∧ p.z < 0

/-- Check if a point lies in the YOZ plane -/
def isInYOZPlane (p : Point3D) : Prop :=
  p.x = 0

/-- Check if a point lies on the OY axis -/
def isOnOYAxis (p : Point3D) : Prop :=
  p.x = 0 ∧ p.z = 0

/-- Check if a point is at the origin -/
def isAtOrigin (p : Point3D) : Prop :=
  p.x = 0 ∧ p.y = 0 ∧ p.z = 0

theorem point_positions :
  let A : Point3D := ⟨3, 2, 6⟩
  let B : Point3D := ⟨-2, 3, 1⟩
  let C : Point3D := ⟨1, -4, -2⟩
  let D : Point3D := ⟨1, -2, -1⟩
  let E : Point3D := ⟨0, 4, 1⟩
  let F : Point3D := ⟨0, 2, 0⟩
  let P : Point3D := ⟨0, 0, 0⟩
  isInFirstOctant A ∧
  isInSecondOctant B ∧
  isInEighthOctant C ∧
  isInEighthOctant D ∧
  isInYOZPlane E ∧
  isOnOYAxis F ∧
  isAtOrigin P := by
  sorry

end NUMINAMATH_CALUDE_point_positions_l321_32105


namespace NUMINAMATH_CALUDE_half_angle_quadrant_l321_32163

-- Define the concept of an angle being in a specific quadrant
def in_second_quadrant (α : Real) : Prop :=
  ∃ k : ℤ, k * 360 + 90 < α ∧ α < k * 360 + 180

def in_first_quadrant (α : Real) : Prop :=
  ∃ n : ℤ, n * 360 < α ∧ α < n * 360 + 90

def in_third_quadrant (α : Real) : Prop :=
  ∃ n : ℤ, n * 360 + 180 < α ∧ α < n * 360 + 270

-- State the theorem
theorem half_angle_quadrant (α : Real) :
  in_second_quadrant α → (in_first_quadrant (α/2) ∨ in_third_quadrant (α/2)) :=
by sorry

end NUMINAMATH_CALUDE_half_angle_quadrant_l321_32163


namespace NUMINAMATH_CALUDE_perpendicular_vectors_k_value_l321_32131

/-- Given plane vectors a and b, if ka + b is perpendicular to a, then k = -1/5 -/
theorem perpendicular_vectors_k_value (a b : ℝ × ℝ) (k : ℝ) 
    (h1 : a = (1, 2))
    (h2 : b = (-3, 2))
    (h3 : (k • a.1 + b.1) * a.1 + (k • a.2 + b.2) * a.2 = 0) :
  k = -1/5 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_k_value_l321_32131


namespace NUMINAMATH_CALUDE_percent_relation_l321_32137

theorem percent_relation (x y z : ℝ) 
  (h1 : 0.45 * z = 0.39 * y) 
  (h2 : y = 0.75 * x) : 
  z = 0.65 * x := by
sorry

end NUMINAMATH_CALUDE_percent_relation_l321_32137


namespace NUMINAMATH_CALUDE_virginia_average_rainfall_l321_32124

/-- The average rainfall in Virginia over five months --/
def average_rainfall (march april may june july : Float) : Float :=
  (march + april + may + june + july) / 5

/-- Theorem stating that the average rainfall in Virginia is 4 inches --/
theorem virginia_average_rainfall :
  average_rainfall 3.79 4.5 3.95 3.09 4.67 = 4 := by
  sorry

end NUMINAMATH_CALUDE_virginia_average_rainfall_l321_32124


namespace NUMINAMATH_CALUDE_amanda_kitchen_upgrade_cost_l321_32156

def kitchen_upgrade_cost (num_knobs : ℕ) (knob_price : ℚ) (num_pulls : ℕ) (pull_price : ℚ) : ℚ :=
  (num_knobs * knob_price) + (num_pulls * pull_price)

theorem amanda_kitchen_upgrade_cost :
  kitchen_upgrade_cost 18 (5/2) 8 4 = 77 := by
  sorry

end NUMINAMATH_CALUDE_amanda_kitchen_upgrade_cost_l321_32156


namespace NUMINAMATH_CALUDE_exactly_two_roots_l321_32186

def equation (x k : ℂ) : Prop :=
  x / (x + 1) + x / (x + 3) = k * x

theorem exactly_two_roots :
  ∃! k : ℂ, (∃ x y : ℂ, x ≠ y ∧ 
    (∀ z : ℂ, equation z k ↔ z = x ∨ z = y)) ↔ 
  k = (4 : ℂ) / 3 :=
sorry

end NUMINAMATH_CALUDE_exactly_two_roots_l321_32186


namespace NUMINAMATH_CALUDE_base4_addition_subtraction_l321_32138

/-- Converts a base 4 number represented as a list of digits to a natural number. -/
def base4ToNat (digits : List Nat) : Nat :=
  digits.foldl (fun acc d => 4 * acc + d) 0

/-- Converts a natural number to its base 4 representation as a list of digits. -/
def natToBase4 (n : Nat) : List Nat :=
  if n = 0 then [0] else
  let rec aux (m : Nat) (acc : List Nat) : List Nat :=
    if m = 0 then acc else aux (m / 4) ((m % 4) :: acc)
  aux n []

theorem base4_addition_subtraction :
  let a := base4ToNat [3, 2, 1]
  let b := base4ToNat [2, 0, 3]
  let c := base4ToNat [1, 1, 2]
  let result := base4ToNat [1, 0, 2, 1]
  (a + b) - c = result := by sorry

end NUMINAMATH_CALUDE_base4_addition_subtraction_l321_32138
