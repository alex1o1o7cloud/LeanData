import Mathlib

namespace outdoor_dining_area_expansion_l491_49152

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

end outdoor_dining_area_expansion_l491_49152


namespace solution_set_inequality_l491_49151

/-- Given that the solution set of ax^2 - bx - 1 ≥ 0 is [-1/2, -1/3], 
    prove that the solution set of ax^2 - bx - 1 < 0 is (2, 3) -/
theorem solution_set_inequality (a b : ℝ) : 
  (∀ x, ax^2 - b*x - 1 ≥ 0 ↔ x ∈ Set.Icc (-1/2) (-1/3)) →
  (∀ x, ax^2 - b*x - 1 < 0 ↔ x ∈ Set.Ioo 2 3) :=
by sorry

end solution_set_inequality_l491_49151


namespace andrews_age_l491_49171

theorem andrews_age (a g : ℚ) 
  (h1 : g = 10 * a)
  (h2 : g - (a + 2) = 57) :
  a = 59 / 9 := by
  sorry

end andrews_age_l491_49171


namespace quadratic_inequality_empty_solution_set_l491_49186

theorem quadratic_inequality_empty_solution_set (a : ℝ) : 
  (∀ x : ℝ, a * x^2 + 2 * a * x + 2 ≥ 0) → 0 ≤ a ∧ a ≤ 2 := by
  sorry

end quadratic_inequality_empty_solution_set_l491_49186


namespace quadrilateral_diagonal_length_l491_49124

/-- Represents a point in 2D space -/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/-- Represents a quadrilateral -/
structure Quadrilateral :=
  (W : Point)
  (X : Point)
  (Y : Point)
  (Z : Point)

/-- Checks if a quadrilateral is convex -/
def is_convex (q : Quadrilateral) : Prop :=
  sorry

/-- Calculates the distance between two points -/
def distance (p1 p2 : Point) : ℝ :=
  sorry

/-- Checks if two line segments intersect at right angles -/
def intersect_at_right_angle (p1 p2 p3 p4 : Point) : Prop :=
  sorry

/-- Checks if two line segments bisect each other -/
def bisect_each_other (p1 p2 p3 p4 : Point) : Prop :=
  sorry

theorem quadrilateral_diagonal_length 
  (q : Quadrilateral)
  (h1 : is_convex q)
  (h2 : distance q.W q.Y = 15)
  (h3 : distance q.X q.Z = 20)
  (h4 : distance q.W q.X = 18)
  (P : Point)
  (h5 : intersect_at_right_angle q.W q.X q.Y q.Z)
  (h6 : bisect_each_other q.W q.X q.Y q.Z) :
  distance q.W P = 9 :=
sorry

end quadrilateral_diagonal_length_l491_49124


namespace quadratic_equation_proof_l491_49165

theorem quadratic_equation_proof (m : ℝ) (p : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ x^2 - 2*x + m - 1 = 0 ∧ y^2 - 2*y + m - 1 = 0) →
  (p^2 - 2*p + m - 1 = 0) →
  ((p^2 - 2*p + 3) * (m + 4) = 7) →
  m = -3 :=
by sorry

end quadratic_equation_proof_l491_49165


namespace function_inequality_condition_l491_49129

open Real

/-- For the function f(x) = ln x - ax, where a ∈ ℝ and x ∈ (1, +∞),
    the inequality f(x) + a < 0 holds for all x in (1, +∞) if and only if a ≥ 1 -/
theorem function_inequality_condition (a : ℝ) :
  (∀ x : ℝ, x > 1 → (log x - a * x + a < 0)) ↔ a ≥ 1 := by
  sorry

end function_inequality_condition_l491_49129


namespace painting_ratio_l491_49139

theorem painting_ratio (monday : ℝ) (total : ℝ) : 
  monday = 30 →
  total = 105 →
  (total - (monday + 2 * monday)) / monday = 1 / 2 := by
sorry

end painting_ratio_l491_49139


namespace inequality_condition_l491_49106

theorem inequality_condition (a b : ℝ) :
  a * Real.sqrt a + b * Real.sqrt b > a * Real.sqrt b + b * Real.sqrt a →
  a ≥ 0 ∧ b ≥ 0 ∧ a ≠ b :=
by sorry

end inequality_condition_l491_49106


namespace sector_perimeter_ratio_l491_49195

theorem sector_perimeter_ratio (α : ℝ) (r R : ℝ) (h_positive : 0 < α ∧ 0 < r ∧ 0 < R) 
  (h_area_ratio : (α * r^2) / (α * R^2) = 1/4) : 
  (2*r + α*r) / (2*R + α*R) = 1/2 := by
sorry

end sector_perimeter_ratio_l491_49195


namespace coupon_discount_proof_l491_49103

/-- Calculates the discount given the costs and final amount paid -/
def calculate_discount (magazine_cost pencil_cost final_amount : ℚ) : ℚ :=
  magazine_cost + pencil_cost - final_amount

theorem coupon_discount_proof :
  let magazine_cost : ℚ := 85/100
  let pencil_cost : ℚ := 1/2
  let final_amount : ℚ := 1
  calculate_discount magazine_cost pencil_cost final_amount = 35/100 := by
  sorry

end coupon_discount_proof_l491_49103


namespace seating_arrangement_solution_l491_49194

/-- Represents a seating arrangement --/
structure SeatingArrangement where
  total_people : ℕ
  row_size_1 : ℕ
  row_size_2 : ℕ
  rows_of_size_1 : ℕ
  rows_of_size_2 : ℕ

/-- Defines a valid seating arrangement --/
def is_valid_arrangement (s : SeatingArrangement) : Prop :=
  s.total_people = s.row_size_1 * s.rows_of_size_1 + s.row_size_2 * s.rows_of_size_2

/-- The specific seating arrangement for our problem --/
def problem_arrangement : SeatingArrangement :=
  { total_people := 58
  , row_size_1 := 7
  , row_size_2 := 9
  , rows_of_size_1 := 7  -- This value is not given in the problem, but needed for the structure
  , rows_of_size_2 := 1  -- This is what we want to prove
  }

/-- The main theorem to prove --/
theorem seating_arrangement_solution :
  is_valid_arrangement problem_arrangement ∧
  ∀ s : SeatingArrangement,
    s.total_people = problem_arrangement.total_people ∧
    s.row_size_1 = problem_arrangement.row_size_1 ∧
    s.row_size_2 = problem_arrangement.row_size_2 ∧
    is_valid_arrangement s →
    s.rows_of_size_2 = problem_arrangement.rows_of_size_2 :=
by
  sorry

end seating_arrangement_solution_l491_49194


namespace hyperbola_properties_l491_49168

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

end hyperbola_properties_l491_49168


namespace sum_f_positive_l491_49185

def f (x : ℝ) : ℝ := x^3 + x

theorem sum_f_positive (a b c : ℝ) (hab : a + b > 0) (hbc : b + c > 0) (hca : c + a > 0) :
  f a + f b + f c > 0 := by
  sorry

end sum_f_positive_l491_49185


namespace unique_number_l491_49188

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def digit_product (n : ℕ) : ℕ :=
  (n / 10) * (n % 10)

def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

theorem unique_number : ∃! n : ℕ,
  is_two_digit n ∧
  Odd n ∧
  n % 9 = 0 ∧
  is_perfect_square (digit_product n) ∧
  n = 9 := by
sorry

end unique_number_l491_49188


namespace sqrt_200_simplification_l491_49184

theorem sqrt_200_simplification : Real.sqrt 200 = 10 * Real.sqrt 2 := by
  sorry

end sqrt_200_simplification_l491_49184


namespace line_passes_through_quadrants_l491_49104

theorem line_passes_through_quadrants 
  (a b c : ℝ) 
  (h1 : a * b < 0) 
  (h2 : b * c < 0) : 
  ∃ (x y : ℝ), 
    (a * x + b * y + c = 0) ∧ 
    ((x > 0 ∧ y > 0) ∨ (x < 0 ∧ y > 0) ∨ (x < 0 ∧ y < 0)) :=
by sorry

end line_passes_through_quadrants_l491_49104


namespace equation_equivalence_l491_49179

theorem equation_equivalence (x y : ℝ) : 
  (3 * x^2 + 4 * x + 7 * y + 2 = 0) ∧ (3 * x + 2 * y + 5 = 0) →
  4 * y^2 + 33 * y + 11 = 0 := by
  sorry

end equation_equivalence_l491_49179


namespace black_car_overtakes_l491_49116

/-- Represents the scenario of three cars racing on a highway -/
structure CarRace where
  red_speed : ℝ
  green_speed : ℝ
  black_speed : ℝ
  red_black_distance : ℝ
  black_green_distance : ℝ

/-- Theorem stating the condition for the black car to overtake the red car before the green car overtakes the black car -/
theorem black_car_overtakes (race : CarRace) 
  (h1 : race.red_speed = 40)
  (h2 : race.green_speed = 60)
  (h3 : race.red_black_distance = 10)
  (h4 : race.black_green_distance = 5)
  (h5 : race.black_speed > 40) :
  race.black_speed > 53.33 ↔ 
  (10 / (race.black_speed - 40) < 5 / (60 - race.black_speed)) := by
  sorry

end black_car_overtakes_l491_49116


namespace remainder_mod_five_l491_49175

theorem remainder_mod_five : (1234 * 1456 * 1789 * 2005 + 123) % 5 = 3 := by
  sorry

end remainder_mod_five_l491_49175


namespace prob_not_losing_l491_49145

/-- The probability of Hou Yifan winning a chess match against a computer -/
def prob_win : ℝ := 0.65

/-- The probability of a draw in a chess match between Hou Yifan and a computer -/
def prob_draw : ℝ := 0.25

/-- Theorem: The probability of Hou Yifan not losing is 0.9 -/
theorem prob_not_losing : prob_win + prob_draw = 0.9 := by
  sorry

end prob_not_losing_l491_49145


namespace ninetieth_term_is_13_l491_49137

def sequence_sum (n : ℕ) : ℕ := n * (n + 1) / 2

theorem ninetieth_term_is_13 :
  ∃ (seq : ℕ → ℕ),
    (∀ n : ℕ, ∀ k : ℕ, k > sequence_sum n → k ≤ sequence_sum (n + 1) → seq k = n + 1) →
    seq 90 = 13 :=
by
  sorry

end ninetieth_term_is_13_l491_49137


namespace linear_system_solution_l491_49125

theorem linear_system_solution (x y z : ℚ) : 
  x + 2 * y = 12 ∧ 
  y + 3 * z = 15 ∧ 
  3 * x - z = 6 → 
  x = 54 / 17 ∧ y = 75 / 17 ∧ z = 60 / 17 := by
sorry

end linear_system_solution_l491_49125


namespace sixth_power_of_sqrt_two_plus_sqrt_two_l491_49110

theorem sixth_power_of_sqrt_two_plus_sqrt_two :
  (Real.sqrt (2 + Real.sqrt 2)) ^ 6 = 16 + 10 * Real.sqrt 2 := by
  sorry

end sixth_power_of_sqrt_two_plus_sqrt_two_l491_49110


namespace hugo_prime_given_win_l491_49121

/-- The number of players in the game -/
def num_players : ℕ := 5

/-- The number of sides on the die -/
def die_sides : ℕ := 8

/-- The set of prime numbers on the die -/
def prime_rolls : Set ℕ := {2, 3, 5, 7}

/-- The probability of rolling a prime number -/
def prob_prime : ℚ := 1/2

/-- The probability of Hugo winning the game -/
def prob_hugo_wins : ℚ := 1/num_players

/-- The probability that all other players roll non-prime or smaller prime -/
def prob_others_smaller : ℚ := (1/2)^(num_players - 1)

/-- The main theorem: probability of Hugo's first roll being prime given he won -/
theorem hugo_prime_given_win : 
  (prob_prime * prob_others_smaller) / prob_hugo_wins = 5/32 := by sorry

end hugo_prime_given_win_l491_49121


namespace largest_prime_factor_l491_49182

theorem largest_prime_factor : 
  (∃ p : ℕ, Nat.Prime p ∧ p ∣ (16^4 + 2 * 16^2 + 1 - 13^4) ∧ 
    ∀ q : ℕ, Nat.Prime q → q ∣ (16^4 + 2 * 16^2 + 1 - 13^4) → q ≤ p) ∧
  Nat.Prime 71 ∧ 
  71 ∣ (16^4 + 2 * 16^2 + 1 - 13^4) :=
by sorry

end largest_prime_factor_l491_49182


namespace living_room_to_bedroom_ratio_l491_49101

/-- Energy usage of lights in Noah's house -/
def energy_usage (bedroom_watts_per_hour : ℝ) (hours : ℝ) (total_watts : ℝ) : Prop :=
  let bedroom_energy := bedroom_watts_per_hour * hours
  let office_energy := 3 * bedroom_energy
  let living_room_energy := total_watts - bedroom_energy - office_energy
  (living_room_energy / bedroom_energy = 4)

/-- Theorem: The ratio of living room light energy to bedroom light energy is 4:1 -/
theorem living_room_to_bedroom_ratio :
  energy_usage 6 2 96 := by
  sorry

end living_room_to_bedroom_ratio_l491_49101


namespace new_average_score_l491_49160

theorem new_average_score (n : ℕ) (initial_avg : ℚ) (new_score : ℚ) :
  n = 9 →
  initial_avg = 80 →
  new_score = 100 →
  (n * initial_avg + new_score) / (n + 1) = 82 := by
  sorry

end new_average_score_l491_49160


namespace functional_equation_solution_l491_49130

/-- A bounded real-valued function satisfying a specific functional equation. -/
def FunctionalEquation (f : ℝ → ℝ) : Prop :=
  (∃ M : ℝ, ∀ x, |f x| ≤ M) ∧ 
  (∀ x y, f (x * f y) + y * f x = x * f y + f (x * y))

/-- The theorem stating the only possible forms of f satisfying the functional equation. -/
theorem functional_equation_solution (f : ℝ → ℝ) (h : FunctionalEquation f) :
  (∀ x, f x = 0) ∨ 
  (∀ x, x < 0 → f x = -2*x) ∧ (∀ x, x ≥ 0 → f x = 0) :=
sorry

end functional_equation_solution_l491_49130


namespace simplify_polynomial_product_l491_49113

theorem simplify_polynomial_product (x : ℝ) :
  (3*x - 2) * (5*x^12 + 3*x^11 + 7*x^10 + 4*x^9 + x^8) =
  15*x^13 - x^12 + 15*x^11 - 2*x^10 - 5*x^9 - 2*x^8 := by
  sorry

end simplify_polynomial_product_l491_49113


namespace tunnel_length_l491_49136

/-- The length of a tunnel given train passage information -/
theorem tunnel_length (train_length : ℝ) (total_time : ℝ) (inside_time : ℝ) : 
  train_length = 300 →
  total_time = 60 →
  inside_time = 30 →
  ∃ (tunnel_length : ℝ) (train_speed : ℝ),
    tunnel_length + train_length = total_time * train_speed ∧
    tunnel_length - train_length = inside_time * train_speed ∧
    tunnel_length = 900 := by
  sorry

end tunnel_length_l491_49136


namespace best_standing_for_consistent_93rd_l491_49176

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

end best_standing_for_consistent_93rd_l491_49176


namespace john_classes_l491_49115

theorem john_classes (packs_per_student : ℕ) (students_per_class : ℕ) (total_packs : ℕ) 
  (h1 : packs_per_student = 2)
  (h2 : students_per_class = 30)
  (h3 : total_packs = 360) :
  total_packs / (packs_per_student * students_per_class) = 6 := by
  sorry

end john_classes_l491_49115


namespace tan_3_expression_zero_l491_49126

theorem tan_3_expression_zero (θ : Real) (h : Real.tan θ = 3) :
  (1 - Real.sin θ) / Real.cos θ - Real.cos θ / (1 + Real.sin θ) = 0 := by
  sorry

end tan_3_expression_zero_l491_49126


namespace only_two_and_five_plus_25_square_l491_49148

theorem only_two_and_five_plus_25_square (N : ℕ+) : 
  (∀ p : ℕ, Nat.Prime p → p ∣ N → (p = 2 ∨ p = 5)) →
  (∃ k : ℕ, N + 25 = k^2) →
  (N = 200 ∨ N = 2000) := by
sorry

end only_two_and_five_plus_25_square_l491_49148


namespace lollipop_consumption_days_l491_49161

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

end lollipop_consumption_days_l491_49161


namespace smallest_three_digit_candy_number_l491_49157

theorem smallest_three_digit_candy_number : ∃ n : ℕ,
  (100 ≤ n ∧ n < 1000) ∧
  (n - 7) % 9 = 0 ∧
  (n + 9) % 7 = 0 ∧
  (∀ m : ℕ, (100 ≤ m ∧ m < n) → (m - 7) % 9 ≠ 0 ∨ (m + 9) % 7 ≠ 0) ∧
  n = 124 := by
sorry

end smallest_three_digit_candy_number_l491_49157


namespace P_zero_equals_eleven_l491_49164

variables (a b c : ℝ) (P : ℝ → ℝ)

/-- The roots of the cubic equation -/
axiom root_equation : a^3 + 3*a^2 + 5*a + 7 = 0 ∧ 
                      b^3 + 3*b^2 + 5*b + 7 = 0 ∧ 
                      c^3 + 3*c^2 + 5*c + 7 = 0

/-- Properties of polynomial P -/
axiom P_properties : P a = b + c ∧ 
                     P b = a + c ∧ 
                     P c = a + b ∧ 
                     P (a + b + c) = -16

/-- Theorem: P(0) equals 11 -/
theorem P_zero_equals_eleven : P 0 = 11 := by sorry

end P_zero_equals_eleven_l491_49164


namespace initial_boarders_count_l491_49196

theorem initial_boarders_count (B D : ℕ) : 
  (B : ℚ) / D = 2 / 5 →  -- Original ratio
  ((B + 15 : ℚ) / D = 1 / 2) →  -- New ratio after 15 boarders joined
  B = 60 := by
sorry

end initial_boarders_count_l491_49196


namespace exclusive_or_implies_disjunction_l491_49153

theorem exclusive_or_implies_disjunction (p q : Prop) : 
  ((p ∧ ¬q) ∨ (¬p ∧ q)) → (p ∨ q) :=
by
  sorry

end exclusive_or_implies_disjunction_l491_49153


namespace tommys_balloons_l491_49181

theorem tommys_balloons (initial_balloons : ℝ) (mom_gave : ℝ) (total_balloons : ℝ)
  (h1 : mom_gave = 78.5)
  (h2 : total_balloons = 132.25)
  (h3 : total_balloons = initial_balloons + mom_gave) :
  initial_balloons = 53.75 := by
  sorry

end tommys_balloons_l491_49181


namespace sqrt_27_div_sqrt_3_equals_3_l491_49193

theorem sqrt_27_div_sqrt_3_equals_3 : Real.sqrt 27 / Real.sqrt 3 = 3 := by
  sorry

end sqrt_27_div_sqrt_3_equals_3_l491_49193


namespace littleTwelve_game_count_l491_49144

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

end littleTwelve_game_count_l491_49144


namespace h_satisfies_equation_l491_49169

-- Define the polynomial h(x)
def h (x : ℝ) : ℝ := -2*x^5 - x^3 + 5*x^2 - 6*x - 3

-- State the theorem
theorem h_satisfies_equation : 
  ∀ x : ℝ, 2*x^5 + 4*x^3 - 3*x^2 + x + 7 + h x = -x^3 + 2*x^2 - 5*x + 4 :=
by
  sorry

end h_satisfies_equation_l491_49169


namespace abs_plus_one_minimum_l491_49114

theorem abs_plus_one_minimum :
  ∃ (min : ℝ) (x₀ : ℝ), (∀ x : ℝ, min ≤ |x| + 1) ∧ (min = |x₀| + 1) ∧ (min = 1 ∧ x₀ = 0) :=
by sorry

end abs_plus_one_minimum_l491_49114


namespace remaining_calories_l491_49108

-- Define the given conditions
def calories_per_serving : ℕ := 110
def servings_per_block : ℕ := 16
def servings_eaten : ℕ := 5

-- Define the theorem
theorem remaining_calories :
  (servings_per_block - servings_eaten) * calories_per_serving = 1210 := by
  sorry

end remaining_calories_l491_49108


namespace prob_hit_third_shot_prob_hit_at_least_once_l491_49109

-- Define the probability of hitting the target in one shot
def hit_probability : ℝ := 0.9

-- Define the number of shots
def num_shots : ℕ := 4

-- Theorem for the probability of hitting the target on the 3rd shot
theorem prob_hit_third_shot : 
  hit_probability = 0.9 := by sorry

-- Theorem for the probability of hitting the target at least once
theorem prob_hit_at_least_once : 
  1 - (1 - hit_probability) ^ num_shots = 1 - 0.1 ^ 4 := by sorry

end prob_hit_third_shot_prob_hit_at_least_once_l491_49109


namespace value_preserving_2x_squared_value_preserving_x_squared_minus_2x_plus_m_l491_49132

/-- A function is value-preserving on an interval [a, b] if it is monotonic
and its range on [a, b] is exactly [a, b] -/
def is_value_preserving (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  a < b ∧ 
  (∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → (f x < f y ∨ f y < f x)) ∧
  (∀ y, a ≤ y ∧ y ≤ b → ∃ x, a ≤ x ∧ x ≤ b ∧ f x = y)

/-- The function f(x) = 2x² has a unique value-preserving interval [0, 1/2] -/
theorem value_preserving_2x_squared :
  ∃! (a b : ℝ), is_value_preserving (fun x ↦ 2 * x^2) a b ∧ a = 0 ∧ b = 1/2 :=
sorry

/-- The function g(x) = x² - 2x + m has value-preserving intervals
if and only if m ∈ [1, 5/4) ∪ [2, 9/4) -/
theorem value_preserving_x_squared_minus_2x_plus_m (m : ℝ) :
  (∃ a b, is_value_preserving (fun x ↦ x^2 - 2*x + m) a b) ↔ 
  (1 ≤ m ∧ m < 5/4) ∨ (2 ≤ m ∧ m < 9/4) :=
sorry

end value_preserving_2x_squared_value_preserving_x_squared_minus_2x_plus_m_l491_49132


namespace correct_option_b_l491_49162

variable (y : ℝ)

theorem correct_option_b (y : ℝ) :
  (-2 * y^3) * (-y) = 2 * y^4 ∧
  (-y^3) * (-y) ≠ -y ∧
  ((-2*y)^3) * (-y) ≠ -8 * y^4 ∧
  ((-y)^12) * (-y) ≠ -3 * y^13 :=
sorry

end correct_option_b_l491_49162


namespace c_to_a_ratio_l491_49107

/-- Represents the share of money for each person in Rupees -/
structure Shares where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Represents the conditions of the problem -/
def ProblemConditions (s : Shares) : Prop :=
  s.c = 56 ∧ 
  s.a + s.b + s.c = 287 ∧ 
  s.b = 0.65 * s.a

/-- Theorem stating the ratio of C's share to A's share in paisa -/
theorem c_to_a_ratio (s : Shares) 
  (h : ProblemConditions s) : (s.c * 100) / (s.a * 100) = 0.4 := by
  sorry

#check c_to_a_ratio

end c_to_a_ratio_l491_49107


namespace intersection_line_of_two_circles_l491_49166

/-- Given two circles with equations x^2 + y^2 + 4x - 4y - 1 = 0 and x^2 + y^2 + 2x - 13 = 0,
    the line passing through their intersection points has the equation x - 2y + 6 = 0 -/
theorem intersection_line_of_two_circles (x y : ℝ) : 
  (x^2 + y^2 + 4*x - 4*y - 1 = 0) ∧ (x^2 + y^2 + 2*x - 13 = 0) →
  (x - 2*y + 6 = 0) :=
by sorry

end intersection_line_of_two_circles_l491_49166


namespace f_derivative_at_zero_l491_49190

noncomputable def f (x : ℝ) : ℝ := (x + 2) * Real.exp x

theorem f_derivative_at_zero : 
  deriv f 0 = 3 := by sorry

end f_derivative_at_zero_l491_49190


namespace min_value_expression_equality_condition_l491_49191

theorem min_value_expression (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  4 * a^3 + 8 * b^3 + 18 * c^3 + 1 / (9 * a * b * c) ≥ 8 / Real.sqrt 3 :=
by sorry

theorem equality_condition (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (4 * a^3 + 8 * b^3 + 18 * c^3 + 1 / (9 * a * b * c) = 8 / Real.sqrt 3) ↔
  (4 * a^3 = 8 * b^3 ∧ 8 * b^3 = 18 * c^3 ∧ 24 * a * b * c = 1 / (9 * a * b * c)) :=
by sorry

end min_value_expression_equality_condition_l491_49191


namespace speed_conversion_l491_49119

/-- Conversion factor from kilometers per hour to meters per second -/
def kmph_to_ms : ℝ := 0.277778

/-- Given speed in kilometers per hour -/
def given_speed : ℝ := 252

/-- Equivalent speed in meters per second -/
def equivalent_speed : ℝ := 70

/-- Theorem stating that the given speed in kmph is equal to the equivalent speed in m/s -/
theorem speed_conversion :
  given_speed * kmph_to_ms = equivalent_speed := by sorry

end speed_conversion_l491_49119


namespace regular_pentagon_perimeter_l491_49183

/-- The perimeter of a regular pentagon with side length 15 cm is 75 cm. -/
theorem regular_pentagon_perimeter :
  ∀ (side_length perimeter : ℝ),
  side_length = 15 →
  perimeter = 5 * side_length →
  perimeter = 75 :=
by
  sorry

end regular_pentagon_perimeter_l491_49183


namespace wage_difference_l491_49156

theorem wage_difference (w1 w2 : ℝ) 
  (h1 : w1 > 0) 
  (h2 : w2 > 0) 
  (h3 : 0.4 * w2 = 1.6 * (0.2 * w1)) : 
  (w1 - w2) / w1 = 0.2 := by
  sorry

end wage_difference_l491_49156


namespace product_evaluation_l491_49177

theorem product_evaluation : (3 + 1) * (3^2 + 1^2) * (3^4 + 1^4) = 3280 := by
  sorry

end product_evaluation_l491_49177


namespace connie_tickets_connie_redeemed_fifty_tickets_l491_49142

theorem connie_tickets : ℕ → Prop := fun total =>
  let koala := total / 2
  let earbuds := 10
  let bracelets := 15
  (koala + earbuds + bracelets = total) → total = 50

-- Proof
theorem connie_redeemed_fifty_tickets : ∃ total, connie_tickets total :=
  sorry

end connie_tickets_connie_redeemed_fifty_tickets_l491_49142


namespace geometric_sequence_property_l491_49150

/-- A geometric sequence -/
def geometric_sequence (α : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, α (n + 1) = r * α n

/-- The theorem stating that if α_4 · α_5 · α_6 = 27 in a geometric sequence, then α_5 = 3 -/
theorem geometric_sequence_property (α : ℕ → ℝ) :
  geometric_sequence α → α 4 * α 5 * α 6 = 27 → α 5 = 3 := by
  sorry

end geometric_sequence_property_l491_49150


namespace dot_product_equation_is_line_l491_49127

/-- Represents a 2D vector -/
structure Vec2D where
  x : ℝ
  y : ℝ

/-- Dot product of two 2D vectors -/
def dot (v w : Vec2D) : ℝ := v.x * w.x + v.y * w.y

/-- Theorem stating that the equation r ⋅ a = m represents a line -/
theorem dot_product_equation_is_line (a : Vec2D) (m : ℝ) :
  ∃ (A B C : ℝ), ∀ (r : Vec2D), dot r a = m ↔ A * r.x + B * r.y + C = 0 := by
  sorry

end dot_product_equation_is_line_l491_49127


namespace square_difference_l491_49172

theorem square_difference (a b : ℝ) (h1 : a + b = 10) (h2 : a - b = 4) : a^2 - b^2 = 40 := by
  sorry

end square_difference_l491_49172


namespace imaginary_part_of_z_l491_49141

theorem imaginary_part_of_z (i : ℂ) (h : i * i = -1) : 
  Complex.im (i / (i - 1)) = -1/2 := by sorry

end imaginary_part_of_z_l491_49141


namespace pam_current_age_l491_49105

/-- Represents a person's age -/
structure Age where
  years : ℕ

/-- Represents the current state -/
structure CurrentState where
  pam_age : Age
  rena_age : Age

/-- Represents the future state after 10 years -/
structure FutureState where
  pam_age : Age
  rena_age : Age

/-- The conditions of the problem -/
def problem_conditions (current : CurrentState) (future : FutureState) : Prop :=
  (current.pam_age.years * 2 = current.rena_age.years) ∧
  (future.rena_age.years = future.pam_age.years + 5) ∧
  (future.pam_age.years = current.pam_age.years + 10) ∧
  (future.rena_age.years = current.rena_age.years + 10)

/-- The theorem to prove -/
theorem pam_current_age
  (current : CurrentState)
  (future : FutureState)
  (h : problem_conditions current future) :
  current.pam_age.years = 5 := by
  sorry

end pam_current_age_l491_49105


namespace modulo_six_equivalence_l491_49149

theorem modulo_six_equivalence : 47^1987 - 22^1987 ≡ 1 [ZMOD 6] := by sorry

end modulo_six_equivalence_l491_49149


namespace jenny_jump_distance_l491_49187

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

end jenny_jump_distance_l491_49187


namespace sin_2012_deg_l491_49100

theorem sin_2012_deg : Real.sin (2012 * π / 180) = -Real.sin (32 * π / 180) := by
  sorry

end sin_2012_deg_l491_49100


namespace exactly_ten_maas_l491_49170

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

end exactly_ten_maas_l491_49170


namespace xy_value_l491_49173

theorem xy_value (x y : ℝ) 
  (h1 : (4:ℝ)^x / (2:ℝ)^(x+y) = 16)
  (h2 : (9:ℝ)^(x+y) / (3:ℝ)^(5*y) = 81) : 
  x * y = 32 := by
  sorry

end xy_value_l491_49173


namespace box_difference_b_and_d_l491_49123

/-- Represents the number of boxes of table tennis balls taken by each person. -/
structure BoxCount where
  a : ℕ  -- Number of boxes taken by A
  b : ℕ  -- Number of boxes taken by B
  c : ℕ  -- Number of boxes taken by C
  d : ℕ  -- Number of boxes taken by D

/-- Represents the money owed between individuals. -/
structure MoneyOwed where
  a_to_c : ℕ  -- Amount A owes to C
  b_to_d : ℕ  -- Amount B owes to D

/-- Theorem stating the difference in boxes between B and D is 18. -/
theorem box_difference_b_and_d (boxes : BoxCount) (money : MoneyOwed) : 
  boxes.b = boxes.a + 4 →  -- A took 4 boxes less than B
  boxes.d = boxes.c + 8 →  -- C took 8 boxes less than D
  money.a_to_c = 112 →     -- A owes C 112 yuan
  money.b_to_d = 72 →      -- B owes D 72 yuan
  boxes.b - boxes.d = 18 := by
  sorry

#check box_difference_b_and_d

end box_difference_b_and_d_l491_49123


namespace max_shoe_pairs_l491_49198

theorem max_shoe_pairs (initial_pairs : ℕ) (lost_shoes : ℕ) (max_remaining_pairs : ℕ) : 
  initial_pairs = 27 → lost_shoes = 9 → max_remaining_pairs = 18 →
  max_remaining_pairs = initial_pairs - lost_shoes / 2 := by
sorry

end max_shoe_pairs_l491_49198


namespace melanie_attended_games_l491_49192

theorem melanie_attended_games 
  (total_games : ℕ) 
  (missed_games : ℕ) 
  (attended_games : ℕ) 
  (h1 : total_games = 64) 
  (h2 : missed_games = 32) 
  (h3 : attended_games = total_games - missed_games) : 
  attended_games = 32 :=
by sorry

end melanie_attended_games_l491_49192


namespace f_properties_l491_49158

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

end f_properties_l491_49158


namespace white_tile_count_in_specific_arrangement_l491_49140

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

end white_tile_count_in_specific_arrangement_l491_49140


namespace complex_magnitude_product_l491_49174

theorem complex_magnitude_product : Complex.abs ((3 * Real.sqrt 2 - 3 * Complex.I) * (2 * Real.sqrt 3 + 6 * Complex.I)) = 36 := by
  sorry

end complex_magnitude_product_l491_49174


namespace hyperbola_asymptote_slope_l491_49133

/-- Given a hyperbola with equation x²/144 - y²/81 = 1, prove that the slope of its asymptotes is 3/4 -/
theorem hyperbola_asymptote_slope :
  ∃ (m : ℚ), (∀ (x y : ℚ), x^2 / 144 - y^2 / 81 = 1 →
    (y = m * x ∨ y = -m * x) → m = 3/4) :=
by sorry

end hyperbola_asymptote_slope_l491_49133


namespace max_ab_and_min_fraction_l491_49112

theorem max_ab_and_min_fraction (a b x y : ℝ) : 
  a > 0 → b > 0 → 4 * a + b = 1 → 
  x > 0 → y > 0 → x + y = 1 → 
  (∀ a' b', a' > 0 → b' > 0 → 4 * a' + b' = 1 → a * b ≥ a' * b') ∧ 
  (∀ x' y', x' > 0 → y' > 0 → x' + y' = 1 → 4 / x + 9 / y ≤ 4 / x' + 9 / y') ∧
  a * b = 1 / 16 ∧ 
  4 / x + 9 / y = 25 := by
sorry

end max_ab_and_min_fraction_l491_49112


namespace fraction_sum_20_equals_10_9_l491_49146

def fraction_sum (n : ℕ) : ℚ :=
  (Finset.range n).sum (λ i => 2 / ((i + 1) * (i + 4)))

theorem fraction_sum_20_equals_10_9 : fraction_sum 20 = 10 / 9 := by
  sorry

end fraction_sum_20_equals_10_9_l491_49146


namespace max_unique_sums_l491_49122

def nickel : ℕ := 5
def dime : ℕ := 10
def quarter : ℕ := 25
def half_dollar : ℕ := 50

def coin_set : List ℕ := [nickel, nickel, nickel, dime, dime, dime, quarter, quarter, half_dollar, half_dollar]

def unique_sums (coins : List ℕ) : Finset ℕ :=
  (do
    let c1 <- coins
    let c2 <- coins
    pure (c1 + c2)
  ).toFinset

theorem max_unique_sums :
  Finset.card (unique_sums coin_set) = 10 := by sorry

end max_unique_sums_l491_49122


namespace max_gcd_of_coprime_linear_combination_l491_49197

theorem max_gcd_of_coprime_linear_combination (m n : ℕ) :
  Nat.gcd m n = 1 →
  ∃ a b : ℕ, Nat.gcd (m + 2000 * n) (n + 2000 * m) = 2000^2 - 1 ∧
            ∀ c d : ℕ, Nat.gcd (c + 2000 * d) (d + 2000 * c) ≤ 2000^2 - 1 :=
by sorry

end max_gcd_of_coprime_linear_combination_l491_49197


namespace lily_pad_half_coverage_l491_49199

/-- Represents the number of days it takes for lily pads to cover the entire lake -/
def full_coverage_days : ℕ := 39

/-- Represents the growth factor of lily pads per day -/
def daily_growth_factor : ℕ := 2

/-- Calculates the number of days required to cover half the lake -/
def half_coverage_days : ℕ := full_coverage_days - 1

theorem lily_pad_half_coverage :
  half_coverage_days = 38 :=
sorry

end lily_pad_half_coverage_l491_49199


namespace javelin_throw_distance_l491_49138

theorem javelin_throw_distance (first second third : ℝ) 
  (h1 : first = 2 * second)
  (h2 : first = (1 / 2) * third)
  (h3 : first + second + third = 1050) :
  first = 300 := by
  sorry

end javelin_throw_distance_l491_49138


namespace f_100_equals_2_l491_49118

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 0 then x^3 else Real.log x

-- Theorem statement
theorem f_100_equals_2 : f 100 = 2 := by
  sorry

end f_100_equals_2_l491_49118


namespace z_purely_imaginary_and_fourth_quadrant_l491_49143

def z (m : ℝ) : ℂ := Complex.mk (m^2 - 8*m + 15) (m^2 - 5*m)

theorem z_purely_imaginary_and_fourth_quadrant :
  (∃! m : ℝ, (z m).re = 0 ∧ (z m).im ≠ 0 ∧ m = 3) ∧
  (¬∃ m : ℝ, (z m).re > 0 ∧ (z m).im < 0) := by
  sorry

end z_purely_imaginary_and_fourth_quadrant_l491_49143


namespace range_of_a_l491_49163

theorem range_of_a (a : ℝ) : 
  (∀ x ∈ (Set.Ioo 0 1), (x + Real.log a) / Real.exp x - a * Real.log x / x > 0) →
  a ∈ Set.Icc (Real.exp (-1)) 1 ∧ a ≠ 1 :=
sorry

end range_of_a_l491_49163


namespace two_distinct_roots_l491_49178

/-- The equation has exactly two distinct real roots for x when p is in the specified range -/
theorem two_distinct_roots (p : ℝ) : 
  (∃! x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
    Real.sqrt (2*p + 1 - x₁^2) + Real.sqrt (3*x₁ + p + 4) = Real.sqrt (x₁^2 + 9*x₁ + 3*p + 9) ∧
    Real.sqrt (2*p + 1 - x₂^2) + Real.sqrt (3*x₂ + p + 4) = Real.sqrt (x₂^2 + 9*x₂ + 3*p + 9)) ↔
  (-1/4 < p ∧ p ≤ 0) ∨ p ≥ 2 :=
sorry

end two_distinct_roots_l491_49178


namespace paperboy_delivery_ways_l491_49154

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

end paperboy_delivery_ways_l491_49154


namespace absolute_value_inequality_l491_49120

theorem absolute_value_inequality (y : ℝ) : 
  |((8 - 2*y) / 4)| < 3 ↔ -2 < y ∧ y < 10 := by sorry

end absolute_value_inequality_l491_49120


namespace add_decimals_l491_49117

theorem add_decimals : (7.56 : ℝ) + (4.29 : ℝ) = 11.85 := by sorry

end add_decimals_l491_49117


namespace x_plus_y_equals_negative_eight_l491_49131

theorem x_plus_y_equals_negative_eight 
  (h1 : |x| + x - y = 16) 
  (h2 : x - |y| + y = -8) : 
  x + y = -8 := by sorry

end x_plus_y_equals_negative_eight_l491_49131


namespace tims_weekend_ride_distance_l491_49134

/-- Tim's weekly biking schedule and distance calculation -/
theorem tims_weekend_ride_distance 
  (work_distance : ℝ) 
  (work_days : ℕ) 
  (speed : ℝ) 
  (total_biking_hours : ℝ) 
  (h1 : work_distance = 20)
  (h2 : work_days = 5)
  (h3 : speed = 25)
  (h4 : total_biking_hours = 16) :
  let workday_distance := 2 * work_distance * work_days
  let workday_hours := workday_distance / speed
  let weekend_hours := total_biking_hours - workday_hours
  weekend_hours * speed = 200 := by
sorry


end tims_weekend_ride_distance_l491_49134


namespace prob_exactly_four_questions_value_l491_49189

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


end prob_exactly_four_questions_value_l491_49189


namespace age_ratio_proof_l491_49155

def sachin_age : ℕ := 28
def age_difference : ℕ := 8

def rahul_age : ℕ := sachin_age + age_difference

theorem age_ratio_proof : 
  (sachin_age : ℚ) / (rahul_age : ℚ) = 7 / 9 := by sorry

end age_ratio_proof_l491_49155


namespace magazine_revenue_calculation_l491_49128

/-- Calculates the revenue from magazine sales given the total sales, newspaper sales, prices, and total revenue -/
theorem magazine_revenue_calculation 
  (total_items : ℕ) 
  (newspaper_count : ℕ) 
  (newspaper_price : ℚ) 
  (magazine_price : ℚ) 
  (total_revenue : ℚ) 
  (h1 : total_items = 425)
  (h2 : newspaper_count = 275)
  (h3 : newspaper_price = 5/2)
  (h4 : magazine_price = 19/4)
  (h5 : total_revenue = 123025/100)
  (h6 : newspaper_count ≤ total_items) :
  (total_items - newspaper_count) * magazine_price = 54275/100 := by
  sorry

end magazine_revenue_calculation_l491_49128


namespace button_problem_l491_49147

/-- Proof of the button problem -/
theorem button_problem (green : ℕ) (yellow : ℕ) (blue : ℕ) (total : ℕ) : 
  green = 90 →
  yellow = green + 10 →
  total = 275 →
  total = green + yellow + blue →
  green - blue = 5 := by sorry

end button_problem_l491_49147


namespace area_of_corner_squares_l491_49111

/-- The total area of four smaller squares inscribed in the corners of a 2x2 square with an inscribed circle -/
theorem area_of_corner_squares (s : ℝ) : 
  s > 0 ∧ 
  s^2 - 4*s + 2 = 0 ∧ 
  (∃ (r : ℝ), r = 1 ∧ r^2 + r^2 = s^2) →
  4 * s^2 = (48 - 32 * Real.sqrt 2) / 18 :=
by sorry

end area_of_corner_squares_l491_49111


namespace james_to_remaining_ratio_l491_49135

def total_slices : ℕ := 8
def friend_eats : ℕ := 2
def james_eats : ℕ := 3

def slices_after_friend : ℕ := total_slices - friend_eats

theorem james_to_remaining_ratio :
  (james_eats : ℚ) / slices_after_friend = 1 / 2 := by sorry

end james_to_remaining_ratio_l491_49135


namespace yard_area_l491_49102

/-- The area of a rectangular yard with a square cut-out --/
theorem yard_area (length width cut_side : ℕ) 
  (h1 : length = 20) 
  (h2 : width = 16) 
  (h3 : cut_side = 4) : 
  length * width - cut_side * cut_side = 304 := by
  sorry

end yard_area_l491_49102


namespace sqrt_factor_inside_l491_49180

theorem sqrt_factor_inside (x : ℝ) (h : x > 0) :
  -2 * Real.sqrt (5/2) = -Real.sqrt 10 :=
by sorry

end sqrt_factor_inside_l491_49180


namespace f_domain_is_open_interval_l491_49167

/-- The domain of the function f(x) = ln((3 - x)(x + 1)) -/
def f_domain : Set ℝ :=
  {x : ℝ | (3 - x) * (x + 1) > 0}

/-- Theorem stating that the domain of f(x) = ln((3 - x)(x + 1)) is (-1, 3) -/
theorem f_domain_is_open_interval :
  f_domain = Set.Ioo (-1) 3 :=
by
  sorry

#check f_domain_is_open_interval

end f_domain_is_open_interval_l491_49167


namespace lottery_theorem_l491_49159

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

end lottery_theorem_l491_49159
