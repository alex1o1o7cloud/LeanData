import Mathlib

namespace geometric_series_sum_l3827_382748

theorem geometric_series_sum : 
  let a := 2  -- first term
  let r := 3  -- common ratio
  let n := 7  -- number of terms
  a * (r^n - 1) / (r - 1) = 2186 := by
sorry

end geometric_series_sum_l3827_382748


namespace new_combined_total_capacity_l3827_382703

/-- Represents a weightlifter's lifting capacities -/
structure Lifter where
  cleanAndJerk : ℝ
  snatch : ℝ

/-- Represents the improvement rates for a lifter -/
structure Improvement where
  cleanAndJerkRate : ℝ
  snatchRate : ℝ

/-- Calculates the new lifting capacities after improvement -/
def improve (lifter : Lifter) (imp : Improvement) : Lifter where
  cleanAndJerk := lifter.cleanAndJerk * (1 + imp.cleanAndJerkRate)
  snatch := lifter.snatch * (1 + imp.snatchRate)

/-- Calculates the total lifting capacity of a lifter -/
def totalCapacity (lifter : Lifter) : ℝ :=
  lifter.cleanAndJerk + lifter.snatch

/-- The main theorem to prove -/
theorem new_combined_total_capacity
  (john : Lifter)
  (alice : Lifter)
  (mark : Lifter)
  (johnImp : Improvement)
  (aliceImp : Improvement)
  (markImp : Improvement)
  (h1 : john.cleanAndJerk = 80)
  (h2 : john.snatch = 50)
  (h3 : alice.cleanAndJerk = 90)
  (h4 : alice.snatch = 55)
  (h5 : mark.cleanAndJerk = 100)
  (h6 : mark.snatch = 65)
  (h7 : johnImp.cleanAndJerkRate = 1)  -- doubled means 100% increase
  (h8 : johnImp.snatchRate = 0.8)
  (h9 : aliceImp.cleanAndJerkRate = 0.5)
  (h10 : aliceImp.snatchRate = 0.9)
  (h11 : markImp.cleanAndJerkRate = 0.75)
  (h12 : markImp.snatchRate = 0.7)
  : totalCapacity (improve john johnImp) +
    totalCapacity (improve alice aliceImp) +
    totalCapacity (improve mark markImp) = 775 := by
  sorry

end new_combined_total_capacity_l3827_382703


namespace circle_tangent_to_parabola_directrix_l3827_382702

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the focus of the parabola
def focus : ℝ × ℝ := (1, 0)

-- Define the directrix of the parabola
def directrix (x : ℝ) : Prop := x = -1

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 4

-- Theorem statement
theorem circle_tangent_to_parabola_directrix :
  ∀ x y : ℝ,
  parabola x y →
  (∃ t : ℝ, directrix t ∧ 
    ((x - focus.1)^2 + (y - focus.2)^2 = (t - focus.1)^2)) →
  circle_equation x y :=
sorry

end circle_tangent_to_parabola_directrix_l3827_382702


namespace rectangle_length_l3827_382790

/-- Given a rectangle with width 6 inches and area 48 square inches, prove its length is 8 inches -/
theorem rectangle_length (width : ℝ) (area : ℝ) (h1 : width = 6) (h2 : area = 48) :
  area / width = 8 := by sorry

end rectangle_length_l3827_382790


namespace hcf_problem_l3827_382787

theorem hcf_problem (a b : ℕ+) (h1 : max a b = 414) 
  (h2 : Nat.lcm a b = Nat.gcd a b * 13 * 18) : Nat.gcd a b = 23 := by
  sorry

end hcf_problem_l3827_382787


namespace sqrt_inequality_l3827_382773

theorem sqrt_inequality (x : ℝ) : 
  3 * x - 2 ≥ 0 → (|Real.sqrt (3 * x - 2) - 3| > 1 ↔ x > 6 ∨ (2/3 ≤ x ∧ x < 2)) := by
  sorry

end sqrt_inequality_l3827_382773


namespace only_zero_solution_l3827_382749

theorem only_zero_solution (m n : ℤ) (h : 231 * m^2 = 130 * n^2) : m = 0 ∧ n = 0 := by
  sorry

end only_zero_solution_l3827_382749


namespace imaginary_part_of_reciprocal_l3827_382700

theorem imaginary_part_of_reciprocal (i : ℂ) (h : i^2 = -1) :
  Complex.im (1 / (i - 2)) = -1/5 := by
  sorry

end imaginary_part_of_reciprocal_l3827_382700


namespace baseball_team_size_l3827_382771

/-- Calculates the number of players on a team given the total points, 
    points scored by one player, and points scored by each other player -/
def team_size (total_points : ℕ) (one_player_points : ℕ) (other_player_points : ℕ) : ℕ :=
  (total_points - one_player_points) / other_player_points + 1

/-- Theorem stating that for the given conditions, the team size is 6 -/
theorem baseball_team_size : 
  team_size 68 28 8 = 6 := by
  sorry

end baseball_team_size_l3827_382771


namespace det_specific_matrix_l3827_382753

theorem det_specific_matrix (x : ℝ) : 
  let A : Matrix (Fin 3) (Fin 3) ℝ := !![x + 2, x + 1, x; x, x + 2, x + 1; x + 1, x, x + 2]
  Matrix.det A = x^2 + 11*x + 9 := by
sorry

end det_specific_matrix_l3827_382753


namespace triangle_area_l3827_382746

/-- The area of a triangle with vertices at (2,-3), (-4,2), and (3,-7) is 19/2 -/
theorem triangle_area : 
  let A : ℝ × ℝ := (2, -3)
  let B : ℝ × ℝ := (-4, 2)
  let C : ℝ × ℝ := (3, -7)
  let area := abs ((C.1 - A.1) * (B.2 - A.2) - (C.2 - A.2) * (B.1 - A.1)) / 2
  area = 19 / 2 := by
  sorry

end triangle_area_l3827_382746


namespace geometric_sequence_ratio_l3827_382768

/-- 
Given a geometric sequence {a_n} with common ratio q,
prove that if a₂ = 1 and a₁ + a₃ = -2, then q = -1.
-/
theorem geometric_sequence_ratio (a : ℕ → ℝ) (q : ℝ) :
  (∀ n, a (n + 1) = a n * q) →  -- geometric sequence condition
  a 2 = 1 →                    -- a₂ = 1
  a 1 + a 3 = -2 →             -- a₁ + a₃ = -2
  q = -1 := by
sorry

end geometric_sequence_ratio_l3827_382768


namespace train_crossing_time_l3827_382736

/-- Proves that a train with given length and speed takes the calculated time to cross a post -/
theorem train_crossing_time (train_length : Real) (train_speed_kmh : Real) : 
  train_length = 150 →
  train_speed_kmh = 27 →
  (train_length / (train_speed_kmh * 1000 / 3600)) = 20 := by
  sorry

end train_crossing_time_l3827_382736


namespace calculation_proof_system_of_equations_proof_l3827_382779

-- Problem 1
theorem calculation_proof : (-1)^2023 + Real.sqrt 9 - Real.pi^0 + Real.sqrt (1/8) * Real.sqrt 32 = 3 := by
  sorry

-- Problem 2
theorem system_of_equations_proof :
  ∃ (x y : ℝ), 2*x - y = 5 ∧ 3*x + 2*y = 11 ∧ x = 3 ∧ y = 1 := by
  sorry

end calculation_proof_system_of_equations_proof_l3827_382779


namespace smallest_divisible_by_one_to_ten_l3827_382794

theorem smallest_divisible_by_one_to_ten : ∃ n : ℕ,
  (∀ k : ℕ, 1 ≤ k ∧ k ≤ 10 → k ∣ n) ∧
  (∀ m : ℕ, m < n → ∃ j : ℕ, 1 ≤ j ∧ j ≤ 10 ∧ ¬(j ∣ m)) ∧
  n = 2520 :=
by sorry

end smallest_divisible_by_one_to_ten_l3827_382794


namespace min_reciprocal_sum_l3827_382785

theorem min_reciprocal_sum (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (hsum : x + y = 15) :
  ∃ (min : ℝ), min = 4/15 ∧ ∀ (a b : ℝ), 0 < a → 0 < b → a + b = 15 → min ≤ 1/a + 1/b :=
sorry

end min_reciprocal_sum_l3827_382785


namespace mean_of_five_integers_l3827_382731

theorem mean_of_five_integers (p q r s t : ℤ) 
  (h1 : (p + q + r) / 3 = 9)
  (h2 : (s + t) / 2 = 14) :
  (p + q + r + s + t) / 5 = 11 := by
  sorry

end mean_of_five_integers_l3827_382731


namespace shoes_to_sandals_ratio_l3827_382791

def shoes_sold : ℕ := 72
def sandals_sold : ℕ := 40

theorem shoes_to_sandals_ratio :
  (shoes_sold / sandals_sold : ℚ) = 9 / 5 := by
  sorry

end shoes_to_sandals_ratio_l3827_382791


namespace amusement_park_admission_l3827_382742

theorem amusement_park_admission (child_fee : ℚ) (adult_fee : ℚ) (total_fee : ℚ) (num_children : ℕ) :
  child_fee = 3/2 →
  adult_fee = 4 →
  total_fee = 810 →
  num_children = 180 →
  ∃ (num_adults : ℕ), 
    (child_fee * num_children + adult_fee * num_adults = total_fee) ∧
    (num_children + num_adults = 315) :=
by
  sorry

end amusement_park_admission_l3827_382742


namespace olivias_house_height_l3827_382710

/-- The height of Olivia's house in feet -/
def house_height : ℕ := 81

/-- The length of the shadow cast by Olivia's house in feet -/
def house_shadow : ℕ := 70

/-- The height of the flagpole in feet -/
def flagpole_height : ℕ := 35

/-- The length of the shadow cast by the flagpole in feet -/
def flagpole_shadow : ℕ := 30

/-- The height of the bush in feet -/
def bush_height : ℕ := 14

/-- The length of the shadow cast by the bush in feet -/
def bush_shadow : ℕ := 12

theorem olivias_house_height :
  (house_height : ℚ) / house_shadow = flagpole_height / flagpole_shadow ∧
  (house_height : ℚ) / house_shadow = bush_height / bush_shadow ∧
  house_height = 81 :=
sorry

end olivias_house_height_l3827_382710


namespace min_abs_value_plus_constant_l3827_382745

theorem min_abs_value_plus_constant (x : ℝ) :
  ∀ y : ℝ, |x - 2| + 2023 ≤ |y - 2| + 2023 ↔ x = 2 := by
  sorry

end min_abs_value_plus_constant_l3827_382745


namespace unique_solution_exponential_equation_l3827_382777

theorem unique_solution_exponential_equation :
  ∃! (x y z t : ℕ+), 12^(x:ℕ) + 13^(y:ℕ) - 14^(z:ℕ) = 2013^(t:ℕ) ∧ 
    x = 1 ∧ y = 3 ∧ z = 2 ∧ t = 1 := by
  sorry

end unique_solution_exponential_equation_l3827_382777


namespace fraction_product_cubed_simplify_fraction_cube_l3827_382741

theorem fraction_product_cubed (a b c d : ℚ) :
  (a / b) ^ 3 * (c / d) ^ 3 = ((a * c) / (b * d)) ^ 3 :=
by sorry

theorem simplify_fraction_cube :
  (5 / 8) ^ 3 * (2 / 3) ^ 3 = 125 / 1728 :=
by sorry

end fraction_product_cubed_simplify_fraction_cube_l3827_382741


namespace joan_missed_games_l3827_382783

/-- The number of baseball games Joan missed -/
def games_missed (total_games attended_games : ℕ) : ℕ :=
  total_games - attended_games

/-- Theorem stating that Joan missed 469 games -/
theorem joan_missed_games :
  let total_games : ℕ := 864
  let attended_games : ℕ := 395
  games_missed total_games attended_games = 469 := by
  sorry

end joan_missed_games_l3827_382783


namespace total_cost_of_items_l3827_382732

theorem total_cost_of_items (wallet_cost : ℕ) 
  (h1 : wallet_cost = 22)
  (purse_cost : ℕ) 
  (h2 : purse_cost = 4 * wallet_cost - 3)
  (shoes_cost : ℕ) 
  (h3 : shoes_cost = wallet_cost + purse_cost + 7) :
  wallet_cost + purse_cost + shoes_cost = 221 := by
sorry

end total_cost_of_items_l3827_382732


namespace overall_average_score_problem_solution_l3827_382788

/-- Calculates the overall average score of two classes -/
theorem overall_average_score 
  (n1 : ℕ) (n2 : ℕ) (avg1 : ℝ) (avg2 : ℝ) : 
  (n1 : ℝ) * avg1 + (n2 : ℝ) * avg2 = ((n1 + n2) : ℝ) * ((n1 * avg1 + n2 * avg2) / (n1 + n2)) :=
by sorry

/-- Proves that the overall average score for the given problem is 74 -/
theorem problem_solution 
  (n1 : ℕ) (n2 : ℕ) (avg1 : ℝ) (avg2 : ℝ) 
  (h1 : n1 = 20) (h2 : n2 = 30) (h3 : avg1 = 80) (h4 : avg2 = 70) :
  (n1 * avg1 + n2 * avg2) / (n1 + n2) = 74 :=
by sorry

end overall_average_score_problem_solution_l3827_382788


namespace M_equals_N_l3827_382747

/-- The set M of integers of the form 12m + 8n + 4l where m, n, l are integers -/
def M : Set ℤ := {u | ∃ m n l : ℤ, u = 12*m + 8*n + 4*l}

/-- The set N of integers of the form 20p + 16q + 12r where p, q, r are integers -/
def N : Set ℤ := {u | ∃ p q r : ℤ, u = 20*p + 16*q + 12*r}

/-- Theorem stating that M equals N -/
theorem M_equals_N : M = N := by
  sorry

end M_equals_N_l3827_382747


namespace g_2023_of_2_eq_2_l3827_382769

def g (x : ℚ) : ℚ := (2 - x) / (2 * x + 1)

def g_n : ℕ → ℚ → ℚ
  | 0, x => x
  | 1, x => g x
  | (n + 2), x => g (g_n (n + 1) x)

theorem g_2023_of_2_eq_2 : g_n 2023 2 = 2 := by sorry

end g_2023_of_2_eq_2_l3827_382769


namespace histogram_area_sum_is_one_l3827_382789

/-- Represents a histogram of sample frequency distribution -/
structure Histogram where
  rectangles : List ℝ
  -- Each element in the list represents the area of a small rectangle

/-- The sum of areas of all rectangles in a histogram equals 1 -/
theorem histogram_area_sum_is_one (h : Histogram) : 
  h.rectangles.sum = 1 := by
  sorry

end histogram_area_sum_is_one_l3827_382789


namespace restaurant_bill_calculation_l3827_382759

theorem restaurant_bill_calculation (total_people adults kids : ℕ) (adult_meal_cost : ℚ) :
  total_people = adults + kids →
  total_people = 12 →
  kids = 7 →
  adult_meal_cost = 3 →
  adults * adult_meal_cost = 15 :=
by
  sorry

end restaurant_bill_calculation_l3827_382759


namespace cycle_not_divisible_by_three_l3827_382733

/-- A graph is a type with an edge relation -/
class Graph (V : Type) :=
  (adj : V → V → Prop)

/-- The degree of a vertex in a graph is the number of adjacent vertices -/
def degree {V : Type} [Graph V] (v : V) : ℕ := sorry

/-- A path in a graph is a list of vertices where each consecutive pair is adjacent -/
def is_path {V : Type} [Graph V] (p : List V) : Prop := sorry

/-- A cycle in a graph is a path where the first and last vertices are the same -/
def is_cycle {V : Type} [Graph V] (c : List V) : Prop := sorry

/-- The length of a path or cycle is the number of edges it contains -/
def length {V : Type} [Graph V] (p : List V) : ℕ := sorry

theorem cycle_not_divisible_by_three 
  {V : Type} [Graph V] 
  (h : ∀ v : V, degree v ≥ 3) : 
  ∃ c : List V, is_cycle c ∧ ¬(length c % 3 = 0) := by sorry

end cycle_not_divisible_by_three_l3827_382733


namespace pet_ownership_percentages_l3827_382754

theorem pet_ownership_percentages (total_students : ℕ) (cat_owners : ℕ) (dog_owners : ℕ)
  (h1 : total_students = 500)
  (h2 : cat_owners = 75)
  (h3 : dog_owners = 125) :
  (cat_owners : ℚ) / total_students * 100 = 15 ∧
  (dog_owners : ℚ) / total_students * 100 = 25 := by
sorry

end pet_ownership_percentages_l3827_382754


namespace kendra_shirts_theorem_l3827_382705

/-- Represents the number of shirts Kendra needs for two weeks -/
def shirts_needed : ℕ :=
  let school_days := 5
  let club_days := 3
  let saturday_shirts := 1
  let sunday_shirts := 2
  let weeks := 2
  (school_days + club_days + saturday_shirts + sunday_shirts) * weeks

/-- Theorem stating that Kendra needs 22 shirts to do laundry once every two weeks -/
theorem kendra_shirts_theorem : shirts_needed = 22 := by
  sorry

end kendra_shirts_theorem_l3827_382705


namespace construction_company_higher_utility_l3827_382711

/-- Represents the quality of renovation work -/
structure Quality where
  value : ℝ
  nonneg : value ≥ 0

/-- Represents the cost of renovation work -/
structure Cost where
  value : ℝ
  nonneg : value ≥ 0

/-- Represents the amount of available information about the service provider -/
structure Information where
  value : ℝ
  nonneg : value ≥ 0

/-- Represents a renovation service provider -/
structure ServiceProvider where
  quality : Quality
  cost : Cost
  information : Information

/-- Utility function for renovation service -/
def utilityFunction (α β γ : ℝ) (sp : ServiceProvider) : ℝ :=
  α * sp.quality.value + β * sp.information.value - γ * sp.cost.value

/-- Theorem: Under certain conditions, a construction company can provide higher expected utility -/
theorem construction_company_higher_utility 
  (cc : ServiceProvider) -- construction company
  (prc : ServiceProvider) -- private repair crew
  (α β γ : ℝ) -- utility function parameters
  (h_α : α > 0) -- quality is valued positively
  (h_β : β > 0) -- information is valued positively
  (h_γ : γ > 0) -- cost is valued negatively
  (h_quality : cc.quality.value > prc.quality.value) -- company provides higher quality
  (h_info : cc.information.value > prc.information.value) -- company provides more information
  (h_cost : cc.cost.value > prc.cost.value) -- company is more expensive
  : ∃ (α β γ : ℝ), utilityFunction α β γ cc > utilityFunction α β γ prc :=
sorry

end construction_company_higher_utility_l3827_382711


namespace total_age_reaches_target_in_10_years_l3827_382796

/-- Represents the number of years between each sibling's birth -/
def age_gap : ℕ := 5

/-- Represents the current age of the eldest sibling -/
def eldest_current_age : ℕ := 20

/-- Represents the target total age of all siblings -/
def target_total_age : ℕ := 75

/-- Calculates the total age of the siblings after a given number of years -/
def total_age_after (years : ℕ) : ℕ :=
  (eldest_current_age + years) + 
  (eldest_current_age - age_gap + years) + 
  (eldest_current_age - 2 * age_gap + years)

/-- Theorem stating that it takes 10 years for the total age to reach the target -/
theorem total_age_reaches_target_in_10_years : 
  total_age_after 10 = target_total_age :=
sorry

end total_age_reaches_target_in_10_years_l3827_382796


namespace consecutive_integers_average_l3827_382786

theorem consecutive_integers_average (n m : ℤ) : 
  (n > 0) →
  (m = (n + (n+1) + (n+2) + (n+3) + (n+4) + (n+5) + (n+6)) / 7) →
  ((m + (m+1) + (m+2) + (m+3) + (m+4) + (m+5) + (m+6)) / 7 = n + 6) :=
by sorry

end consecutive_integers_average_l3827_382786


namespace conditional_prob_B_given_A_l3827_382755

/-- The number of class officers -/
def total_officers : ℕ := 6

/-- The number of boys among the class officers -/
def num_boys : ℕ := 4

/-- The number of girls among the class officers -/
def num_girls : ℕ := 2

/-- The number of students selected -/
def num_selected : ℕ := 3

/-- Event A: "boy A being selected" -/
def event_A : Set (Fin total_officers) := sorry

/-- Event B: "girl B being selected" -/
def event_B : Set (Fin total_officers) := sorry

/-- The probability of event A -/
def prob_A : ℚ := 1 / 2

/-- The probability of both events A and B occurring -/
def prob_AB : ℚ := 1 / 5

/-- Theorem: The conditional probability P(B|A) is 2/5 -/
theorem conditional_prob_B_given_A : 
  (prob_AB / prob_A : ℚ) = 2 / 5 := by sorry

end conditional_prob_B_given_A_l3827_382755


namespace power_negative_one_equals_half_l3827_382758

-- Define the theorem
theorem power_negative_one_equals_half : 2^(-1 : ℤ) = (1/2 : ℚ) := by
  sorry

end power_negative_one_equals_half_l3827_382758


namespace functional_equation_solution_l3827_382760

/-- A continuous function satisfying the given functional equation is either constantly 0 or 1/2. -/
theorem functional_equation_solution (f : ℝ → ℝ) (hf : Continuous f)
  (h : ∀ x y : ℝ, f (x^2 - y^2) = f x^2 + f y^2) :
  (∀ x : ℝ, f x = 0) ∨ (∀ x : ℝ, f x = 1/2) := by
  sorry

end functional_equation_solution_l3827_382760


namespace cucumber_weight_after_evaporation_l3827_382792

theorem cucumber_weight_after_evaporation 
  (initial_weight : ℝ) 
  (initial_water_percentage : ℝ) 
  (final_water_percentage : ℝ) :
  initial_weight = 100 →
  initial_water_percentage = 0.99 →
  final_water_percentage = 0.95 →
  ∃ (final_weight : ℝ), 
    final_weight * (1 - final_water_percentage) = initial_weight * (1 - initial_water_percentage) ∧
    final_weight = 20 :=
by sorry

end cucumber_weight_after_evaporation_l3827_382792


namespace curve_self_intersection_l3827_382795

/-- The x-coordinate of a point on the curve as a function of t -/
def x (t : ℝ) : ℝ := t^3 - 3*t + 1

/-- The y-coordinate of a point on the curve as a function of t -/
def y (t : ℝ) : ℝ := t^4 - 4*t^2 + 4

/-- The curve crosses itself at (1, 1) -/
theorem curve_self_intersection :
  ∃ (a b : ℝ), a ≠ b ∧ x a = x b ∧ y a = y b ∧ x a = 1 ∧ y a = 1 :=
sorry

end curve_self_intersection_l3827_382795


namespace min_value_of_some_expression_l3827_382714

/-- The minimum value of |some expression| given the conditions -/
theorem min_value_of_some_expression :
  ∃ (f : ℝ → ℝ),
    (∀ x, |x - 4| + |x + 7| + |f x| ≥ 12) ∧
    (∃ x₀, |x₀ - 4| + |x₀ + 7| + |f x₀| = 12) →
    ∃ x₁, |f x₁| = 1 ∧ ∀ x, |f x| ≥ 1 :=
by sorry


end min_value_of_some_expression_l3827_382714


namespace log_inequality_l3827_382709

theorem log_inequality (h1 : 5^5 < 8^4) (h2 : 13^4 < 8^5) :
  Real.log 3 / Real.log 5 < Real.log 5 / Real.log 8 ∧
  Real.log 5 / Real.log 8 < Real.log 8 / Real.log 13 := by
sorry

end log_inequality_l3827_382709


namespace sherry_opposite_vertex_probability_l3827_382784

/-- The probability that Sherry is at the opposite vertex after k minutes on a triangle -/
def P (k : ℕ) : ℚ :=
  1/6 + 1/(3 * (-2)^k)

/-- Theorem: For k > 0, the probability that Sherry is at the opposite vertex after k minutes on a triangle is P(k) -/
theorem sherry_opposite_vertex_probability (k : ℕ) (h : k > 0) : 
  (1/6 : ℚ) + 1/(3 * (-2)^k) = P k := by
sorry

end sherry_opposite_vertex_probability_l3827_382784


namespace hair_cut_second_day_l3827_382780

/-- The amount of hair cut off on the second day, given the total amount cut off and the amount cut off on the first day. -/
theorem hair_cut_second_day 
  (total_cut : ℝ) 
  (first_day_cut : ℝ) 
  (h1 : total_cut = 0.875) 
  (h2 : first_day_cut = 0.375) : 
  total_cut - first_day_cut = 0.500 := by
sorry

end hair_cut_second_day_l3827_382780


namespace johns_annual_epipen_cost_l3827_382743

/-- Represents the cost of EpiPens for John over a year -/
def annual_epipen_cost (epipen_cost : ℝ) (insurance_coverage : ℝ) (replacements_per_year : ℕ) : ℝ :=
  replacements_per_year * (1 - insurance_coverage) * epipen_cost

/-- Theorem stating that John's annual cost for EpiPens is $250 -/
theorem johns_annual_epipen_cost :
  annual_epipen_cost 500 0.75 2 = 250 := by
  sorry

end johns_annual_epipen_cost_l3827_382743


namespace original_station_count_l3827_382799

/-- The number of combinations of 2 items from a set of k items -/
def combinations (k : ℕ) : ℕ := k * (k - 1) / 2

/-- 
Given:
- m is the original number of stations
- n is the number of new stations added (n > 1)
- The increase in types of passenger tickets is 58

Prove that m = 14
-/
theorem original_station_count (m n : ℕ) 
  (h1 : n > 1) 
  (h2 : combinations (m + n) - combinations m = 58) : 
  m = 14 := by sorry

end original_station_count_l3827_382799


namespace g_13_equals_205_l3827_382734

def g (n : ℕ) : ℕ := n^2 + n + 23

theorem g_13_equals_205 : g 13 = 205 := by
  sorry

end g_13_equals_205_l3827_382734


namespace equation_holds_l3827_382750

theorem equation_holds : Real.sqrt (2 + Real.sqrt (3 + Real.sqrt 0)) = (2 + Real.sqrt 0) ^ (1/4) := by
  sorry

end equation_holds_l3827_382750


namespace number_problem_l3827_382766

theorem number_problem :
  ∃ x : ℝ, x = (1/4) * x + 93.33333333333333 ∧ x = 124.44444444444444 := by
  sorry

end number_problem_l3827_382766


namespace sheets_per_pack_calculation_l3827_382737

/-- Represents the number of sheets in a pack of notebook paper -/
def sheets_per_pack : ℕ := 100

/-- Represents the number of pages Chip takes per day per class -/
def pages_per_day_per_class : ℕ := 2

/-- Represents the number of days Chip takes notes per week -/
def days_per_week : ℕ := 5

/-- Represents the number of classes Chip has -/
def num_classes : ℕ := 5

/-- Represents the number of weeks Chip has been taking notes -/
def num_weeks : ℕ := 6

/-- Represents the number of packs Chip used -/
def packs_used : ℕ := 3

theorem sheets_per_pack_calculation :
  sheets_per_pack = 
    (pages_per_day_per_class * days_per_week * num_classes * num_weeks) / packs_used :=
by sorry

end sheets_per_pack_calculation_l3827_382737


namespace quilt_material_requirement_l3827_382770

/-- Given that 7 quilts can be made with 21 yards of material,
    prove that 12 quilts require 36 yards of material. -/
theorem quilt_material_requirement : 
  (7 : ℚ) * (36 : ℚ) = (12 : ℚ) * (21 : ℚ) := by
  sorry

end quilt_material_requirement_l3827_382770


namespace student_average_greater_than_true_average_l3827_382775

theorem student_average_greater_than_true_average (x y z : ℝ) (h : x < y ∧ y < z) :
  (x + y) / 2 + z > (x + y + z) / 3 := by
  sorry

end student_average_greater_than_true_average_l3827_382775


namespace intersection_point_k_l3827_382735

-- Define the three lines
def line1 (x y : ℚ) : Prop := y = 4 * x - 1
def line2 (x y : ℚ) : Prop := y = -1/3 * x + 11
def line3 (x y k : ℚ) : Prop := y = 2 * x + k

-- Define the condition that all three lines intersect at the same point
def lines_intersect (k : ℚ) : Prop :=
  ∃ x y : ℚ, line1 x y ∧ line2 x y ∧ line3 x y k

-- Theorem statement
theorem intersection_point_k :
  ∃! k : ℚ, lines_intersect k ∧ k = 59/13 := by sorry

end intersection_point_k_l3827_382735


namespace melanie_balloons_l3827_382726

def joan_balloons : ℕ := 40
def total_balloons : ℕ := 81

theorem melanie_balloons : total_balloons - joan_balloons = 41 := by
  sorry

end melanie_balloons_l3827_382726


namespace cylindrical_to_rectangular_conversion_l3827_382763

theorem cylindrical_to_rectangular_conversion :
  let r : ℝ := 5
  let θ : ℝ := π / 3
  let z : ℝ := 2
  let x : ℝ := r * Real.cos θ
  let y : ℝ := r * Real.sin θ
  (x, y, z) = (2.5, 5 * Real.sqrt 3 / 2, 2) :=
by sorry

end cylindrical_to_rectangular_conversion_l3827_382763


namespace range_of_b_l3827_382774

theorem range_of_b (b : ℝ) : 
  Real.sqrt ((b - 2)^2) = 2 - b ↔ b ∈ Set.Iic 2 :=
sorry

end range_of_b_l3827_382774


namespace smallest_divisible_by_1984_l3827_382797

theorem smallest_divisible_by_1984 :
  ∃ (a : ℕ), (a > 0) ∧
  (∀ (n : ℕ), Odd n → (47^n + a * 15^n) % 1984 = 0) ∧
  (∀ (b : ℕ), 0 < b ∧ b < a → ∃ (m : ℕ), Odd m ∧ (47^m + b * 15^m) % 1984 ≠ 0) ∧
  (a = 1055) := by
sorry

end smallest_divisible_by_1984_l3827_382797


namespace trailing_zeros_15_factorial_base_15_l3827_382730

/-- The number of trailing zeros in n! in base b --/
def trailingZeros (n : ℕ) (b : ℕ) : ℕ := sorry

/-- The factorial of a natural number --/
def factorial (n : ℕ) : ℕ := sorry

/-- 15 factorial --/
def factorial15 : ℕ := factorial 15

theorem trailing_zeros_15_factorial_base_15 :
  trailingZeros factorial15 15 = 3 := by sorry

end trailing_zeros_15_factorial_base_15_l3827_382730


namespace area_of_triangle_MOI_l3827_382751

/-- Triangle ABC with given side lengths --/
structure Triangle :=
  (A B C : ℝ × ℝ)
  (ab_length : dist A B = 15)
  (ac_length : dist A C = 8)
  (bc_length : dist B C = 7)

/-- Circumcenter of a triangle --/
def circumcenter (t : Triangle) : ℝ × ℝ := sorry

/-- Incenter of a triangle --/
def incenter (t : Triangle) : ℝ × ℝ := sorry

/-- Point M: center of circle tangent to sides AC, BC, and circumcircle --/
def point_M (t : Triangle) : ℝ × ℝ := sorry

/-- Area of a triangle given three points --/
def triangle_area (p1 p2 p3 : ℝ × ℝ) : ℝ := sorry

/-- Main theorem --/
theorem area_of_triangle_MOI (t : Triangle) :
  triangle_area (circumcenter t) (incenter t) (point_M t) = 7/4 := by sorry

end area_of_triangle_MOI_l3827_382751


namespace count_auspicious_dragon_cards_l3827_382719

/-- The number of ways to select 4 digits from 0 to 9 and arrange them in ascending order -/
def auspicious_dragon_cards : ℕ := sorry

/-- Theorem stating that the number of Auspicious Dragon Cards is 210 -/
theorem count_auspicious_dragon_cards : auspicious_dragon_cards = 210 := by sorry

end count_auspicious_dragon_cards_l3827_382719


namespace multiply_and_simplify_l3827_382757

theorem multiply_and_simplify (x y : ℝ) :
  (3 * x^2 - 4 * y^3) * (9 * x^4 + 12 * x^2 * y^3 + 16 * y^6) = 27 * x^6 - 64 * y^9 := by
  sorry

end multiply_and_simplify_l3827_382757


namespace vector_dot_product_equation_l3827_382740

/-- Given vectors a, b, and c satisfying certain conditions, prove that x = 4 -/
theorem vector_dot_product_equation (a b c : ℝ × ℝ) (x : ℝ) 
  (ha : a = (1, 1))
  (hb : b = (2, 5))
  (hc : c = (3, x))
  (h_dot : ((8 • a - b) • c) = 30) :
  x = 4 := by
sorry

end vector_dot_product_equation_l3827_382740


namespace six_points_fifteen_segments_l3827_382782

/-- The number of line segments formed by connecting n distinct points on a circle --/
def lineSegments (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem: For 6 distinct points on a circle, the number of line segments is 15 --/
theorem six_points_fifteen_segments : lineSegments 6 = 15 := by
  sorry

end six_points_fifteen_segments_l3827_382782


namespace compound_statement_false_l3827_382713

theorem compound_statement_false (p q : Prop) :
  ¬(p ∧ q) → (¬p ∨ ¬q) := by
  sorry

end compound_statement_false_l3827_382713


namespace sample_size_calculation_l3827_382762

theorem sample_size_calculation (num_classes : ℕ) (papers_per_class : ℕ) : 
  num_classes = 8 → papers_per_class = 12 → num_classes * papers_per_class = 96 := by
  sorry

end sample_size_calculation_l3827_382762


namespace sin_equation_solutions_l3827_382721

/-- The number of solutions to 2sin³x - 5sin²x + 2sinx = 0 in [0, 2π] is 5 -/
theorem sin_equation_solutions : 
  let f : ℝ → ℝ := λ x => 2 * Real.sin x ^ 3 - 5 * Real.sin x ^ 2 + 2 * Real.sin x
  ∃! (s : Finset ℝ), s.card = 5 ∧ 
    (∀ x ∈ s, 0 ≤ x ∧ x ≤ 2 * Real.pi ∧ f x = 0) ∧
    (∀ x, 0 ≤ x ∧ x ≤ 2 * Real.pi ∧ f x = 0 → x ∈ s) :=
by sorry

end sin_equation_solutions_l3827_382721


namespace train_speed_l3827_382798

/-- The speed of a train given specific conditions -/
theorem train_speed (train_length : ℝ) (man_speed : ℝ) (passing_time : ℝ) :
  train_length = 500 →
  man_speed = 12 →
  passing_time = 10 →
  ∃ (train_speed : ℝ), train_speed = 168 ∧ 
    (train_speed + man_speed) * passing_time / 3.6 = train_length + man_speed * passing_time / 3.6 :=
by sorry


end train_speed_l3827_382798


namespace factorization_of_x_power_difference_l3827_382767

theorem factorization_of_x_power_difference (m : ℕ) (x : ℝ) (hm : m > 1) :
  x^m - x^(m-2) = x^(m-2) * (x + 1) * (x - 1) := by
  sorry

end factorization_of_x_power_difference_l3827_382767


namespace pyramid_levels_6_l3827_382706

/-- Defines the number of cubes in a pyramid with n levels -/
def pyramid_cubes (n : ℕ) : ℕ := n * (n + 1) * (2 * n + 1) / 6

/-- Theorem stating that a pyramid with 6 levels contains 225 cubes -/
theorem pyramid_levels_6 : pyramid_cubes 6 = 225 := by sorry

end pyramid_levels_6_l3827_382706


namespace oak_trees_planted_l3827_382701

/-- The number of oak trees planted today in the park -/
def trees_planted (current : ℕ) (final : ℕ) : ℕ := final - current

/-- Theorem stating that the number of oak trees planted today is 4 -/
theorem oak_trees_planted : trees_planted 5 9 = 4 := by
  sorry

end oak_trees_planted_l3827_382701


namespace football_game_attendance_l3827_382708

/-- Proves that the number of children attending a football game is 80, given the ticket prices, total attendance, and total money collected. -/
theorem football_game_attendance
  (adult_price : ℕ) -- Price of adult ticket in cents
  (child_price : ℕ) -- Price of child ticket in cents
  (total_attendance : ℕ) -- Total number of attendees
  (total_revenue : ℕ) -- Total revenue in cents
  (h1 : adult_price = 60)
  (h2 : child_price = 25)
  (h3 : total_attendance = 280)
  (h4 : total_revenue = 14000) :
  ∃ (adults children : ℕ),
    adults + children = total_attendance ∧
    adults * adult_price + children * child_price = total_revenue ∧
    children = 80 :=
by sorry

end football_game_attendance_l3827_382708


namespace b_2017_equals_1_l3827_382761

def fibonacci : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | (n + 2) => fibonacci (n + 1) + fibonacci n

def b (n : ℕ) : ℕ := fibonacci n % 3

theorem b_2017_equals_1 : b 2017 = 1 := by sorry

end b_2017_equals_1_l3827_382761


namespace smallest_abs_z_l3827_382739

theorem smallest_abs_z (z : ℂ) (h : Complex.abs (z - 8) + Complex.abs (z + 6 * Complex.I) = 17) :
  ∃ (w : ℂ), Complex.abs w ≤ Complex.abs z ∧ Complex.abs (w - 8) + Complex.abs (w + 6 * Complex.I) = 17 ∧ Complex.abs w = 48 / 17 :=
sorry

end smallest_abs_z_l3827_382739


namespace largest_common_term_less_than_800_l3827_382725

def arithmetic_progression_1 (n : ℕ) : ℤ := 4 + 5 * n
def arithmetic_progression_2 (m : ℕ) : ℤ := 7 + 8 * m

def is_common_term (a : ℤ) : Prop :=
  ∃ n m : ℕ, arithmetic_progression_1 n = a ∧ arithmetic_progression_2 m = a

theorem largest_common_term_less_than_800 :
  ∃ a : ℤ, is_common_term a ∧ a < 800 ∧ ∀ b : ℤ, is_common_term b ∧ b < 800 → b ≤ a :=
by sorry

end largest_common_term_less_than_800_l3827_382725


namespace cubic_fraction_simplification_l3827_382716

theorem cubic_fraction_simplification 
  (a b x : ℝ) 
  (h1 : x = a^3 / b^3) 
  (h2 : a ≠ b) 
  (h3 : b ≠ 0) : 
  (a^3 + b^3) / (a^3 - b^3) = (x + 1) / (x - 1) :=
by sorry

end cubic_fraction_simplification_l3827_382716


namespace equation_solution_l3827_382729

theorem equation_solution (k : ℝ) : 
  (∃ x : ℝ, x = -5 ∧ (1 : ℝ) / 2023 * x - 2 = 3 * x + k) →
  (∃ y : ℝ, y = -3 ∧ (1 : ℝ) / 2023 * (2 * y + 1) - 5 = 6 * y + k) := by
  sorry

end equation_solution_l3827_382729


namespace sin_negative_nine_half_pi_l3827_382728

theorem sin_negative_nine_half_pi : Real.sin (-9 * Real.pi / 2) = -1 := by
  sorry

end sin_negative_nine_half_pi_l3827_382728


namespace orange_bin_count_l3827_382764

theorem orange_bin_count (initial : ℕ) (removed : ℕ) (added : ℕ) : 
  initial = 40 → removed = 37 → added = 7 → initial - removed + added = 10 := by
  sorry

end orange_bin_count_l3827_382764


namespace modulo_eleven_residue_l3827_382781

theorem modulo_eleven_residue :
  (332 + 6 * 44 + 8 * 176 + 3 * 22) % 11 = 2 := by
  sorry

end modulo_eleven_residue_l3827_382781


namespace union_of_sets_l3827_382718

theorem union_of_sets : 
  let A : Set ℕ := {1, 2, 3}
  let B : Set ℕ := {2, 3, 4}
  A ∪ B = {1, 2, 3, 4} := by
  sorry

end union_of_sets_l3827_382718


namespace min_value_fraction_sum_l3827_382752

theorem min_value_fraction_sum (x y : ℝ) (hx : x > 1) (hy : y > 1) :
  x^2 / (y^2 - 1) + y^2 / (x^2 - 1) ≥ 4 ∧
  (x^2 / (y^2 - 1) + y^2 / (x^2 - 1) = 4 ↔ x = Real.sqrt 2 ∧ y = Real.sqrt 2) :=
by sorry

end min_value_fraction_sum_l3827_382752


namespace min_gumballs_for_four_is_ten_l3827_382765

/-- Represents the number of gumballs of each color in the machine -/
structure GumballMachine where
  red : Nat
  white : Nat
  blue : Nat

/-- The minimum number of gumballs needed to guarantee 4 of the same color -/
def minGumballsForFour (machine : GumballMachine) : Nat :=
  10

/-- Theorem stating that for a machine with 9 red, 7 white, and 8 blue gumballs,
    the minimum number of gumballs needed to guarantee 4 of the same color is 10 -/
theorem min_gumballs_for_four_is_ten (machine : GumballMachine)
    (h_red : machine.red = 9)
    (h_white : machine.white = 7)
    (h_blue : machine.blue = 8) :
    minGumballsForFour machine = 10 := by
  sorry


end min_gumballs_for_four_is_ten_l3827_382765


namespace oak_trees_remaining_l3827_382717

/-- The number of oak trees remaining after cutting down damaged trees -/
def remaining_oak_trees (initial : ℕ) (cut_down : ℕ) : ℕ :=
  initial - cut_down

/-- Theorem stating that the number of oak trees remaining is 7 -/
theorem oak_trees_remaining :
  remaining_oak_trees 9 2 = 7 := by
  sorry

end oak_trees_remaining_l3827_382717


namespace train_distance_theorem_l3827_382722

/-- The distance between two trains traveling in opposite directions -/
def distance_between_trains (speed_a speed_b : ℝ) (time_a time_b : ℝ) : ℝ :=
  speed_a * time_a + speed_b * time_b

/-- Theorem: The distance between the trains is 1284 miles -/
theorem train_distance_theorem :
  distance_between_trains 56 23 18 12 = 1284 := by
  sorry

end train_distance_theorem_l3827_382722


namespace units_digit_of_power_l3827_382738

theorem units_digit_of_power (n : ℕ) : (147 ^ 25) ^ 50 ≡ 9 [ZMOD 10] := by
  sorry

end units_digit_of_power_l3827_382738


namespace geometric_series_first_term_l3827_382704

theorem geometric_series_first_term (a r : ℝ) (h1 : |r| < 1) 
  (h2 : a / (1 - r) = 30) (h3 : a^2 / (1 - r^2) = 90) : a = 60 / 11 := by
  sorry

end geometric_series_first_term_l3827_382704


namespace jellybean_probability_l3827_382778

/-- Probability of picking exactly 2 red jellybeans from a bowl -/
theorem jellybean_probability :
  let total_jellybeans : ℕ := 10
  let red_jellybeans : ℕ := 4
  let blue_jellybeans : ℕ := 1
  let white_jellybeans : ℕ := 5
  let picks : ℕ := 3
  
  -- Ensure the total number of jellybeans is correct
  total_jellybeans = red_jellybeans + blue_jellybeans + white_jellybeans →
  
  -- Calculate the probability
  (Nat.choose red_jellybeans 2 * (blue_jellybeans + white_jellybeans)) / 
  Nat.choose total_jellybeans picks = 3 / 10 :=
by
  sorry

end jellybean_probability_l3827_382778


namespace average_weight_problem_l3827_382793

/-- The average weight problem -/
theorem average_weight_problem 
  (A B C D E : ℝ) 
  (h1 : (A + B + C) / 3 = 84)
  (h2 : (A + B + C + D) / 4 = 80)
  (h3 : E = D + 8)
  (h4 : A = 80) :
  (B + C + D + E) / 4 = 79 := by
  sorry

end average_weight_problem_l3827_382793


namespace common_chord_length_l3827_382723

-- Define the two circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 - 8 = 0
def circle2 (x y : ℝ) : Prop := x^2 + y^2 - 3*x + 4*y - 18 = 0

-- Define the common chord
def common_chord (x y : ℝ) : Prop := 3*x - 4*y + 10 = 0

-- Theorem statement
theorem common_chord_length :
  ∃ (length : ℝ), 
    (∀ (x y : ℝ), circle1 x y ∧ circle2 x y → common_chord x y) ∧
    length = 4 :=
sorry

end common_chord_length_l3827_382723


namespace rectangle_dimensions_l3827_382727

theorem rectangle_dimensions (x : ℝ) : 
  (2*x - 3) * (3*x + 4) = 20*x - 12 → x = 7/2 := by
sorry

end rectangle_dimensions_l3827_382727


namespace angle_triple_complement_l3827_382756

theorem angle_triple_complement (x : ℝ) : x = 3 * (90 - x) → x = 67.5 := by
  sorry

end angle_triple_complement_l3827_382756


namespace distance_maximum_at_halfway_l3827_382720

-- Define a square in 2D space
structure Square :=
  (side : ℝ)
  (center : ℝ × ℝ)

-- Define a runner's position on the square
structure RunnerPosition :=
  (square : Square)
  (t : ℝ)  -- Parameter representing time or position along the path (0 ≤ t ≤ 4)

-- Function to calculate the runner's coordinates
def runnerCoordinates (pos : RunnerPosition) : ℝ × ℝ :=
  sorry

-- Function to calculate the straight-line distance from the starting point
def distanceFromStart (pos : RunnerPosition) : ℝ :=
  sorry

theorem distance_maximum_at_halfway (s : Square) :
  ∃ (t_max : ℝ), t_max = 2 ∧
  ∀ (t : ℝ), 0 ≤ t ∧ t ≤ 4 →
    distanceFromStart ⟨s, t⟩ ≤ distanceFromStart ⟨s, t_max⟩ :=
sorry

end distance_maximum_at_halfway_l3827_382720


namespace robert_reading_capacity_l3827_382715

/-- Represents the number of complete books that can be read given the reading speed, book length, and available time. -/
def booksRead (readingSpeed : ℕ) (bookLength : ℕ) (availableTime : ℕ) : ℕ :=
  (readingSpeed * availableTime) / bookLength

/-- Theorem stating that Robert can read 2 complete 360-page books in 8 hours at a speed of 120 pages per hour. -/
theorem robert_reading_capacity :
  booksRead 120 360 8 = 2 := by
  sorry

end robert_reading_capacity_l3827_382715


namespace quadratic_roots_l3827_382707

theorem quadratic_roots (a b c : ℝ) (h : b^2 - 4*a*c > 0) :
  let f (x : ℝ) := 3*a*x^2 + 2*(a + b)*x + (b + c)
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ f x₁ = 0 ∧ f x₂ = 0 :=
by sorry

end quadratic_roots_l3827_382707


namespace meeting_2015_same_as_first_l3827_382772

/-- Represents a point on a line segment --/
structure Point :=
  (position : ℝ)

/-- Represents a person moving on the line segment --/
structure Person :=
  (startPoint : Point)
  (speed : ℝ)
  (startTime : ℝ)

/-- Represents a meeting between two people --/
def Meeting := ℕ → Point

/-- The movement pattern of two people as described in the problem --/
def movementPattern (person1 person2 : Person) : Meeting :=
  sorry

/-- Theorem stating that the 2015th meeting point is the same as the first meeting point --/
theorem meeting_2015_same_as_first 
  (person1 person2 : Person) (pattern : Meeting := movementPattern person1 person2) :
  pattern 2015 = pattern 1 :=
sorry

end meeting_2015_same_as_first_l3827_382772


namespace triangle_formation_l3827_382724

theorem triangle_formation (a b c : ℝ) : 
  a = 4 ∧ b = 9 ∧ c = 9 →
  a + b > c ∧ b + c > a ∧ c + a > b :=
by sorry

end triangle_formation_l3827_382724


namespace inequality_proof_l3827_382776

theorem inequality_proof (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  64 * (a * b * c * d + 1) / (a + b + c + d)^2 ≤ 
  a^2 + b^2 + c^2 + d^2 + 1/a^2 + 1/b^2 + 1/c^2 + 1/d^2 := by
  sorry

end inequality_proof_l3827_382776


namespace ticket_ratio_l3827_382712

/-- Prove the ratio of Peyton's tickets to Tate's total tickets -/
theorem ticket_ratio :
  let tate_initial : ℕ := 32
  let tate_bought : ℕ := 2
  let total_tickets : ℕ := 51
  let tate_total : ℕ := tate_initial + tate_bought
  let peyton_tickets : ℕ := total_tickets - tate_total
  (peyton_tickets : ℚ) / tate_total = 1 / 2 := by
  sorry

end ticket_ratio_l3827_382712


namespace clara_triple_anna_age_l3827_382744

def anna_current_age : ℕ := 54
def clara_current_age : ℕ := 80

theorem clara_triple_anna_age :
  ∃ (years_ago : ℕ), 
    clara_current_age - years_ago = 3 * (anna_current_age - years_ago) ∧
    years_ago = 41 :=
by sorry

end clara_triple_anna_age_l3827_382744
