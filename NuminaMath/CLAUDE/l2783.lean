import Mathlib

namespace simplify_sqrt_expression_l2783_278395

theorem simplify_sqrt_expression (x : ℝ) (hx : x ≠ 0) :
  Real.sqrt (1 + ((x^6 - 2) / (3 * x^3))^2) = (Real.sqrt (x^12 + 5*x^6 + 4)) / (3 * x^3) :=
by sorry

end simplify_sqrt_expression_l2783_278395


namespace quadratic_polynomial_satisfies_conditions_l2783_278364

theorem quadratic_polynomial_satisfies_conditions :
  ∃ (q : ℝ → ℝ),
    (∀ x, q x = -3 * x^2 + 9 * x + 54) ∧
    q (-3) = 0 ∧
    q 6 = 0 ∧
    q 0 = -54 :=
by
  sorry

end quadratic_polynomial_satisfies_conditions_l2783_278364


namespace negation_of_proposition_l2783_278378

theorem negation_of_proposition :
  ¬(∀ x : ℝ, x > 0 → ¬(x > 0)) ↔ ∃ x : ℝ, x ≤ 0 := by sorry

end negation_of_proposition_l2783_278378


namespace f_monotonically_decreasing_implies_k_ge_160_l2783_278304

-- Define the function f(x)
def f (k : ℝ) (x : ℝ) : ℝ := 4 * x^2 - k * x - 8

-- Define the property of being monotonically decreasing on an interval
def monotonically_decreasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f y < f x

-- Theorem statement
theorem f_monotonically_decreasing_implies_k_ge_160 :
  ∀ k : ℝ, monotonically_decreasing_on (f k) 5 20 → k ≥ 160 :=
sorry

end f_monotonically_decreasing_implies_k_ge_160_l2783_278304


namespace quadratic_roots_problem_l2783_278354

theorem quadratic_roots_problem (x₁ x₂ m : ℝ) : 
  (x₁^2 - 2*(m+1)*x₁ + m^2 - 3 = 0) →
  (x₂^2 - 2*(m+1)*x₂ + m^2 - 3 = 0) →
  (x₁^2 + x₂^2 - x₁*x₂ = 33) →
  (m = 2) := by sorry

end quadratic_roots_problem_l2783_278354


namespace harmonic_sum_equality_l2783_278342

/-- The nth harmonic number -/
def h (n : ℕ+) : ℚ :=
  (Finset.range n).sum (fun i => 1 / (i + 1 : ℚ))

/-- The sum of harmonic numbers up to n-1 -/
def sum_h (n : ℕ+) : ℚ :=
  (Finset.range (n - 1)).sum (fun i => h ⟨i + 1, Nat.succ_pos i⟩)

/-- The main theorem: n + sum of h(1) to h(n-1) equals n * h(n) for n ≥ 2 -/
theorem harmonic_sum_equality (n : ℕ+) (hn : n ≥ 2) :
  (n : ℚ) + sum_h n = n * h n := by sorry

end harmonic_sum_equality_l2783_278342


namespace water_tank_capacity_l2783_278385

theorem water_tank_capacity : ∀ C : ℝ, 
  (0.4 * C - 0.1 * C = 36) → C = 120 := by
  sorry

end water_tank_capacity_l2783_278385


namespace victoria_remaining_balance_l2783_278322

/-- Calculates Victoria's remaining balance after shopping --/
theorem victoria_remaining_balance :
  let initial_amount : ℕ := 500
  let rice_price : ℕ := 20
  let rice_quantity : ℕ := 2
  let wheat_price : ℕ := 25
  let wheat_quantity : ℕ := 3
  let soda_price : ℕ := 150
  let soda_quantity : ℕ := 1
  let total_spent : ℕ := rice_price * rice_quantity + wheat_price * wheat_quantity + soda_price * soda_quantity
  let remaining_balance : ℕ := initial_amount - total_spent
  remaining_balance = 235 := by
  sorry

end victoria_remaining_balance_l2783_278322


namespace perfect_cubes_between_500_and_2000_l2783_278316

theorem perfect_cubes_between_500_and_2000 : 
  (Finset.filter (fun n => 500 ≤ n^3 ∧ n^3 ≤ 2000) (Finset.range 13)).card = 5 := by
  sorry

end perfect_cubes_between_500_and_2000_l2783_278316


namespace quadratic_equation_c_value_l2783_278310

theorem quadratic_equation_c_value : 
  ∀ c : ℝ, 
  (∀ x : ℝ, 2 * x^2 + 8 * x + c = 0 ↔ x = (-8 + Real.sqrt 40) / 4 ∨ x = (-8 - Real.sqrt 40) / 4) → 
  c = 3 := by
sorry

end quadratic_equation_c_value_l2783_278310


namespace smallest_butterfly_count_l2783_278380

theorem smallest_butterfly_count (n : ℕ) : n > 0 → (
  (∃ m k : ℕ, m > 0 ∧ k > 0 ∧ n * 44 = m * 17 ∧ n * 44 = k * 25) ∧
  (∃ t : ℕ, n * 44 + n * 17 + n * 25 = 60 * t) ∧
  (∀ x : ℕ, x > 0 ∧ x < n → 
    ¬(∃ y z : ℕ, y > 0 ∧ z > 0 ∧ x * 44 = y * 17 ∧ x * 44 = z * 25) ∨
    ¬(∃ s : ℕ, x * 44 + x * 17 + x * 25 = 60 * s))
) ↔ n = 425 := by
  sorry

end smallest_butterfly_count_l2783_278380


namespace gerald_remaining_pfennigs_l2783_278319

/-- Represents the number of farthings in a pfennig -/
def farthings_per_pfennig : ℕ := 6

/-- Represents the number of farthings Gerald has -/
def geralds_farthings : ℕ := 54

/-- Represents the cost of a meat pie in pfennigs -/
def meat_pie_cost : ℕ := 2

/-- Calculates the number of pfennigs Gerald will have left after buying the pie -/
def remaining_pfennigs : ℕ :=
  geralds_farthings / farthings_per_pfennig - meat_pie_cost

/-- Theorem stating that Gerald will have 7 pfennigs left after buying the pie -/
theorem gerald_remaining_pfennigs :
  remaining_pfennigs = 7 := by sorry

end gerald_remaining_pfennigs_l2783_278319


namespace share_ratio_l2783_278323

/-- Prove that the ratio of A's share to the combined share of B and C is 2:3 --/
theorem share_ratio (total a b c : ℚ) (x : ℚ) : 
  total = 200 →
  a = 80 →
  a = x * (b + c) →
  b = (6/9) * (a + c) →
  a + b + c = total →
  a / (b + c) = 2/3 :=
by sorry

end share_ratio_l2783_278323


namespace fans_attended_l2783_278396

def stadium_capacity : ℕ := 60000
def seats_sold_percentage : ℚ := 75 / 100
def fans_stayed_home : ℕ := 5000

theorem fans_attended (capacity : ℕ) (sold_percentage : ℚ) (stayed_home : ℕ) 
  (h1 : capacity = stadium_capacity)
  (h2 : sold_percentage = seats_sold_percentage)
  (h3 : stayed_home = fans_stayed_home) :
  (capacity : ℚ) * sold_percentage - stayed_home = 40000 := by
  sorry

end fans_attended_l2783_278396


namespace matrix_sum_theorem_l2783_278328

def matrix_element (i j : Nat) : Int :=
  if j % i = 0 then 1 else -1

def sum_3j : Int :=
  (matrix_element 3 2) + (matrix_element 3 3) + (matrix_element 3 4) + (matrix_element 3 5)

def sum_i4 : Int :=
  (matrix_element 2 4) + (matrix_element 3 4) + (matrix_element 4 4)

theorem matrix_sum_theorem : sum_3j + sum_i4 = -1 := by
  sorry

end matrix_sum_theorem_l2783_278328


namespace equation_solutions_l2783_278355

def equation (x : ℝ) : Prop :=
  1 / (x^2 + 9*x - 12) + 1 / (x^2 + 5*x - 14) - 1 / (x^2 - 15*x - 18) = 0

theorem equation_solutions :
  {x : ℝ | equation x} = {2, -9, 6, -3} := by sorry

end equation_solutions_l2783_278355


namespace smallest_a_for_nonempty_solution_l2783_278388

theorem smallest_a_for_nonempty_solution : ∃ (a : ℕ), 
  (a > 0) ∧ 
  (∃ x : ℝ, 2*|x-3| + |x-4| < a^2 + a) ∧
  (∀ b : ℕ, (b > 0 ∧ b < a) → ¬∃ x : ℝ, 2*|x-3| + |x-4| < b^2 + b) ∧
  a = 1 := by
  sorry

end smallest_a_for_nonempty_solution_l2783_278388


namespace ten_player_tournament_matches_l2783_278302

/-- A round-robin tournament where each player plays every other player exactly once. -/
structure RoundRobinTournament where
  num_players : ℕ
  num_players_pos : 0 < num_players

/-- The number of matches in a round-robin tournament. -/
def num_matches (t : RoundRobinTournament) : ℕ := t.num_players.choose 2

theorem ten_player_tournament_matches :
  ∀ t : RoundRobinTournament, t.num_players = 10 → num_matches t = 45 := by
  sorry

end ten_player_tournament_matches_l2783_278302


namespace instantaneous_velocity_at_2_l2783_278386

-- Define the displacement function
def S (t : ℝ) : ℝ := 3 * t - t^2

-- Define the velocity function as the derivative of displacement
def v (t : ℝ) : ℝ := 3 - 2 * t

-- Theorem statement
theorem instantaneous_velocity_at_2 : v 2 = -1 := by
  sorry

end instantaneous_velocity_at_2_l2783_278386


namespace average_of_a_and_b_l2783_278350

theorem average_of_a_and_b (a b c : ℝ) (h1 : (b + c) / 2 = 90) (h2 : c - a = 90) : 
  (a + b) / 2 = 45 := by
  sorry

end average_of_a_and_b_l2783_278350


namespace system_solution_l2783_278324

def satisfies_system (u v w : ℝ) : Prop :=
  u + v * w = 12 ∧ v + w * u = 12 ∧ w + u * v = 12

def solution_set : Set (ℝ × ℝ × ℝ) :=
  {(3, 3, 3), (-4, -4, -4), (1, 1, 11), (11, 1, 1), (1, 11, 1)}

theorem system_solution :
  {p : ℝ × ℝ × ℝ | satisfies_system p.1 p.2.1 p.2.2} = solution_set := by
  sorry

end system_solution_l2783_278324


namespace percentage_difference_l2783_278312

theorem percentage_difference (x y : ℝ) (h : x = 12 * y) :
  (x - y) / x * 100 = 91.67 :=
sorry

end percentage_difference_l2783_278312


namespace rented_cars_at_3600_max_revenue_l2783_278369

/-- Represents the rental company's car fleet and pricing model. -/
structure RentalCompany where
  total_cars : ℕ
  base_rent : ℕ
  rent_increment : ℕ
  rented_maintenance : ℕ
  unrented_maintenance : ℕ

/-- Calculates the number of rented cars given a certain rent. -/
def rented_cars (company : RentalCompany) (rent : ℕ) : ℕ :=
  company.total_cars - (rent - company.base_rent) / company.rent_increment

/-- Calculates the monthly revenue given a certain rent. -/
def monthly_revenue (company : RentalCompany) (rent : ℕ) : ℕ :=
  let rented := rented_cars company rent
  rent * rented - company.rented_maintenance * rented - 
    company.unrented_maintenance * (company.total_cars - rented)

/-- The rental company with the given parameters. -/
def our_company : RentalCompany := {
  total_cars := 100,
  base_rent := 3000,
  rent_increment := 50,
  rented_maintenance := 150,
  unrented_maintenance := 50
}

/-- Theorem stating the number of rented cars when rent is 3600 yuan. -/
theorem rented_cars_at_3600 : 
  rented_cars our_company 3600 = 88 := by sorry

/-- Theorem stating the rent that maximizes revenue and the maximum revenue. -/
theorem max_revenue : 
  ∃ (max_rent : ℕ), max_rent = 4050 ∧ 
  monthly_revenue our_company max_rent = 37050 ∧
  ∀ (rent : ℕ), monthly_revenue our_company rent ≤ monthly_revenue our_company max_rent := by sorry

end rented_cars_at_3600_max_revenue_l2783_278369


namespace reflection_line_sum_l2783_278320

/-- Given a line y = mx + b, if the reflection of point (2,3) across this line is (10,7), then m + b = 15 -/
theorem reflection_line_sum (m b : ℝ) : 
  (∃ (x y : ℝ), 
    -- The midpoint of (2,3) and (10,7) lies on the line y = mx + b
    y = m * x + b ∧ 
    x = (2 + 10) / 2 ∧ 
    y = (3 + 7) / 2 ∧
    -- The slope of the line is perpendicular to the slope of the segment connecting (2,3) and (10,7)
    m * ((7 - 3) / (10 - 2)) = -1) →
  m + b = 15 := by
sorry


end reflection_line_sum_l2783_278320


namespace andrew_age_proof_l2783_278373

/-- Andrew's age in years -/
def andrew_age : ℚ := 30 / 7

/-- Andrew's grandfather's age in years -/
def grandfather_age : ℚ := 15 * andrew_age

theorem andrew_age_proof :
  andrew_age = 30 / 7 ∧
  grandfather_age = 15 * andrew_age ∧
  grandfather_age - andrew_age = 60 :=
sorry

end andrew_age_proof_l2783_278373


namespace javier_children_count_l2783_278375

/-- The number of children in Javier's household -/
def num_children : ℕ := 
  let total_legs : ℕ := 22
  let javier_wife_legs : ℕ := 2 + 2
  let dog_legs : ℕ := 2 * 4
  let cat_legs : ℕ := 1 * 4
  let remaining_legs : ℕ := total_legs - (javier_wife_legs + dog_legs + cat_legs)
  remaining_legs / 2

theorem javier_children_count : num_children = 3 := by
  sorry

end javier_children_count_l2783_278375


namespace correct_calculation_l2783_278387

theorem correct_calculation (x y : ℝ) : 3 * x * y - 2 * y * x = x * y := by
  sorry

end correct_calculation_l2783_278387


namespace michaels_money_ratio_l2783_278311

/-- Given the following conditions:
    - Michael has $42 initially
    - Michael gives some money to his brother
    - His brother buys $3 worth of candy
    - His brother has $35 left after buying candy
    - His brother had $17 at first
    Prove that the ratio of money Michael gave to his brother to Michael's initial money is 1:2 -/
theorem michaels_money_ratio :
  ∀ (initial_money : ℕ) (brother_initial : ℕ) (candy_cost : ℕ) (brother_final : ℕ),
    initial_money = 42 →
    brother_initial = 17 →
    candy_cost = 3 →
    brother_final = 35 →
    ∃ (money_given : ℕ),
      money_given = brother_final + candy_cost - brother_initial ∧
      2 * money_given = initial_money :=
by sorry

end michaels_money_ratio_l2783_278311


namespace sqrt_expression_equality_l2783_278365

theorem sqrt_expression_equality : 
  |Real.sqrt 2 - Real.sqrt 3| - Real.sqrt 4 + Real.sqrt 2 * (Real.sqrt 2 + 1) = Real.sqrt 3 := by
  sorry

end sqrt_expression_equality_l2783_278365


namespace circle_equation_from_diameter_specific_circle_equation_l2783_278333

/-- Given two points A and B as the endpoints of a circle's diameter, 
    prove that the equation of the circle is (x-h)^2 + (y-k)^2 = r^2,
    where (h,k) is the midpoint of AB and r is half the distance between A and B. -/
theorem circle_equation_from_diameter (A B : ℝ × ℝ) :
  let (x₁, y₁) := A
  let (x₂, y₂) := B
  let h := (x₁ + x₂) / 2
  let k := (y₁ + y₂) / 2
  let r := Real.sqrt (((x₁ - x₂)^2 + (y₁ - y₂)^2) / 4)
  ∀ (x y : ℝ), (x - h)^2 + (y - k)^2 = r^2 ↔ 
    ((x - x₁)^2 + (y - y₁)^2) * ((x - x₂)^2 + (y - y₂)^2) = 
    ((x - x₁)^2 + (y - y₁)^2 + (x - x₂)^2 + (y - y₂)^2)^2 / 4 :=
by sorry

/-- The equation of the circle with diameter endpoints A(4,9) and B(6,3) is (x-5)^2 + (y-6)^2 = 10 -/
theorem specific_circle_equation : 
  ∀ (x y : ℝ), (x - 5)^2 + (y - 6)^2 = 10 ↔ 
    ((x - 4)^2 + (y - 9)^2) * ((x - 6)^2 + (y - 3)^2) = 
    ((x - 4)^2 + (y - 9)^2 + (x - 6)^2 + (y - 3)^2)^2 / 4 :=
by sorry

end circle_equation_from_diameter_specific_circle_equation_l2783_278333


namespace polynomial_division_theorem_l2783_278352

theorem polynomial_division_theorem (x : ℝ) :
  (8*x^3 - 9*x^2 + 21*x - 47) * (x + 2) + 86 = 8*x^4 + 7*x^3 + 3*x^2 - 5*x - 8 := by
  sorry

end polynomial_division_theorem_l2783_278352


namespace new_members_weight_l2783_278300

/-- Theorem: Calculate the combined weight of new group members -/
theorem new_members_weight (original_size : ℕ) (weight_increase : ℝ) 
  (original_member1 original_member2 original_member3 : ℝ) :
  original_size = 8 →
  weight_increase = 4.2 →
  original_member1 = 60 →
  original_member2 = 75 →
  original_member3 = 65 →
  (original_member1 + original_member2 + original_member3 + 
    original_size * weight_increase) = 233.6 := by
  sorry

end new_members_weight_l2783_278300


namespace edwards_spending_l2783_278377

theorem edwards_spending (initial_amount : ℚ) : 
  initial_amount - 130 - (0.25 * (initial_amount - 130)) = 270 → 
  initial_amount = 490 := by
sorry

end edwards_spending_l2783_278377


namespace circle_equation_characterization_l2783_278366

/-- A circle with center on the x-axis, radius √2, passing through (-2, 1) -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ
  passes_through : ℝ × ℝ
  center_on_x_axis : center.2 = 0
  radius_is_sqrt_2 : radius = Real.sqrt 2
  passes_through_point : passes_through = (-2, 1)

/-- The equation of the circle -/
def circle_equation (c : Circle) (x y : ℝ) : Prop :=
  ((x - c.center.1) ^ 2 + y ^ 2) = c.radius ^ 2

theorem circle_equation_characterization (c : Circle) :
  ∃ a : ℝ, (a = -1 ∨ a = -3) ∧
    ∀ x y : ℝ, circle_equation c x y ↔ ((x + a) ^ 2 + y ^ 2 = 2) :=
  sorry

end circle_equation_characterization_l2783_278366


namespace not_necessary_not_sufficient_condition_l2783_278397

theorem not_necessary_not_sufficient_condition (a b : ℝ) : 
  ¬(((a > 0 ∧ b > 0) → (a * b < ((a + b) / 2)^2)) ∧
    ((a * b < ((a + b) / 2)^2) → (a > 0 ∧ b > 0))) := by
  sorry

end not_necessary_not_sufficient_condition_l2783_278397


namespace problem_solution_l2783_278357

theorem problem_solution (x : ℝ) (h : x + 1/x = 3) :
  (x - 3)^4 + 81 / (x - 3)^4 = 63 := by
  sorry

end problem_solution_l2783_278357


namespace probability_of_specific_draw_l2783_278394

def total_silverware : ℕ := 24
def forks : ℕ := 8
def spoons : ℕ := 8
def knives : ℕ := 8
def pieces_drawn : ℕ := 4

def favorable_outcomes : ℕ := forks * spoons * knives * (forks - 1 + spoons - 1 + knives - 1)
def total_outcomes : ℕ := Nat.choose total_silverware pieces_drawn

theorem probability_of_specific_draw :
  (favorable_outcomes : ℚ) / total_outcomes = 214 / 253 := by sorry

end probability_of_specific_draw_l2783_278394


namespace total_jogging_distance_l2783_278308

/-- The total distance jogged over three days is the sum of the distances jogged each day. -/
theorem total_jogging_distance 
  (monday_distance tuesday_distance wednesday_distance : ℕ) 
  (h1 : monday_distance = 2)
  (h2 : tuesday_distance = 5)
  (h3 : wednesday_distance = 9) :
  monday_distance + tuesday_distance + wednesday_distance = 16 := by
sorry

end total_jogging_distance_l2783_278308


namespace f_derivative_correct_l2783_278337

/-- The exponential function -/
noncomputable def exp (x : ℝ) : ℝ := Real.exp x

/-- The function f(x) = e^(-2x) -/
noncomputable def f (x : ℝ) : ℝ := exp (-2 * x)

/-- The derivative of f(x) -/
noncomputable def f_derivative (x : ℝ) : ℝ := -2 * exp (-2 * x)

theorem f_derivative_correct :
  ∀ x : ℝ, deriv f x = f_derivative x :=
by sorry

end f_derivative_correct_l2783_278337


namespace car_distance_proof_l2783_278348

/-- Calculates the distance traveled by a car given its speed and time -/
def distance_traveled (speed : ℝ) (time : ℝ) : ℝ := speed * time

/-- The speed of the car in miles per hour -/
def car_speed : ℝ := 80

/-- The time the car traveled in hours -/
def travel_time : ℝ := 4.5

theorem car_distance_proof : 
  distance_traveled car_speed travel_time = 360 := by sorry

end car_distance_proof_l2783_278348


namespace fourth_group_frequency_l2783_278360

/-- Given a set of data with 50 items divided into 5 groups, prove that the frequency of the fourth group is 12 -/
theorem fourth_group_frequency
  (total_items : ℕ)
  (num_groups : ℕ)
  (freq_group1 : ℕ)
  (freq_group2 : ℕ)
  (freq_group3 : ℕ)
  (freq_group5 : ℕ)
  (h_total : total_items = 50)
  (h_groups : num_groups = 5)
  (h_freq1 : freq_group1 = 10)
  (h_freq2 : freq_group2 = 8)
  (h_freq3 : freq_group3 = 11)
  (h_freq5 : freq_group5 = 9) :
  total_items - (freq_group1 + freq_group2 + freq_group3 + freq_group5) = 12 :=
by sorry

end fourth_group_frequency_l2783_278360


namespace right_triangle_in_circle_l2783_278329

theorem right_triangle_in_circle (diameter : ℝ) (leg1 : ℝ) (leg2 : ℝ) : 
  diameter = 10 → leg1 = 6 → leg2 * leg2 = diameter * diameter - leg1 * leg1 → leg2 = 8 := by
  sorry

end right_triangle_in_circle_l2783_278329


namespace inequality_holds_iff_equal_l2783_278376

/-- The floor function -/
noncomputable def floor (x : ℝ) : ℤ :=
  Int.floor x

/-- The inequality holds for all real α and β iff m = n -/
theorem inequality_holds_iff_equal (m n : ℕ+) : 
  (∀ α β : ℝ, floor ((m + n : ℝ) * α) + floor ((m + n : ℝ) * β) ≥ 
    floor (m * α) + floor (n * β) + floor (n * (α + β))) ↔ m = n := by
  sorry

end inequality_holds_iff_equal_l2783_278376


namespace exists_divisible_by_2022_l2783_278371

def concatenate_numbers (n m : ℕ) : ℕ :=
  sorry

theorem exists_divisible_by_2022 :
  ∃ n m : ℕ, n > m ∧ m ≥ 1 ∧ (concatenate_numbers n m) % 2022 = 0 :=
sorry

end exists_divisible_by_2022_l2783_278371


namespace moving_circles_touch_times_l2783_278346

/-- The problem of two moving circles touching each other --/
theorem moving_circles_touch_times
  (r₁ : ℝ) (v₁ : ℝ) (d₁ : ℝ)
  (r₂ : ℝ) (v₂ : ℝ) (d₂ : ℝ)
  (h₁ : r₁ = 981)
  (h₂ : v₁ = 7)
  (h₃ : d₁ = 2442)
  (h₄ : r₂ = 980)
  (h₅ : v₂ = 5)
  (h₆ : d₂ = 1591) :
  ∃ (t₁ t₂ : ℝ),
    t₁ = 111 ∧ t₂ = 566 ∧
    (∀ t, (d₁ - v₁ * t)^2 + (d₂ - v₂ * t)^2 = (r₁ + r₂)^2 → t = t₁ ∨ t = t₂) :=
by sorry

end moving_circles_touch_times_l2783_278346


namespace range_of_m_range_of_t_l2783_278341

noncomputable section

-- Define the functions f and g
def f (x t : ℝ) : ℝ := -x^2 + 2 * Real.exp 1 * x + t - 1
def g (x : ℝ) : ℝ := x + (Real.exp 1)^2 / x

-- State the theorem for the range of m
theorem range_of_m :
  ∀ m : ℝ, (∃ x : ℝ, x > 0 ∧ g x = m) ↔ m ≥ 2 * Real.exp 1 :=
sorry

-- State the theorem for the range of t
theorem range_of_t :
  ∀ t : ℝ, (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁ > 0 ∧ x₂ > 0 ∧ g x₁ - f x₁ t = 0 ∧ g x₂ - f x₂ t = 0)
  ↔ t > 2 * Real.exp 1 - (Real.exp 1)^2 + 1 :=
sorry

end

end range_of_m_range_of_t_l2783_278341


namespace sphere_radius_from_surface_area_l2783_278330

theorem sphere_radius_from_surface_area (A : ℝ) (r : ℝ) (h : A = 64 * Real.pi) :
  A = 4 * Real.pi * r^2 → r = 4 := by
  sorry

end sphere_radius_from_surface_area_l2783_278330


namespace ratio_transformation_l2783_278359

theorem ratio_transformation (a c : ℝ) (h : c ≠ 0) :
  (3 * a) / (c / 3) = 9 * (a / c) := by sorry

end ratio_transformation_l2783_278359


namespace sin_equation_range_l2783_278317

theorem sin_equation_range : 
  let f : ℝ → ℝ := λ x => Real.sin x ^ 2 - 2 * Real.sin x
  ∃ (a_min a_max : ℝ), a_min = -1 ∧ a_max = 3 ∧
    (∀ a : ℝ, (∃ x : ℝ, f x = a) ↔ a_min ≤ a ∧ a ≤ a_max) :=
by sorry

end sin_equation_range_l2783_278317


namespace expression_evaluation_l2783_278318

theorem expression_evaluation : 3 - 5 * (6 - 2^3) / 2 = 8 := by
  sorry

end expression_evaluation_l2783_278318


namespace complex_number_range_l2783_278326

theorem complex_number_range (m : ℝ) :
  let z : ℂ := 1 + Complex.I + m / (1 + Complex.I)
  (0 < z.re ∧ 0 < z.im) ↔ -2 < m ∧ m < 2 := by
  sorry

end complex_number_range_l2783_278326


namespace rachels_apple_tree_l2783_278358

theorem rachels_apple_tree (initial : ℕ) : 
  (initial - 2 + 3 = 5) → initial = 4 := by
  sorry

end rachels_apple_tree_l2783_278358


namespace larger_integer_value_l2783_278382

theorem larger_integer_value (a b : ℕ+) 
  (h_quotient : (a : ℚ) / (b : ℚ) = 7 / 3)
  (h_product : (a : ℕ) * b = 189) :
  max a b = 21 := by
  sorry

end larger_integer_value_l2783_278382


namespace lizzies_group_size_l2783_278363

theorem lizzies_group_size (total : ℕ) (difference : ℕ) : 
  total = 91 → difference = 17 → ∃ (other : ℕ), other + (other + difference) = total ∧ other + difference = 54 :=
by
  sorry

end lizzies_group_size_l2783_278363


namespace opposite_of_two_and_two_thirds_l2783_278315

theorem opposite_of_two_and_two_thirds :
  -(2 + 2/3) = -(2 + 2/3) := by sorry

end opposite_of_two_and_two_thirds_l2783_278315


namespace tangent_line_inclination_angle_range_l2783_278389

open Real Set

theorem tangent_line_inclination_angle_range :
  ∀ x : ℝ, 
  let P : ℝ × ℝ := (x, Real.sin x)
  let θ := Real.arctan (Real.cos x)
  θ ∈ Icc 0 (π/4) ∪ Ico (3*π/4) π := by
sorry

end tangent_line_inclination_angle_range_l2783_278389


namespace average_salary_l2783_278327

def salary_a : ℕ := 9000
def salary_b : ℕ := 5000
def salary_c : ℕ := 11000
def salary_d : ℕ := 7000
def salary_e : ℕ := 9000

def total_salary : ℕ := salary_a + salary_b + salary_c + salary_d + salary_e
def num_people : ℕ := 5

theorem average_salary :
  (total_salary : ℚ) / num_people = 8200 := by sorry

end average_salary_l2783_278327


namespace weight_of_a_l2783_278306

/-- Given the weights of five people A, B, C, D, and E, prove that A weighs 75 kg -/
theorem weight_of_a (a b c d e : ℝ) : 
  (a + b + c) / 3 = 84 →
  (a + b + c + d) / 4 = 80 →
  e = d + 3 →
  (b + c + d + e) / 4 = 79 →
  a = 75 := by
sorry

end weight_of_a_l2783_278306


namespace reciprocal_expression_l2783_278367

theorem reciprocal_expression (x y : ℝ) (h : x * y = 1) :
  (x + 1 / y) * (2 * y - 1 / x) = 2 := by
  sorry

end reciprocal_expression_l2783_278367


namespace min_draws_for_sum_30_l2783_278336

-- Define the set of integers from 0 to 20
def integerSet : Set ℕ := {n : ℕ | n ≤ 20}

-- Define a function to check if two numbers in a list sum to 30
def hasPairSum30 (list : List ℕ) : Prop :=
  ∃ (a b : ℕ), a ∈ list ∧ b ∈ list ∧ a ≠ b ∧ a + b = 30

-- Theorem: The minimum number of integers to guarantee a pair summing to 30 is 10
theorem min_draws_for_sum_30 :
  ∀ (drawn : List ℕ),
    (∀ n ∈ drawn, n ∈ integerSet) →
    (drawn.length ≥ 10 → hasPairSum30 drawn) ∧
    (∃ subset : List ℕ, subset.length = 9 ∧ ∀ n ∈ subset, n ∈ integerSet ∧ ¬hasPairSum30 subset) :=
by sorry

end min_draws_for_sum_30_l2783_278336


namespace agathas_bike_frame_cost_l2783_278340

/-- Agatha's bike purchase problem -/
theorem agathas_bike_frame_cost (total : ℕ) (wheel_cost : ℕ) (remaining : ℕ) (frame_cost : ℕ) :
  total = 60 →
  wheel_cost = 25 →
  remaining = 20 →
  frame_cost = total - wheel_cost - remaining →
  frame_cost = 15 := by
sorry

end agathas_bike_frame_cost_l2783_278340


namespace reflected_ray_equation_l2783_278370

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a line in 2D space -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Function to reflect a point about the y-axis -/
def reflectAboutYAxis (p : Point) : Point :=
  { x := -p.x, y := p.y }

/-- Function to get the equation of a line given two points -/
def lineFromPoints (p1 p2 : Point) : Line :=
  { a := p2.y - p1.y,
    b := p1.x - p2.x,
    c := p1.y * p2.x - p1.x * p2.y }

/-- Theorem stating the equation of the reflected ray -/
theorem reflected_ray_equation
  (start : Point)
  (slope : ℝ)
  (h_start : start = { x := 2, y := 3 })
  (h_slope : slope = 1/2) :
  let intersect : Point := { x := 0, y := 2 }
  let reflected_start : Point := reflectAboutYAxis start
  let reflected_line : Line := lineFromPoints intersect reflected_start
  reflected_line = { a := 1, b := 2, c := -4 } :=
sorry

end reflected_ray_equation_l2783_278370


namespace bow_collection_problem_l2783_278332

theorem bow_collection_problem (total : ℕ) (yellow : ℕ) :
  yellow = 36 →
  (1 : ℚ) / 4 * total + (1 : ℚ) / 3 * total + (1 : ℚ) / 6 * total + yellow = total →
  (1 : ℚ) / 6 * total = 24 := by
  sorry

end bow_collection_problem_l2783_278332


namespace intersection_line_l2783_278301

-- Define the two circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 = 10
def circle2 (x y : ℝ) : Prop := (x-1)^2 + (y-3)^2 = 10

-- Define the line
def line (x y : ℝ) : Prop := x + 3*y - 5 = 0

-- Theorem statement
theorem intersection_line :
  ∀ x y : ℝ, circle1 x y ∧ circle2 x y → line x y :=
by
  sorry

end intersection_line_l2783_278301


namespace scalar_cross_product_sum_l2783_278392

variable {V : Type*} [AddCommGroup V] [Module ℝ V]

def cross_product : V → V → V := sorry

theorem scalar_cross_product_sum (a b c d : V) (h : a + b + c + d = 0) :
  ∃! k : ℝ, ∀ (a b c d : V), a + b + c + d = 0 →
    k • (cross_product c b) + cross_product b c + cross_product c a + 
    cross_product a d + cross_product d d = 0 ∧ k = 0 := by
  sorry

end scalar_cross_product_sum_l2783_278392


namespace equation_solution_l2783_278381

theorem equation_solution :
  let f (x : ℝ) := 1 / (x + 8) + 1 / (x + 5) - 1 / (x + 11) - 1 / (x + 4)
  ∀ x : ℝ, f x = 0 ↔ x = (-3 + Real.sqrt 37) / 2 ∨ x = (-3 - Real.sqrt 37) / 2 := by
  sorry

end equation_solution_l2783_278381


namespace different_color_probability_l2783_278334

/-- The probability of drawing two chips of different colors from a bag containing 
    7 blue chips, 5 yellow chips, and 4 red chips, when drawing with replacement. -/
theorem different_color_probability :
  let total_chips := 7 + 5 + 4
  let p_blue := 7 / total_chips
  let p_yellow := 5 / total_chips
  let p_red := 4 / total_chips
  let p_different := p_blue * (p_yellow + p_red) + 
                     p_yellow * (p_blue + p_red) + 
                     p_red * (p_blue + p_yellow)
  p_different = 83 / 128 :=
by sorry

end different_color_probability_l2783_278334


namespace cards_distribution_l2783_278331

theorem cards_distribution (total_cards : Nat) (num_people : Nat) (h1 : total_cards = 60) (h2 : num_people = 9) :
  let cards_per_person := total_cards / num_people
  let extra_cards := total_cards % num_people
  let people_with_extra := extra_cards
  num_people - people_with_extra = 3 := by
  sorry

end cards_distribution_l2783_278331


namespace trap_is_feeder_l2783_278345

/-- A sequence of real numbers -/
def Sequence := ℕ → ℝ

/-- An interval is a trap (cover) for a sequence if only finitely many terms lie outside it -/
def IsTrap (s : Sequence) (a b : ℝ) : Prop :=
  ∃ N : ℕ, ∀ n : ℕ, n ≥ N → a ≤ s n ∧ s n ≤ b

/-- An interval is a feeder for a sequence if infinitely many terms lie inside it -/
def IsFeeder (s : Sequence) (a b : ℝ) : Prop :=
  ∀ N : ℕ, ∃ n : ℕ, n ≥ N ∧ a ≤ s n ∧ s n ≤ b

/-- Theorem: Every trap is a feeder -/
theorem trap_is_feeder (s : Sequence) (a b : ℝ) :
  IsTrap s a b → IsFeeder s a b := by
  sorry

end trap_is_feeder_l2783_278345


namespace cubic_root_sum_of_squares_reciprocal_l2783_278384

theorem cubic_root_sum_of_squares_reciprocal (a b c : ℝ) : 
  a^3 - 12*a^2 + 20*a - 3 = 0 → 
  b^3 - 12*b^2 + 20*b - 3 = 0 → 
  c^3 - 12*c^2 + 20*c - 3 = 0 → 
  a ≠ b → b ≠ c → a ≠ c →
  1/a^2 + 1/b^2 + 1/c^2 = 328/9 := by sorry

end cubic_root_sum_of_squares_reciprocal_l2783_278384


namespace polar_to_rectangular_min_a_for_inequality_l2783_278313

-- Part A
theorem polar_to_rectangular (ρ θ : ℝ) (x y : ℝ) :
  ρ^2 * Real.cos θ - ρ = 0 ↔ x = 1 :=
sorry

-- Part B
theorem min_a_for_inequality (a : ℝ) :
  (∀ x ∈ Set.Icc 0 5, |2 - x| + |x + 1| ≤ a) ↔ a ≥ 9 :=
sorry

end polar_to_rectangular_min_a_for_inequality_l2783_278313


namespace two_angles_in_fourth_quadrant_l2783_278368

def is_fourth_quadrant (angle : Int) : Bool :=
  let normalized := angle % 360
  normalized > 270 || normalized ≤ 0

def count_fourth_quadrant (angles : List Int) : Nat :=
  (angles.filter is_fourth_quadrant).length

theorem two_angles_in_fourth_quadrant :
  count_fourth_quadrant [-20, -400, -2000, 1600] = 2 := by
  sorry

end two_angles_in_fourth_quadrant_l2783_278368


namespace absolute_value_equality_l2783_278351

theorem absolute_value_equality (x : ℝ) : 
  |x^2 - 8*x + 12| = x^2 - 8*x + 12 ↔ x ≤ 2 ∨ x ≥ 6 := by sorry

end absolute_value_equality_l2783_278351


namespace probability_one_white_ball_l2783_278309

/-- The probability of drawing exactly one white ball when drawing three balls from a bag containing
    four white balls and three black balls of the same size. -/
theorem probability_one_white_ball (total_balls : ℕ) (white_balls : ℕ) (black_balls : ℕ) 
  (h1 : total_balls = white_balls + black_balls)
  (h2 : white_balls = 4)
  (h3 : black_balls = 3)
  (h4 : total_balls > 0) :
  (white_balls : ℚ) / total_balls * 
  (black_balls : ℚ) / (total_balls - 1) * 
  (black_balls - 1 : ℚ) / (total_balls - 2) * 3 = 12 / 35 :=
sorry

end probability_one_white_ball_l2783_278309


namespace lillys_daily_savings_l2783_278338

/-- Proves that the daily savings amount is $2 given the conditions of Lilly's flower-buying plan for Maria's birthday. -/
theorem lillys_daily_savings 
  (saving_period : ℕ) 
  (flower_cost : ℚ) 
  (total_flowers : ℕ) 
  (h1 : saving_period = 22)
  (h2 : flower_cost = 4)
  (h3 : total_flowers = 11) : 
  (total_flowers : ℚ) * flower_cost / saving_period = 2 := by
  sorry

end lillys_daily_savings_l2783_278338


namespace profit_percentage_l2783_278347

theorem profit_percentage (selling_price cost_price : ℝ) (h : cost_price = 0.75 * selling_price) :
  (selling_price - cost_price) / cost_price * 100 = 100 / 3 := by
  sorry

end profit_percentage_l2783_278347


namespace angle_sum_around_point_l2783_278321

theorem angle_sum_around_point (y : ℝ) : 
  y > 0 ∧ 150 + y + y = 360 → y = 105 := by
  sorry

end angle_sum_around_point_l2783_278321


namespace subset_implies_a_value_l2783_278361

def A : Set ℤ := {0, 1}
def B (a : ℤ) : Set ℤ := {-1, 0, a+3}

theorem subset_implies_a_value (h : A ⊆ B a) : a = -2 := by
  sorry

end subset_implies_a_value_l2783_278361


namespace opposite_face_color_l2783_278314

/-- Represents the colors used on the cube faces -/
inductive Color
| Orange
| Silver
| Yellow
| Violet
| Indigo
| Turquoise

/-- Represents a face of the cube -/
inductive Face
| Top
| Bottom
| Front
| Back
| Left
| Right

/-- Represents a view of the cube -/
structure View where
  top : Color
  front : Color
  right : Color

/-- Represents a cube with colored faces -/
structure Cube where
  faces : Face → Color

/-- Checks if all colors in a list are unique -/
def allUnique (colors : List Color) : Prop :=
  colors.Nodup

/-- The theorem to be proved -/
theorem opposite_face_color (c : Cube)
  (view1 : View)
  (view2 : View)
  (view3 : View)
  (h1 : view1 = { top := Color.Orange, front := Color.Yellow, right := Color.Silver })
  (h2 : view2 = { top := Color.Orange, front := Color.Indigo, right := Color.Silver })
  (h3 : view3 = { top := Color.Orange, front := Color.Violet, right := Color.Silver })
  (h_unique : allUnique [Color.Orange, Color.Silver, Color.Yellow, Color.Violet, Color.Indigo, Color.Turquoise])
  (h_cube : c.faces Face.Top = Color.Orange ∧
            c.faces Face.Right = Color.Silver ∧
            c.faces Face.Front ∈ [Color.Yellow, Color.Indigo, Color.Violet] ∧
            c.faces Face.Left ∈ [Color.Yellow, Color.Indigo, Color.Violet] ∧
            c.faces Face.Back ∈ [Color.Yellow, Color.Indigo, Color.Violet])
  : c.faces Face.Bottom = Color.Turquoise → c.faces Face.Top = Color.Orange :=
by sorry

end opposite_face_color_l2783_278314


namespace patio_layout_change_l2783_278398

theorem patio_layout_change (initial_tiles initial_rows initial_columns : ℕ) 
  (new_rows : ℕ) (h1 : initial_tiles = 160) (h2 : initial_rows = 10) 
  (h3 : initial_columns * initial_rows = initial_tiles)
  (h4 : new_rows = initial_rows + 4) :
  ∃ (new_columns : ℕ), 
    new_columns * new_rows = initial_tiles ∧ 
    initial_columns - new_columns = 5 := by
  sorry

end patio_layout_change_l2783_278398


namespace no_real_solutions_ratio_equation_l2783_278391

theorem no_real_solutions_ratio_equation :
  ∀ x : ℝ, (x + 3) / (2 * x + 5) ≠ (5 * x + 4) / (8 * x + 6) :=
by
  sorry

end no_real_solutions_ratio_equation_l2783_278391


namespace floor_sqrt_245_l2783_278344

theorem floor_sqrt_245 : ⌊Real.sqrt 245⌋ = 15 := by
  sorry

end floor_sqrt_245_l2783_278344


namespace mean_equality_implies_z_l2783_278349

theorem mean_equality_implies_z (z : ℝ) : 
  (8 + 15 + 24) / 3 = (16 + z) / 2 → z = 15.34 := by
  sorry

end mean_equality_implies_z_l2783_278349


namespace divisibility_and_sum_of_primes_l2783_278362

theorem divisibility_and_sum_of_primes :
  ∃ (p₁ p₂ p₃ : ℕ),
    Prime p₁ ∧ Prime p₂ ∧ Prime p₃ ∧
    p₁ ≠ p₂ ∧ p₁ ≠ p₃ ∧ p₂ ≠ p₃ ∧
    (p₁ ∣ (2^10 - 1)) ∧ (p₂ ∣ (2^10 - 1)) ∧ (p₃ ∣ (2^10 - 1)) ∧
    (∀ q : ℕ, Prime q → (q ∣ (2^10 - 1)) → (q = p₁ ∨ q = p₂ ∨ q = p₃)) ∧
    p₁ + p₂ + p₃ = 45 :=
by sorry

end divisibility_and_sum_of_primes_l2783_278362


namespace triangle_problem_l2783_278303

theorem triangle_problem (a b c : ℝ) 
  (h : |a - Real.sqrt 7| + Real.sqrt (b - 5) + (c - 4 * Real.sqrt 2)^2 = 0) :
  a = Real.sqrt 7 ∧ b = 5 ∧ c = 4 * Real.sqrt 2 ∧
  ∃ (x y z : ℝ), x^2 + y^2 = z^2 ∧ 
  Set.toFinset {x, y, z} = Set.toFinset {a, b, c} :=
sorry

end triangle_problem_l2783_278303


namespace frisbee_throwing_problem_l2783_278399

/-- Frisbee throwing problem -/
theorem frisbee_throwing_problem 
  (bess_distance : ℝ) 
  (bess_throws : ℕ) 
  (holly_throws : ℕ) 
  (total_distance : ℝ) 
  (h1 : bess_distance = 20)
  (h2 : bess_throws = 4)
  (h3 : holly_throws = 5)
  (h4 : total_distance = 200)
  (h5 : bess_distance * bess_throws * 2 + holly_throws * holly_distance = total_distance) :
  holly_distance = 8 := by
  sorry


end frisbee_throwing_problem_l2783_278399


namespace roses_in_vase_l2783_278383

-- Define the initial number of roses and orchids
def initial_roses : ℕ := 5
def initial_orchids : ℕ := 3

-- Define the current number of orchids
def current_orchids : ℕ := 2

-- Define the difference between roses and orchids
def rose_orchid_difference : ℕ := 10

-- Theorem to prove
theorem roses_in_vase :
  ∃ (current_roses : ℕ),
    current_roses = current_orchids + rose_orchid_difference ∧
    current_roses > initial_roses ∧
    current_roses = 12 :=
by sorry

end roses_in_vase_l2783_278383


namespace no_integer_solution_quadratic_l2783_278339

theorem no_integer_solution_quadratic (x : ℤ) : x^2 + 3 ≥ 2*x := by
  sorry

end no_integer_solution_quadratic_l2783_278339


namespace perimeter_of_square_region_l2783_278353

/-- The perimeter of a region formed by 8 congruent squares arranged in a 2x4 rectangle,
    given that the total area of the region is 512 square centimeters. -/
theorem perimeter_of_square_region (total_area : ℝ) (num_squares : ℕ) (rows cols : ℕ) :
  total_area = 512 →
  num_squares = 8 →
  rows = 2 →
  cols = 4 →
  let square_side := Real.sqrt (total_area / num_squares)
  let perimeter := 2 * square_side * (rows + cols)
  perimeter = 128 := by sorry

end perimeter_of_square_region_l2783_278353


namespace fashion_show_evening_wear_correct_evening_wear_count_l2783_278374

theorem fashion_show_evening_wear (num_models : ℕ) (bathing_suits_per_model : ℕ) 
  (runway_time : ℕ) (total_show_time : ℕ) : ℕ :=
  let total_bathing_suit_trips := num_models * bathing_suits_per_model
  let bathing_suit_time := total_bathing_suit_trips * runway_time
  let evening_wear_time := total_show_time - bathing_suit_time
  let evening_wear_trips := evening_wear_time / runway_time
  evening_wear_trips / num_models

theorem correct_evening_wear_count : 
  fashion_show_evening_wear 6 2 2 60 = 3 := by
  sorry

end fashion_show_evening_wear_correct_evening_wear_count_l2783_278374


namespace no_infinite_sequence_exists_l2783_278343

theorem no_infinite_sequence_exists : ¬ ∃ (k : ℕ → ℝ), 
  (∀ n : ℕ, k (n + 1) = k n - 1 / k n) ∧ 
  (∀ n : ℕ, k n * k (n + 1) ≥ 0) :=
sorry

end no_infinite_sequence_exists_l2783_278343


namespace max_winning_pieces_l2783_278390

/-- Represents the game board -/
def Board := Fin 1000 → Option Nat

/-- The maximum number of pieces a player can place in one turn -/
def max_placement : Nat := 17

/-- Checks if a series of pieces is consecutive -/
def is_consecutive (b : Board) (start finish : Fin 1000) : Prop :=
  ∀ i : Fin 1000, start ≤ i ∧ i ≤ finish → b i.val ≠ none

/-- Represents a valid move by the first player -/
def valid_first_move (b1 b2 : Board) : Prop :=
  ∃ placed : Nat, placed ≤ max_placement ∧
    (∀ i : Fin 1000, b1 i = none → b2 i = none ∨ (∃ n : Nat, b2 i = some n)) ∧
    (∀ i : Fin 1000, b1 i ≠ none → b2 i = b1 i)

/-- Represents a valid move by the second player -/
def valid_second_move (b1 b2 : Board) : Prop :=
  ∃ start finish : Fin 1000, start ≤ finish ∧ is_consecutive b1 start finish ∧
    (∀ i : Fin 1000, (i < start ∨ finish < i) → b2 i = b1 i) ∧
    (∀ i : Fin 1000, start ≤ i ∧ i ≤ finish → b2 i = none)

/-- Checks if the first player has won -/
def first_player_wins (b : Board) (n : Nat) : Prop :=
  ∃ start finish : Fin 1000, start ≤ finish ∧ 
    is_consecutive b start finish ∧
    (∀ i : Fin 1000, i < start ∨ finish < i → b i = none) ∧
    (finish - start + 1 : Nat) = n

/-- The main theorem stating that 98 is the maximum number of pieces for which
    the first player can always win -/
theorem max_winning_pieces : 
  (∀ n : Nat, n ≤ 98 → 
    ∀ initial : Board, (∀ i : Fin 1000, initial i = none) → 
      ∃ strategy : Nat → Board → Board,
        ∀ opponent_strategy : Board → Board,
          ∃ final : Board, first_player_wins final n) ∧
  ¬(∀ n : Nat, n ≤ 99 → 
    ∀ initial : Board, (∀ i : Fin 1000, initial i = none) → 
      ∃ strategy : Nat → Board → Board,
        ∀ opponent_strategy : Board → Board,
          ∃ final : Board, first_player_wins final n) :=
sorry

end max_winning_pieces_l2783_278390


namespace exponential_inequality_l2783_278305

theorem exponential_inequality (x : ℝ) : 
  (1/4 : ℝ)^(x^2 - 8) > (4 : ℝ)^(-2*x) ↔ -2 < x ∧ x < 4 := by
sorry

end exponential_inequality_l2783_278305


namespace vector_v_satisfies_conditions_l2783_278393

/-- Parametric equation of a line in 2D space -/
structure ParamLine2D where
  x : ℝ → ℝ
  y : ℝ → ℝ

/-- Two-dimensional vector -/
structure Vector2D where
  x : ℝ
  y : ℝ

/-- Definition of line l -/
def line_l : ParamLine2D :=
  { x := λ t => 2 + 5*t,
    y := λ t => 3 + 2*t }

/-- Definition of line m -/
def line_m : ParamLine2D :=
  { x := λ s => -7 + 5*s,
    y := λ s => 9 + 2*s }

/-- Point A on line l -/
def point_A : Vector2D :=
  { x := line_l.x 0,
    y := line_l.y 0 }

/-- Point B on line m -/
def point_B : Vector2D :=
  { x := line_m.x 0,
    y := line_m.y 0 }

/-- Vector v that PA is projected onto -/
def vector_v : Vector2D :=
  { x := -2,
    y := 5 }

/-- Theorem: The vector v satisfies the given conditions -/
theorem vector_v_satisfies_conditions :
  vector_v.y - vector_v.x = 7 ∧
  (∃ (P : Vector2D), P.x = point_A.x ∧ P.y = point_A.y) ∧
  (∀ (B : Vector2D), B.x = line_m.x 0 ∧ B.y = line_m.y 0 →
    ∃ (k : ℝ), vector_v.x * k = 0 ∧ vector_v.y * k = 0) :=
by sorry

end vector_v_satisfies_conditions_l2783_278393


namespace round_0_6457_to_hundredth_l2783_278307

/-- Rounds a number to the nearest hundredth -/
def roundToHundredth (x : ℚ) : ℚ :=
  (⌊x * 100 + 0.5⌋ : ℚ) / 100

/-- The theorem states that rounding 0.6457 to the nearest hundredth results in 0.65 -/
theorem round_0_6457_to_hundredth :
  roundToHundredth (6457 / 10000) = 65 / 100 := by sorry

end round_0_6457_to_hundredth_l2783_278307


namespace angle_conversion_l2783_278379

theorem angle_conversion (angle : Real) : ∃ (k : Int) (α : Real), 
  angle = k * (2 * Real.pi) + α ∧ 
  0 ≤ α ∧ 
  α < 2 * Real.pi ∧ 
  angle = -1125 * (Real.pi / 180) ∧ 
  angle = -8 * Real.pi + 7 * Real.pi / 4 := by
  sorry

end angle_conversion_l2783_278379


namespace horner_v4_at_2_l2783_278325

def horner_polynomial (x : ℝ) : ℝ := x^6 - 12*x^5 + 60*x^4 - 160*x^3 + 240*x^2 - 192*x + 64

def horner_v4 (x : ℝ) : ℝ := 
  let v1 := x - 12
  let v2 := v1 * x + 60
  let v3 := v2 * x - 160
  v3 * x + 240

theorem horner_v4_at_2 : horner_v4 2 = 240 := by sorry

end horner_v4_at_2_l2783_278325


namespace pink_roses_count_l2783_278356

theorem pink_roses_count (total_rows : ℕ) (roses_per_row : ℕ) 
  (red_fraction : ℚ) (white_fraction : ℚ) :
  total_rows = 10 →
  roses_per_row = 20 →
  red_fraction = 1/2 →
  white_fraction = 3/5 →
  (total_rows * roses_per_row * (1 - red_fraction) * (1 - white_fraction) : ℚ) = 40 :=
by sorry

end pink_roses_count_l2783_278356


namespace quadratic_coefficient_l2783_278335

/-- A quadratic function with vertex form (x + h)^2 + k -/
def quadratic_vertex_form (a h k : ℝ) (x : ℝ) : ℝ := a * (x + h)^2 + k

theorem quadratic_coefficient (f : ℝ → ℝ) (a : ℝ) :
  (∃ b c : ℝ, ∀ x, f x = a * x^2 + b * x + c) →
  (f (-4) = 0 ∧ f 1 = -75) →
  a = -3 := by
  sorry

end quadratic_coefficient_l2783_278335


namespace area_inequalities_l2783_278372

/-- An acute-angled triangle -/
structure AcuteTriangle where
  A : ℝ
  B : ℝ
  C : ℝ
  acute_A : 0 < A ∧ A < π/2
  acute_B : 0 < B ∧ B < π/2
  acute_C : 0 < C ∧ C < π/2
  sum_angles : A + B + C = π

/-- Area of the orthic triangle -/
def orthic_area (t : AcuteTriangle) : ℝ := sorry

/-- Area of the tangential triangle -/
def tangential_area (t : AcuteTriangle) : ℝ := sorry

/-- Area of the contact triangle -/
def contact_area (t : AcuteTriangle) : ℝ := sorry

/-- Area of the excentral triangle -/
def excentral_area (t : AcuteTriangle) : ℝ := sorry

/-- Area of the medial triangle -/
def medial_area (t : AcuteTriangle) : ℝ := sorry

/-- A triangle is equilateral if all its angles are equal -/
def is_equilateral (t : AcuteTriangle) : Prop :=
  t.A = t.B ∧ t.B = t.C

/-- The main theorem -/
theorem area_inequalities (t : AcuteTriangle) :
  orthic_area t ≤ tangential_area t ∧
  tangential_area t = contact_area t ∧
  contact_area t ≤ excentral_area t ∧
  excentral_area t ≤ medial_area t ∧
  (orthic_area t = medial_area t ↔ is_equilateral t) :=
sorry

end area_inequalities_l2783_278372
