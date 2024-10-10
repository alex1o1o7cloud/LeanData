import Mathlib

namespace probability_in_specific_rectangle_l3057_305774

/-- A rectangle in 2D space --/
structure Rectangle where
  x1 : ℝ
  y1 : ℝ
  x2 : ℝ
  y2 : ℝ

/-- The probability that a randomly selected point in the rectangle is closer to one point than another --/
def probability_closer_to_point (r : Rectangle) (p1 : ℝ × ℝ) (p2 : ℝ × ℝ) : ℝ :=
  sorry

/-- The theorem statement --/
theorem probability_in_specific_rectangle : 
  let r : Rectangle := { x1 := 0, y1 := 0, x2 := 3, y2 := 2 }
  probability_closer_to_point r (0, 0) (4, 2) = 5/6 := by
  sorry

end probability_in_specific_rectangle_l3057_305774


namespace binomial_expansion_problem_l3057_305733

theorem binomial_expansion_problem (m n : ℕ) (hm : m ≠ 0) (hn : n ≥ 2) :
  (∀ k, k ∈ Finset.range (n + 1) → k ≠ 5 → Nat.choose n k ≤ Nat.choose n 5) ∧
  Nat.choose n 2 * m^2 = 9 * Nat.choose n 1 * m →
  m = 2 ∧ n = 10 ∧ ((-17)^n) % 6 = 1 := by
sorry

end binomial_expansion_problem_l3057_305733


namespace fourth_power_inequality_l3057_305781

theorem fourth_power_inequality (a b c : ℝ) : a^4 + b^4 + c^4 ≥ a*b*c*(a + b + c) := by
  sorry

end fourth_power_inequality_l3057_305781


namespace both_sports_lovers_l3057_305732

/-- The number of students who like basketball -/
def basketball_lovers : ℕ := 7

/-- The number of students who like cricket -/
def cricket_lovers : ℕ := 8

/-- The number of students who like basketball or cricket or both -/
def total_lovers : ℕ := 12

/-- The number of students who like both basketball and cricket -/
def both_lovers : ℕ := basketball_lovers + cricket_lovers - total_lovers

theorem both_sports_lovers : both_lovers = 3 := by sorry

end both_sports_lovers_l3057_305732


namespace equation_solution_l3057_305736

theorem equation_solution (a : ℝ) (h1 : a ≠ -2) (h2 : a ≠ -3) (h3 : a ≠ 1/2) :
  let x : ℝ := (2*a - 1) / (a + 3)
  (2 : ℝ) ^ ((a + 3) / (a + 2)) * (32 : ℝ) ^ (1 / (x * (a + 2))) = (4 : ℝ) ^ (1 / x) :=
by sorry

end equation_solution_l3057_305736


namespace min_value_theorem_l3057_305721

theorem min_value_theorem (x y : ℝ) 
  (h1 : x > 1/6) 
  (h2 : y > 0) 
  (h3 : x + y = 1/3) : 
  (∀ a b : ℝ, a > 1/6 ∧ b > 0 ∧ a + b = 1/3 → 
    1/(6*a - 1) + 6/b ≥ 1/(6*x - 1) + 6/y) ∧ 
  1/(6*x - 1) + 6/y = 49 := by
  sorry

end min_value_theorem_l3057_305721


namespace equation_solution_l3057_305755

theorem equation_solution : ∃ x : ℚ, (4 * x + 5 * x = 350 - 10 * (x - 5)) ∧ (x = 400 / 19) := by
  sorry

end equation_solution_l3057_305755


namespace min_xyz_value_l3057_305737

/-- Given real numbers x, y, z satisfying the given conditions, 
    the minimum value of xyz is 9√11 - 32 -/
theorem min_xyz_value (x y z : ℝ) 
    (h1 : x * y + 2 * z = 1) 
    (h2 : x^2 + y^2 + z^2 = 5) : 
  ∀ (a b c : ℝ), a * b + 2 * c = 1 → a^2 + b^2 + c^2 = 5 → 
    x * y * z ≤ a * b * c ∧ 
    ∃ (x₀ y₀ z₀ : ℝ), x₀ * y₀ + 2 * z₀ = 1 ∧ x₀^2 + y₀^2 + z₀^2 = 5 ∧ 
      x₀ * y₀ * z₀ = 9 * Real.sqrt 11 - 32 :=
by
  sorry

#check min_xyz_value

end min_xyz_value_l3057_305737


namespace recipe_total_cups_l3057_305714

/-- Represents the ratio of ingredients in the recipe -/
structure RecipeRatio where
  butter : ℚ
  flour : ℚ
  sugar : ℚ

/-- Calculates the total cups of ingredients given a recipe ratio and cups of sugar used -/
def totalCups (ratio : RecipeRatio) (sugarCups : ℚ) : ℚ :=
  let partValue := sugarCups / ratio.sugar
  ratio.butter * partValue + ratio.flour * partValue + sugarCups

/-- Theorem: Given the specified recipe ratio and sugar amount, the total cups is 27.5 -/
theorem recipe_total_cups :
  let ratio : RecipeRatio := { butter := 1, flour := 6, sugar := 4 }
  totalCups ratio 10 = 27.5 := by
  sorry

#eval totalCups { butter := 1, flour := 6, sugar := 4 } 10

end recipe_total_cups_l3057_305714


namespace remainder_sum_l3057_305768

theorem remainder_sum (c d : ℤ) 
  (hc : c % 52 = 48) 
  (hd : d % 87 = 82) : 
  (c + d) % 29 = 22 := by
sorry

end remainder_sum_l3057_305768


namespace distribute_five_books_three_people_l3057_305782

/-- The number of ways to distribute books among people -/
def distribute_books (n_books : ℕ) (n_people : ℕ) (min_books : ℕ) (max_books : ℕ) : ℕ := sorry

/-- Theorem stating the number of ways to distribute 5 books among 3 people -/
theorem distribute_five_books_three_people : 
  distribute_books 5 3 1 2 = 90 := by sorry

end distribute_five_books_three_people_l3057_305782


namespace dave_trips_l3057_305765

/-- The number of trips Dave needs to make to carry all trays -/
def number_of_trips (trays_per_trip : ℕ) (trays_table1 : ℕ) (trays_table2 : ℕ) : ℕ :=
  (trays_table1 + trays_table2 + trays_per_trip - 1) / trays_per_trip

theorem dave_trips :
  number_of_trips 9 17 55 = 8 :=
by sorry

end dave_trips_l3057_305765


namespace initial_ratio_is_four_to_five_l3057_305792

-- Define the initial number of men and women
variable (M W : ℕ)

-- Define the final number of men and women
def final_men := M + 2
def final_women := 2 * (W - 3)

-- Theorem statement
theorem initial_ratio_is_four_to_five : 
  final_men = 14 ∧ final_women = 24 → M * 5 = W * 4 := by
  sorry

end initial_ratio_is_four_to_five_l3057_305792


namespace mean_equality_implies_z_value_l3057_305750

theorem mean_equality_implies_z_value :
  let mean1 := (8 + 15 + 24) / 3
  let mean2 := (18 + z) / 2
  mean1 = mean2 → z = 40 / 3 := by
sorry

end mean_equality_implies_z_value_l3057_305750


namespace smallest_group_size_l3057_305738

theorem smallest_group_size (n : ℕ) : 
  (n % 18 = 0 ∧ n % 60 = 0) → n ≥ Nat.lcm 18 60 := by
  sorry

#eval Nat.lcm 18 60

end smallest_group_size_l3057_305738


namespace walker_children_puzzle_l3057_305726

def is_aabb (n : ℕ) : Prop :=
  ∃ a b : ℕ, n = a * 1000 + a * 100 + b * 10 + b ∧ 0 < a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9

def divisible_by_nine_out_of_ten (n : ℕ) : Prop :=
  ∃ k : ℕ, k ∈ (Finset.range 10).filter (λ i => i ≠ 0) ∧
    ∀ i ∈ (Finset.range 10).filter (λ i => i ≠ 0), i ≠ k → n % i = 0

theorem walker_children_puzzle :
  ∀ n : ℕ, is_aabb n → divisible_by_nine_out_of_ten n →
    ∃ (x y : ℕ), x + y = n → 
      ∃ k : ℕ, k ∈ (Finset.range 10).filter (λ i => i ≠ 0) ∧ n % k ≠ 0 ∧ k = 9 :=
sorry

end walker_children_puzzle_l3057_305726


namespace solve_y_equation_l3057_305773

theorem solve_y_equation : ∃ y : ℚ, (3 * y) / 7 = 21 ∧ y = 49 := by
  sorry

end solve_y_equation_l3057_305773


namespace sqrt_sum_equation_solutions_l3057_305775

theorem sqrt_sum_equation_solutions :
  ∀ x : ℝ, (Real.sqrt ((3 + Real.sqrt 5) ^ x) + Real.sqrt ((3 - Real.sqrt 5) ^ x) = 6) ↔ (x = 2 ∨ x = -2) :=
by sorry

end sqrt_sum_equation_solutions_l3057_305775


namespace max_sum_of_roots_l3057_305783

theorem max_sum_of_roots (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) 
  (hsum : x + y + z = 3) : 
  Real.sqrt (2 * x + 1) + Real.sqrt (2 * y + 1) + Real.sqrt (2 * z + 1) ≤ 3 * Real.sqrt 3 ∧
  (Real.sqrt (2 * x + 1) + Real.sqrt (2 * y + 1) + Real.sqrt (2 * z + 1) = 3 * Real.sqrt 3 ↔ 
    x = 1 ∧ y = 1 ∧ z = 1) :=
by sorry

end max_sum_of_roots_l3057_305783


namespace divisible_by_thirty_l3057_305720

theorem divisible_by_thirty (a b : ℤ) : 
  30 ∣ (a * b * (a^4 - b^4)) := by sorry

end divisible_by_thirty_l3057_305720


namespace max_value_expression_l3057_305712

theorem max_value_expression (x y : ℝ) :
  (2 * x + 3 * y + 4) / Real.sqrt (x^2 + y^2 + 4) ≤ Real.sqrt 29 := by
  sorry

end max_value_expression_l3057_305712


namespace prop_A_prop_B_prop_C_false_prop_D_main_theorem_l3057_305760

-- Define the basic concepts
def Line : Type := sorry
def Plane : Type := sorry
def Point : Type := sorry

-- Define the relationships
def perpendicular (l1 l2 : Line) : Prop := sorry
def parallel (l1 l2 : Line) : Prop := sorry
def intersect (l : Line) (p : Plane) : Prop := sorry
def onPlane (p : Point) (pl : Plane) : Prop := sorry
def angle (l1 l2 : Line) : ℝ := sorry
def collinear (p1 p2 p3 : Point) : Prop := sorry

-- Proposition A
theorem prop_A (l1 l2 l3 : Line) :
  parallel l1 l2 → perpendicular l3 l1 → perpendicular l3 l2 := sorry

-- Proposition B
theorem prop_B (a b c : Line) (θ : ℝ) :
  parallel a b → ¬(intersect c a) → ¬(intersect c b) → angle c a = θ → angle c b = θ := sorry

-- Proposition C (false statement)
theorem prop_C_false :
  ∃ (p1 p2 p3 p4 : Point) (pl : Plane), 
    ¬(onPlane p1 pl ∧ onPlane p2 pl ∧ onPlane p3 pl ∧ onPlane p4 pl) →
    ¬(collinear p1 p2 p3 ∨ collinear p1 p2 p4 ∨ collinear p1 p3 p4 ∨ collinear p2 p3 p4) := sorry

-- Proposition D
theorem prop_D (a : Line) (α : Plane) (P : Point) :
  parallel a α → onPlane P α → ∃ (l : Line), parallel l a ∧ onPlane P l ∧ (∀ (Q : Point), onPlane Q l → onPlane Q α) := sorry

-- Main theorem stating that A, B, and D are true while C is false
theorem main_theorem : 
  (∀ l1 l2 l3, parallel l1 l2 → perpendicular l3 l1 → perpendicular l3 l2) ∧
  (∀ a b c θ, parallel a b → ¬(intersect c a) → ¬(intersect c b) → angle c a = θ → angle c b = θ) ∧
  (∃ p1 p2 p3 p4 pl, ¬(onPlane p1 pl ∧ onPlane p2 pl ∧ onPlane p3 pl ∧ onPlane p4 pl) ∧
    (collinear p1 p2 p3 ∨ collinear p1 p2 p4 ∨ collinear p1 p3 p4 ∨ collinear p2 p3 p4)) ∧
  (∀ a α P, parallel a α → onPlane P α → 
    ∃ l, parallel l a ∧ onPlane P l ∧ (∀ Q, onPlane Q l → onPlane Q α)) := sorry

end prop_A_prop_B_prop_C_false_prop_D_main_theorem_l3057_305760


namespace jack_bake_sale_goal_l3057_305744

/-- The price of a brownie that allows Jack to reach his sales goal -/
def brownie_price : ℚ := by sorry

theorem jack_bake_sale_goal (num_brownies : ℕ) (num_lemon_squares : ℕ) (lemon_square_price : ℚ)
  (num_cookies : ℕ) (cookie_price : ℚ) (total_goal : ℚ) :
  num_brownies = 4 →
  num_lemon_squares = 5 →
  lemon_square_price = 2 →
  num_cookies = 7 →
  cookie_price = 4 →
  total_goal = 50 →
  num_brownies * brownie_price + num_lemon_squares * lemon_square_price + num_cookies * cookie_price = total_goal →
  brownie_price = 3 := by sorry

end jack_bake_sale_goal_l3057_305744


namespace cone_height_for_given_volume_and_angle_l3057_305705

/-- Represents a cone with given volume and vertex angle -/
structure Cone where
  volume : ℝ
  vertexAngle : ℝ

/-- Calculates the height of a cone given its volume and vertex angle -/
def coneHeight (c : Cone) : ℝ :=
  sorry

/-- Theorem stating that a cone with volume 19683π and vertex angle 90° has height 39 -/
theorem cone_height_for_given_volume_and_angle :
  let c : Cone := { volume := 19683 * Real.pi, vertexAngle := 90 }
  coneHeight c = 39 := by
  sorry

end cone_height_for_given_volume_and_angle_l3057_305705


namespace initial_money_calculation_l3057_305784

theorem initial_money_calculation (remaining_money : ℝ) (spent_percentage : ℝ) 
  (h1 : remaining_money = 840)
  (h2 : spent_percentage = 0.3)
  : (remaining_money / (1 - spent_percentage)) = 1200 := by
  sorry

end initial_money_calculation_l3057_305784


namespace plywood_area_l3057_305731

theorem plywood_area (width length area : ℝ) :
  width = 6 →
  length = 4 →
  area = width * length →
  area = 24 :=
by sorry

end plywood_area_l3057_305731


namespace max_gcd_sum_780_l3057_305752

theorem max_gcd_sum_780 :
  ∃ (a b : ℕ+), a + b = 780 ∧ 
  ∀ (c d : ℕ+), c + d = 780 → Nat.gcd c d ≤ Nat.gcd a b ∧
  Nat.gcd a b = 390 :=
by sorry

end max_gcd_sum_780_l3057_305752


namespace square_difference_l3057_305790

theorem square_difference (x y : ℝ) (h1 : x + y = 8) (h2 : x - y = 2) : x^2 - y^2 = 16 := by
  sorry

end square_difference_l3057_305790


namespace parabola_axis_of_symmetry_l3057_305741

/-- Given a parabola y = -(x+2)^2 - 3, its axis of symmetry is the line x = -2 -/
theorem parabola_axis_of_symmetry :
  let f : ℝ → ℝ := λ x => -(x + 2)^2 - 3
  ∃! a : ℝ, ∀ x y : ℝ, f x = f y → (x + y) / 2 = a :=
by sorry

end parabola_axis_of_symmetry_l3057_305741


namespace radio_cost_price_l3057_305795

theorem radio_cost_price (selling_price : ℝ) (loss_percentage : ℝ) (cost_price : ℝ) : 
  selling_price = 1430 →
  loss_percentage = 20.555555555555554 →
  selling_price = cost_price * (1 - loss_percentage / 100) →
  cost_price = 1800 := by
sorry

end radio_cost_price_l3057_305795


namespace simple_interest_calculation_l3057_305746

/-- Calculate simple interest given principal, rate, and time -/
def simple_interest (principal : ℚ) (rate : ℚ) (time : ℚ) : ℚ :=
  principal * rate * time / 100

/-- Calculate total sum after simple interest -/
def total_sum (principal : ℚ) (rate : ℚ) (time : ℚ) : ℚ :=
  principal + simple_interest principal rate time

theorem simple_interest_calculation (P : ℚ) :
  total_sum P (5 : ℚ) (5 : ℚ) = 16065 →
  simple_interest P (5 : ℚ) (5 : ℚ) = 3213 := by
  sorry

end simple_interest_calculation_l3057_305746


namespace mans_age_twice_sons_age_l3057_305787

/-- Represents the current age of the son -/
def sonAge : ℕ := 22

/-- Represents the age difference between the man and his son -/
def ageDifference : ℕ := 24

/-- Represents the number of years until the man's age is twice his son's age -/
def yearsUntilTwice : ℕ := 2

/-- Theorem stating that in 'yearsUntilTwice' years, the man's age will be twice his son's age -/
theorem mans_age_twice_sons_age :
  (sonAge + ageDifference + yearsUntilTwice) = 2 * (sonAge + yearsUntilTwice) := by
  sorry

end mans_age_twice_sons_age_l3057_305787


namespace fifteen_tomorrow_fishers_l3057_305771

/-- Represents the fishing schedule in the coastal village -/
structure FishingSchedule where
  daily : Nat
  everyOtherDay : Nat
  everyThreeDay : Nat
  yesterday : Nat
  today : Nat

/-- Calculates the number of people fishing tomorrow based on the given schedule -/
def tomorrowFishers (schedule : FishingSchedule) : Nat :=
  sorry

/-- Theorem stating that given the specific fishing schedule, 15 people will fish tomorrow -/
theorem fifteen_tomorrow_fishers (schedule : FishingSchedule) 
  (h1 : schedule.daily = 7)
  (h2 : schedule.everyOtherDay = 8)
  (h3 : schedule.everyThreeDay = 3)
  (h4 : schedule.yesterday = 12)
  (h5 : schedule.today = 10) :
  tomorrowFishers schedule = 15 := by
  sorry

end fifteen_tomorrow_fishers_l3057_305771


namespace trisection_intersection_x_coordinate_l3057_305715

theorem trisection_intersection_x_coordinate : 
  let f : ℝ → ℝ := λ x => Real.log x
  let x₁ : ℝ := 2
  let x₂ : ℝ := 500
  let y₁ : ℝ := f x₁
  let y₂ : ℝ := f x₂
  let yC : ℝ := (2/3) * y₁ + (1/3) * y₂
  ∃ x₃ : ℝ, f x₃ = yC ∧ x₃ = 10 * (2^(2/3)) * (5^(1/3)) :=
by sorry

end trisection_intersection_x_coordinate_l3057_305715


namespace graces_age_l3057_305713

/-- Grace's age problem -/
theorem graces_age (mother_age : ℕ) (grandmother_age : ℕ) (grace_age : ℕ) :
  mother_age = 80 →
  grandmother_age = 2 * mother_age →
  grace_age = (3 * grandmother_age) / 8 →
  grace_age = 60 := by
  sorry

end graces_age_l3057_305713


namespace average_bag_weight_l3057_305780

def bag_weights : List ℕ := [25, 30, 31, 32, 34, 35, 37, 39, 40, 41, 42, 44, 45, 48]

theorem average_bag_weight :
  (bag_weights.sum : ℚ) / bag_weights.length = 71/2 := by sorry

end average_bag_weight_l3057_305780


namespace exists_composite_carmichael_number_l3057_305702

theorem exists_composite_carmichael_number : ∃ n : ℕ, 
  n > 1 ∧ 
  ¬ Nat.Prime n ∧ 
  ∀ a : ℤ, (n : ℤ) ∣ (a^n - a) := by
  sorry

end exists_composite_carmichael_number_l3057_305702


namespace complex_modulus_problem_l3057_305722

theorem complex_modulus_problem (z : ℂ) (h : (1 - I) * z = 1 + I) : Complex.abs z = 1 := by
  sorry

end complex_modulus_problem_l3057_305722


namespace intersection_equals_B_l3057_305723

/-- The set A of solutions to x^2 - 4x + 3 = 0 -/
def A : Set ℝ := {x | x^2 - 4*x + 3 = 0}

/-- The set B of solutions to mx + 1 = 0 for some real m -/
def B (m : ℝ) : Set ℝ := {x | m*x + 1 = 0}

/-- The theorem stating the set of values for m that satisfy A ∩ B = B -/
theorem intersection_equals_B : 
  {m : ℝ | A ∩ B m = B m} = {-1, -1/3, 0} := by sorry

end intersection_equals_B_l3057_305723


namespace arithmetic_sequence_sum_l3057_305791

def arithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∀ n m : ℕ, a (n + 1) - a n = a (m + 1) - a m

theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  arithmeticSequence a →
  a 1 + a 2017 = 10 →
  a 1 * a 2017 = 16 →
  a 2 + a 1009 + a 2016 = 15 := by
  sorry

end arithmetic_sequence_sum_l3057_305791


namespace last_digit_product_divisible_by_three_l3057_305798

theorem last_digit_product_divisible_by_three (n : ℕ) :
  let a := (2^n % 10)
  ∃ k : ℤ, a * (2^n - a) = 3 * k :=
sorry

end last_digit_product_divisible_by_three_l3057_305798


namespace geometric_series_first_term_l3057_305766

theorem geometric_series_first_term 
  (r : ℝ) 
  (S : ℝ) 
  (h1 : r = 1/5)
  (h2 : S = 100)
  (h3 : S = a / (1 - r)) :
  a = 80 := by
sorry

end geometric_series_first_term_l3057_305766


namespace marks_of_a_l3057_305796

theorem marks_of_a (a b c d e : ℝ) : 
  (a + b + c) / 3 = 48 →
  (a + b + c + d) / 4 = 47 →
  e = d + 3 →
  (b + c + d + e) / 4 = 48 →
  a = 43 := by
sorry

end marks_of_a_l3057_305796


namespace sum_of_roots_zero_l3057_305776

def is_quadratic (Q : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, ∀ x, Q x = a * x^2 + b * x + c

theorem sum_of_roots_zero (Q : ℝ → ℝ) 
  (h_quad : is_quadratic Q)
  (h_ineq : ∀ x : ℝ, Q (x^3 - x) ≥ Q (x^2 - 1)) :
  ∃ r₁ r₂ : ℝ, (∀ x, Q x = 0 ↔ x = r₁ ∨ x = r₂) ∧ r₁ + r₂ = 0 :=
sorry

end sum_of_roots_zero_l3057_305776


namespace rectangular_park_area_l3057_305789

/-- A rectangular park with a perimeter of 80 feet and length three times its width has an area of 300 square feet. -/
theorem rectangular_park_area : ∀ l w : ℝ,
  l > 0 → w > 0 →  -- Ensure positive dimensions
  2 * (l + w) = 80 →  -- Perimeter condition
  l = 3 * w →  -- Length is three times the width
  l * w = 300 := by
sorry

end rectangular_park_area_l3057_305789


namespace even_function_positive_x_l3057_305763

def is_even_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

theorem even_function_positive_x 
  (f : ℝ → ℝ) 
  (h_even : is_even_function f) 
  (h_neg : ∀ x < 0, f x = x * (x - 1)) : 
  ∀ x > 0, f x = x * (x + 1) := by
sorry

end even_function_positive_x_l3057_305763


namespace inequality_solution_set_l3057_305708

theorem inequality_solution_set :
  {x : ℝ | x^2 + 2*x - 3 ≥ 0} = {x : ℝ | x ≤ -3 ∨ x ≥ 1} := by
  sorry

end inequality_solution_set_l3057_305708


namespace parallel_resistors_combined_resistance_l3057_305786

theorem parallel_resistors_combined_resistance :
  let r1 : ℚ := 2
  let r2 : ℚ := 5
  let r3 : ℚ := 6
  let r : ℚ := (1 / r1 + 1 / r2 + 1 / r3)⁻¹
  r = 15 / 13 := by sorry

end parallel_resistors_combined_resistance_l3057_305786


namespace triangle_theorem_l3057_305739

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    if a*sin(A) = 4b*sin(B) and a*c = √5*(a^2 - b^2 - c^2),
    then cos(A) = -√5/5 and sin(2B - A) = -2√5/5 -/
theorem triangle_theorem (a b c A B C : ℝ) 
  (h1 : a * Real.sin A = 4 * b * Real.sin B)
  (h2 : a * c = Real.sqrt 5 * (a^2 - b^2 - c^2)) :
  Real.cos A = -(Real.sqrt 5 / 5) ∧ 
  Real.sin (2 * B - A) = -(2 * Real.sqrt 5 / 5) := by
  sorry

end triangle_theorem_l3057_305739


namespace inverse_113_mod_114_l3057_305797

theorem inverse_113_mod_114 : ∃ x : ℕ, x ∈ Finset.range 114 ∧ (113 * x) % 114 = 1 :=
by
  -- The proof goes here
  sorry

end inverse_113_mod_114_l3057_305797


namespace ruby_candies_l3057_305718

/-- The number of friends Ruby shares her candies with -/
def num_friends : ℕ := 9

/-- The number of candies each friend receives -/
def candies_per_friend : ℕ := 4

/-- The initial number of candies Ruby has -/
def initial_candies : ℕ := num_friends * candies_per_friend

theorem ruby_candies : initial_candies = 36 := by
  sorry

end ruby_candies_l3057_305718


namespace repeating_decimal_47_equals_fraction_sum_of_numerator_and_denominator_l3057_305793

/-- Represents a repeating decimal with a two-digit repetend -/
def RepeatingDecimal (a b : ℕ) : ℚ := (10 * a + b : ℚ) / 99

theorem repeating_decimal_47_equals_fraction :
  RepeatingDecimal 4 7 = 47 / 99 :=
sorry

theorem sum_of_numerator_and_denominator :
  (47 : ℕ) + 99 = 146 :=
sorry

end repeating_decimal_47_equals_fraction_sum_of_numerator_and_denominator_l3057_305793


namespace binary_11011_equals_27_l3057_305764

def binary_to_decimal (b : List Bool) : Nat :=
  b.enum.foldl (fun acc (i, bit) => acc + if bit then 2^i else 0) 0

theorem binary_11011_equals_27 :
  binary_to_decimal [true, true, false, true, true] = 27 := by
  sorry

end binary_11011_equals_27_l3057_305764


namespace min_additional_coins_alex_coin_distribution_l3057_305735

theorem min_additional_coins (num_friends : ℕ) (initial_coins : ℕ) : ℕ :=
  let min_required := (num_friends * (num_friends + 1)) / 2
  if min_required > initial_coins then
    min_required - initial_coins
  else
    0

theorem alex_coin_distribution : min_additional_coins 15 90 = 30 := by
  sorry

end min_additional_coins_alex_coin_distribution_l3057_305735


namespace simplify_expression_l3057_305753

theorem simplify_expression (x y : ℝ) :
  3 * (x + y)^2 - 7 * (x + y) + 8 * (x + y)^2 + 6 * (x + y) = 11 * (x + y)^2 - (x + y) := by
  sorry

end simplify_expression_l3057_305753


namespace logarithm_expression_equality_l3057_305707

theorem logarithm_expression_equality : 
  (Real.log 243 / Real.log 3) / (Real.log 3 / Real.log 81) - 
  (Real.log 729 / Real.log 3) / (Real.log 3 / Real.log 27) = 2 := by
  sorry

end logarithm_expression_equality_l3057_305707


namespace gcd_power_two_minus_one_l3057_305716

theorem gcd_power_two_minus_one : 
  Nat.gcd (2^2000 - 1) (2^1990 - 1) = 2^10 - 1 := by sorry

end gcd_power_two_minus_one_l3057_305716


namespace thomas_savings_years_l3057_305734

/-- Represents the savings scenario for Thomas --/
structure SavingsScenario where
  allowance : ℕ  -- Weekly allowance in the first year
  wage : ℕ       -- Hourly wage from the second year
  hours : ℕ      -- Weekly work hours from the second year
  carCost : ℕ    -- Cost of the car
  spending : ℕ   -- Weekly spending
  remaining : ℕ  -- Amount still needed to buy the car

/-- Calculates the number of years Thomas has been saving --/
def yearsOfSaving (s : SavingsScenario) : ℕ :=
  2  -- This is the value we want to prove

/-- Theorem stating that Thomas has been saving for 2 years --/
theorem thomas_savings_years (s : SavingsScenario) 
  (h1 : s.allowance = 50)
  (h2 : s.wage = 9)
  (h3 : s.hours = 30)
  (h4 : s.carCost = 15000)
  (h5 : s.spending = 35)
  (h6 : s.remaining = 2000) :
  yearsOfSaving s = 2 := by
  sorry

#check thomas_savings_years

end thomas_savings_years_l3057_305734


namespace locus_is_circle_l3057_305730

/-- Given a right triangle with sides s, s, and s√2, this function represents the locus of points P 
    such that the sum of squares of distances from P to the vertices equals a. -/
def locus (s a : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | ∃ (A B C : ℝ × ℝ), 
    -- A, B, C form a right triangle with sides s, s, s√2
    (B.1 - A.1)^2 + (B.2 - A.2)^2 = s^2 ∧
    (C.1 - A.1)^2 + (C.2 - A.2)^2 = s^2 ∧
    (C.1 - B.1)^2 + (C.2 - B.2)^2 = 2*s^2 ∧
    -- Sum of squares of distances from P to vertices equals a
    (p.1 - A.1)^2 + (p.2 - A.2)^2 + 
    (p.1 - B.1)^2 + (p.2 - B.2)^2 + 
    (p.1 - C.1)^2 + (p.2 - C.2)^2 = a}

/-- The constant K dependent on the triangle's dimensions -/
def K (s : ℝ) : ℝ := 2 * s^2

/-- Theorem stating that the locus is a circle if and only if a > K -/
theorem locus_is_circle (s a : ℝ) (h_s : s > 0) : 
  ∃ (center : ℝ × ℝ) (radius : ℝ), locus s a = Metric.ball center radius ↔ a > K s :=
sorry

end locus_is_circle_l3057_305730


namespace max_int_diff_l3057_305767

theorem max_int_diff (x y : ℤ) (hx : 6 < x ∧ x < 10) (hy : 10 < y ∧ y < 17) :
  (∀ a b : ℤ, 6 < a ∧ a < 10 ∧ 10 < b ∧ b < 17 → y - x ≥ b - a) ∧ y - x = 7 :=
sorry

end max_int_diff_l3057_305767


namespace jason_seashell_count_l3057_305703

def seashell_count (initial : ℕ) (given_tim : ℕ) (given_lily : ℕ) (found : ℕ) (lost : ℕ) : ℕ :=
  initial - given_tim - given_lily + found - lost

theorem jason_seashell_count : 
  seashell_count 49 13 7 15 5 = 39 := by sorry

end jason_seashell_count_l3057_305703


namespace quadratic_roots_condition_l3057_305706

theorem quadratic_roots_condition (a : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ a * x^2 - 4*x + 3 = 0 ∧ a * y^2 - 4*y + 3 = 0) ↔ 
  (a < 4/3 ∧ a ≠ 0) :=
sorry

end quadratic_roots_condition_l3057_305706


namespace special_function_at_50_l3057_305769

/-- A function satisfying f(xy) = xf(y) for all real x and y, and f(1) = 10 -/
def special_function (f : ℝ → ℝ) : Prop :=
  (∀ x y : ℝ, f (x * y) = x * f y) ∧ (f 1 = 10)

/-- Theorem: If f is a special function, then f(50) = 500 -/
theorem special_function_at_50 (f : ℝ → ℝ) (h : special_function f) : f 50 = 500 := by
  sorry

end special_function_at_50_l3057_305769


namespace inscribed_circle_radius_l3057_305740

/-- The radius of the inscribed circle in a triangle with side lengths 8, 10, and 14 is √6. -/
theorem inscribed_circle_radius (DE DF EF : ℝ) (h1 : DE = 8) (h2 : DF = 10) (h3 : EF = 14) :
  let s := (DE + DF + EF) / 2
  let K := Real.sqrt (s * (s - DE) * (s - DF) * (s - EF))
  K / s = Real.sqrt 6 := by sorry

end inscribed_circle_radius_l3057_305740


namespace club_officer_selection_l3057_305728

/-- The number of ways to choose officers of the same gender from a club -/
def choose_officers (total_members : ℕ) (boys : ℕ) (girls : ℕ) : ℕ :=
  2 * (boys * (boys - 1) * (boys - 2))

/-- Theorem: Choosing officers from a club with specific conditions -/
theorem club_officer_selection :
  let total_members : ℕ := 30
  let boys : ℕ := 15
  let girls : ℕ := 15
  choose_officers total_members boys girls = 5460 := by
  sorry

end club_officer_selection_l3057_305728


namespace area_of_triangle_DCE_l3057_305777

/-- Given a rectangle BDEF with AB = 24 and EF = 15, and triangle BCE with area 60,
    prove that the area of triangle DCE is 30 -/
theorem area_of_triangle_DCE (AB EF : ℝ) (area_BCE : ℝ) :
  AB = 24 →
  EF = 15 →
  area_BCE = 60 →
  let BC := (2 * area_BCE) / EF
  let DC := EF - BC
  let DE := (2 * area_BCE) / BC
  (1/2) * DC * DE = 30 := by
  sorry

end area_of_triangle_DCE_l3057_305777


namespace equation_solution_l3057_305778

theorem equation_solution : ∃! x : ℚ, (x^2 + 3*x + 5) / (x + 6) = x + 7 := by
  sorry

end equation_solution_l3057_305778


namespace parabola_y_range_l3057_305761

-- Define the parabola
def parabola (x y : ℝ) : Prop := x^2 = 8*y

-- Define the focus-to-point distance
def focus_distance (y : ℝ) : ℝ := y + 2

-- Define the condition for intersection with directrix
def intersects_directrix (y : ℝ) : Prop := focus_distance y > 4

theorem parabola_y_range (x y : ℝ) :
  parabola x y → intersects_directrix y → y > 2 := by
  sorry

end parabola_y_range_l3057_305761


namespace geometric_sequence_a8_l3057_305711

def is_geometric_sequence (a : ℕ+ → ℚ) : Prop :=
  ∃ q : ℚ, ∀ n : ℕ+, a (n + 1) = a n * q

theorem geometric_sequence_a8 (a : ℕ+ → ℚ) :
  is_geometric_sequence a →
  a 2 = 1 / 16 →
  a 5 = 1 / 2 →
  a 8 = 4 := by
sorry

end geometric_sequence_a8_l3057_305711


namespace equation_and_inequality_system_l3057_305757

theorem equation_and_inequality_system :
  -- Part 1: Equation
  (let equation := fun x : ℝ => 2 * x * (x - 2) = 1
   let solution1 := (2 + Real.sqrt 6) / 2
   let solution2 := (2 - Real.sqrt 6) / 2
   equation solution1 ∧ equation solution2) ∧
  -- Part 2: Inequality system
  (let inequality1 := fun x : ℝ => 2 * x + 3 > 1
   let inequality2 := fun x : ℝ => x - 2 ≤ (1 / 2) * (x + 2)
   ∀ x : ℝ, (inequality1 x ∧ inequality2 x) ↔ (-1 < x ∧ x ≤ 6)) := by
  sorry

end equation_and_inequality_system_l3057_305757


namespace total_pages_in_book_l3057_305747

/-- The number of pages Suzanne read on Monday -/
def monday_pages : ℝ := 15.5

/-- The number of pages Suzanne read on Tuesday -/
def tuesday_pages : ℝ := 1.5 * monday_pages + 16

/-- The total number of pages Suzanne read in two days -/
def total_pages_read : ℝ := monday_pages + tuesday_pages

/-- The theorem stating the total number of pages in the book -/
theorem total_pages_in_book : total_pages_read * 2 = 109.5 := by sorry

end total_pages_in_book_l3057_305747


namespace complex_power_magnitude_l3057_305770

theorem complex_power_magnitude : 
  Complex.abs ((2/3 : ℂ) + (5/6 : ℂ) * Complex.I)^8 = 2825761/1679616 := by
sorry

end complex_power_magnitude_l3057_305770


namespace solution_set_and_range_l3057_305751

def f (x : ℝ) : ℝ := |2 * x - 1| + 1

theorem solution_set_and_range :
  (∀ x : ℝ, f x ≤ 6 ↔ -2 ≤ x ∧ x ≤ 3) ∧
  (∀ m : ℝ, (∃ n : ℝ, f n ≤ m - f (-n)) ↔ 4 ≤ m) := by sorry

end solution_set_and_range_l3057_305751


namespace largest_power_of_two_dividing_n_l3057_305727

def n : ℕ := 15^4 - 9^4

theorem largest_power_of_two_dividing_n : 
  ∃ k : ℕ, k = 5 ∧ 2^k ∣ n ∧ ∀ m : ℕ, 2^m ∣ n → m ≤ k :=
sorry

end largest_power_of_two_dividing_n_l3057_305727


namespace number_of_divisors_l3057_305788

-- Define the number we're working with
def n : ℕ := 3465

-- Define the prime factorization of n
axiom prime_factorization : n = 3^2 * 5^1 * 7^2

-- Define the function to count positive divisors
def count_divisors (m : ℕ) : ℕ := sorry

-- Theorem stating the number of positive divisors of n
theorem number_of_divisors : count_divisors n = 18 := by sorry

end number_of_divisors_l3057_305788


namespace repeating_decimal_equals_fraction_l3057_305754

/-- The repeating decimal 0.37̄246 expressed as a rational number -/
def repeating_decimal : ℚ := 37246 / 99900

theorem repeating_decimal_equals_fraction : 
  repeating_decimal = 371874 / 99900 := by sorry

end repeating_decimal_equals_fraction_l3057_305754


namespace number_of_shooting_orders_l3057_305704

/-- Represents the number of targets in each column -/
def targets_per_column : Fin 3 → ℕ
  | 0 => 4  -- Column A
  | 1 => 3  -- Column B
  | 2 => 3  -- Column C

/-- The total number of targets -/
def total_targets : ℕ := 10

/-- The number of initial shooting sequences -/
def initial_sequences : ℕ := 2

/-- Calculates the number of permutations for the remaining shots -/
def remaining_permutations : ℕ :=
  Nat.factorial 8 / (Nat.factorial 3 * Nat.factorial 2 * Nat.factorial 3)

/-- Theorem stating the total number of different orders to break the targets -/
theorem number_of_shooting_orders :
  initial_sequences * remaining_permutations = 1120 := by sorry

end number_of_shooting_orders_l3057_305704


namespace smallest_interesting_number_l3057_305748

/-- A natural number is interesting if 2n is a perfect square and 15n is a perfect cube. -/
def is_interesting (n : ℕ) : Prop :=
  ∃ (a b : ℕ), 2 * n = a ^ 2 ∧ 15 * n = b ^ 3

/-- 1800 is the smallest interesting number. -/
theorem smallest_interesting_number : 
  is_interesting 1800 ∧ ∀ m < 1800, ¬is_interesting m :=
by sorry

end smallest_interesting_number_l3057_305748


namespace monomial_count_l3057_305719

-- Define a type for algebraic expressions
inductive AlgebraicExpr
  | Constant (c : ℚ)
  | Variable (v : String)
  | Product (e1 e2 : AlgebraicExpr)
  | Sum (e1 e2 : AlgebraicExpr)
  | Fraction (num den : AlgebraicExpr)

-- Define what a monomial is
def isMonomial : AlgebraicExpr → Bool
  | AlgebraicExpr.Constant _ => true
  | AlgebraicExpr.Variable _ => true
  | AlgebraicExpr.Product e1 e2 => isMonomial e1 && isMonomial e2
  | _ => false

-- Define the list of given expressions
def givenExpressions : List AlgebraicExpr := [
  AlgebraicExpr.Product (AlgebraicExpr.Constant (-1/2)) (AlgebraicExpr.Product (AlgebraicExpr.Variable "m") (AlgebraicExpr.Variable "n")),
  AlgebraicExpr.Variable "m",
  AlgebraicExpr.Constant (1/2),
  AlgebraicExpr.Fraction (AlgebraicExpr.Variable "b") (AlgebraicExpr.Variable "a"),
  AlgebraicExpr.Sum (AlgebraicExpr.Product (AlgebraicExpr.Constant 2) (AlgebraicExpr.Variable "m")) (AlgebraicExpr.Constant 1),
  AlgebraicExpr.Fraction (AlgebraicExpr.Sum (AlgebraicExpr.Variable "x") (AlgebraicExpr.Product (AlgebraicExpr.Constant (-1)) (AlgebraicExpr.Variable "y"))) (AlgebraicExpr.Constant 5),
  AlgebraicExpr.Fraction 
    (AlgebraicExpr.Sum (AlgebraicExpr.Product (AlgebraicExpr.Constant 2) (AlgebraicExpr.Variable "x")) (AlgebraicExpr.Variable "y"))
    (AlgebraicExpr.Sum (AlgebraicExpr.Variable "x") (AlgebraicExpr.Product (AlgebraicExpr.Constant (-1)) (AlgebraicExpr.Variable "y"))),
  AlgebraicExpr.Sum 
    (AlgebraicExpr.Sum 
      (AlgebraicExpr.Product (AlgebraicExpr.Variable "x") (AlgebraicExpr.Variable "x")) 
      (AlgebraicExpr.Product (AlgebraicExpr.Constant 2) (AlgebraicExpr.Variable "x")))
    (AlgebraicExpr.Constant (3/2))
]

-- Theorem statement
theorem monomial_count : 
  (givenExpressions.filter isMonomial).length = 3 := by sorry

end monomial_count_l3057_305719


namespace intersection_implies_a_value_l3057_305724

theorem intersection_implies_a_value (a : ℝ) : 
  let A : Set ℝ := {-1, 2, 3}
  let B : Set ℝ := {a + 2, a^2 + 2}
  A ∩ B = {3} → a = -1 := by
sorry

end intersection_implies_a_value_l3057_305724


namespace election_winner_percentage_l3057_305700

theorem election_winner_percentage (total_votes winner_votes margin : ℕ) : 
  winner_votes = 1044 →
  margin = 288 →
  total_votes = winner_votes + (winner_votes - margin) →
  (winner_votes : ℚ) / (total_votes : ℚ) = 58 / 100 := by
sorry

end election_winner_percentage_l3057_305700


namespace greatest_possible_N_l3057_305759

theorem greatest_possible_N : ∃ (N : ℕ), 
  (N = 5) ∧ 
  (∀ k : ℕ, k > 5 → ¬∃ (S : Finset ℕ), 
    (Finset.card S = 2^k - 1) ∧ 
    (∀ x y : ℕ, x ∈ S → y ∈ S → x ≠ y → x ≠ y) ∧
    (Finset.sum S id = 2014)) ∧
  (∃ (S : Finset ℕ), 
    (Finset.card S = 2^5 - 1) ∧ 
    (∀ x y : ℕ, x ∈ S → y ∈ S → x ≠ y → x ≠ y) ∧
    (Finset.sum S id = 2014)) :=
by sorry

end greatest_possible_N_l3057_305759


namespace integer_representation_l3057_305749

theorem integer_representation (n : ℤ) : ∃ (x y z : ℕ+), (n : ℤ) = x^2 + y^2 - z^2 := by
  sorry

end integer_representation_l3057_305749


namespace equation_solution_l3057_305758

theorem equation_solution :
  ∃ x : ℚ, 5 * (x - 10) = 6 * (3 - 3 * x) + 10 ∧ x = 78 / 23 := by
  sorry

end equation_solution_l3057_305758


namespace optimal_selling_price_l3057_305709

/-- Represents the selling price of grapes in yuan per kilogram -/
def selling_price : ℝ := 21

/-- Represents the cost price of grapes in yuan per kilogram -/
def cost_price : ℝ := 16

/-- Represents the daily sales volume in kilograms when the price is 26 yuan -/
def base_sales : ℝ := 320

/-- Represents the increase in sales volume for each yuan decrease in price -/
def sales_increase_rate : ℝ := 80

/-- Represents the target daily profit in yuan -/
def target_profit : ℝ := 3600

/-- Calculates the daily sales volume based on the selling price -/
def sales_volume (x : ℝ) : ℝ := base_sales + sales_increase_rate * (26 - x)

/-- Calculates the daily profit based on the selling price -/
def daily_profit (x : ℝ) : ℝ := (x - cost_price) * sales_volume x

/-- Theorem stating that the chosen selling price satisfies the profit goal and is optimal -/
theorem optimal_selling_price : 
  daily_profit selling_price = target_profit ∧ 
  (∀ y, y < selling_price → daily_profit y < target_profit) :=
sorry

end optimal_selling_price_l3057_305709


namespace polynomial_maximum_l3057_305779

/-- The polynomial function we're analyzing -/
def f (x : ℝ) : ℝ := -3 * x^2 - 6 * x + 12

/-- The maximum value of the polynomial -/
def max_value : ℝ := 15

/-- The x-value at which the maximum occurs -/
def max_point : ℝ := -1

theorem polynomial_maximum :
  (∀ x : ℝ, f x ≤ max_value) ∧ f max_point = max_value :=
sorry

end polynomial_maximum_l3057_305779


namespace other_communities_count_l3057_305701

theorem other_communities_count (total : ℕ) (muslim_percent hindu_percent sikh_percent : ℚ) : 
  total = 300 →
  muslim_percent = 44/100 →
  hindu_percent = 28/100 →
  sikh_percent = 10/100 →
  (total : ℚ) * (1 - (muslim_percent + hindu_percent + sikh_percent)) = 54 := by
sorry

end other_communities_count_l3057_305701


namespace rowing_round_trip_time_l3057_305772

/-- Calculates the total time for a round trip rowing journey given the rowing speed, current speed, and distance. -/
theorem rowing_round_trip_time 
  (rowing_speed : ℝ) 
  (current_speed : ℝ) 
  (distance : ℝ) 
  (h1 : rowing_speed = 10)
  (h2 : current_speed = 2)
  (h3 : distance = 24) : 
  (distance / (rowing_speed + current_speed)) + (distance / (rowing_speed - current_speed)) = 5 := by
  sorry

#check rowing_round_trip_time

end rowing_round_trip_time_l3057_305772


namespace max_product_sum_2000_l3057_305710

theorem max_product_sum_2000 :
  (∃ (x y : ℤ), x + y = 2000 ∧ x * y = 1000000) ∧
  (∀ (a b : ℤ), a + b = 2000 → a * b ≤ 1000000) := by
  sorry

end max_product_sum_2000_l3057_305710


namespace arithmetic_calculation_l3057_305729

theorem arithmetic_calculation : 10 - 9 * 8 + 7^2 - 6 / 3 * 2 + 1 = -16 := by
  sorry

end arithmetic_calculation_l3057_305729


namespace xy_max_value_l3057_305742

theorem xy_max_value (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 3 * x + 2 * y = 12) :
  x * y ≤ 6 := by
sorry

end xy_max_value_l3057_305742


namespace monotone_increasing_condition_l3057_305756

/-- The function f(x) = e^x - ln(x+m) is monotonically increasing on [0,1] iff m ≥ 1 -/
theorem monotone_increasing_condition (m : ℝ) :
  (∀ x ∈ Set.Icc 0 1, MonotoneOn (fun x => Real.exp x - Real.log (x + m)) (Set.Icc 0 1)) ↔ m ≥ 1 := by
  sorry

end monotone_increasing_condition_l3057_305756


namespace cubic_sum_over_product_l3057_305785

theorem cubic_sum_over_product (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) 
  (h_sum : x + y + z = 0) : 
  (x^3 + y^3 + z^3) / (x*y*z) = 3 := by
  sorry

end cubic_sum_over_product_l3057_305785


namespace distance_to_midpoint_l3057_305725

-- Define the triangle PQR
structure RightTriangle where
  PQ : ℝ
  PR : ℝ
  QR : ℝ
  is_right : PQ^2 = PR^2 + QR^2

-- Define the specific triangle given in the problem
def triangle_PQR : RightTriangle :=
  { PQ := 15
    PR := 9
    QR := 12
    is_right := by norm_num }

-- Theorem statement
theorem distance_to_midpoint (t : RightTriangle) (h : t = triangle_PQR) :
  (t.PQ / 2 : ℝ) = 7.5 := by
  sorry

end distance_to_midpoint_l3057_305725


namespace average_headcount_rounded_l3057_305794

def fall_headcount_03_04 : ℕ := 11500
def fall_headcount_04_05 : ℕ := 11300
def fall_headcount_05_06 : ℕ := 11400

def average_headcount : ℚ := (fall_headcount_03_04 + fall_headcount_04_05 + fall_headcount_05_06) / 3

theorem average_headcount_rounded : 
  round average_headcount = 11400 := by sorry

end average_headcount_rounded_l3057_305794


namespace towel_folding_theorem_l3057_305745

/-- Represents the number of towels a person can fold in a given time -/
structure FoldingRate where
  towels : ℕ
  minutes : ℕ

/-- Calculates the number of towels folded in one hour given a folding rate -/
def towelsPerHour (rate : FoldingRate) : ℕ :=
  (60 / rate.minutes) * rate.towels

/-- The total number of towels folded by all three people in one hour -/
def totalTowelsPerHour (jane kyla anthony : FoldingRate) : ℕ :=
  towelsPerHour jane + towelsPerHour kyla + towelsPerHour anthony

theorem towel_folding_theorem (jane kyla anthony : FoldingRate)
  (h1 : jane = ⟨3, 5⟩)
  (h2 : kyla = ⟨5, 10⟩)
  (h3 : anthony = ⟨7, 20⟩) :
  totalTowelsPerHour jane kyla anthony = 87 := by
  sorry

#eval totalTowelsPerHour ⟨3, 5⟩ ⟨5, 10⟩ ⟨7, 20⟩

end towel_folding_theorem_l3057_305745


namespace probability_greater_than_four_l3057_305743

-- Define a standard six-sided die
def standardDie : Finset Nat := Finset.range 6

-- Define the probability of an event on the die
def probability (event : Finset Nat) : Rat :=
  event.card / standardDie.card

-- Define the event of rolling a number greater than 4
def greaterThanFour : Finset Nat := Finset.filter (λ x => x > 4) standardDie

-- Theorem statement
theorem probability_greater_than_four :
  probability greaterThanFour = 1 / 3 := by
  sorry

end probability_greater_than_four_l3057_305743


namespace max_sum_of_factors_l3057_305762

theorem max_sum_of_factors (x y : ℕ+) (h : x * y = 48) : 
  ∃ (a b : ℕ+), a * b = 48 ∧ a + b ≤ x + y ∧ a + b = 49 :=
sorry

end max_sum_of_factors_l3057_305762


namespace same_color_probability_l3057_305799

/-- The number of white balls in the bag -/
def white_balls : ℕ := 3

/-- The number of red balls in the bag -/
def red_balls : ℕ := 2

/-- The total number of balls in the bag -/
def total_balls : ℕ := white_balls + red_balls

/-- The number of balls drawn from the bag -/
def drawn_balls : ℕ := 2

/-- Calculates the number of combinations of n items taken r at a time -/
def combinations (n r : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial r * Nat.factorial (n - r))

/-- The probability of drawing two balls of the same color -/
theorem same_color_probability : 
  (combinations white_balls drawn_balls + combinations red_balls drawn_balls) / 
  combinations total_balls drawn_balls = 2 / 5 := by
  sorry

end same_color_probability_l3057_305799


namespace exists_initial_points_for_82_l3057_305717

/-- The function that calculates the number of points after one application of the procedure -/
def points_after_one_step (n : ℕ) : ℕ := 3 * n - 2

/-- The function that calculates the number of points after two applications of the procedure -/
def points_after_two_steps (n : ℕ) : ℕ := 9 * n - 8

/-- Theorem stating that there exists an initial number of points that results in 82 points after two steps -/
theorem exists_initial_points_for_82 : ∃ n : ℕ, n > 0 ∧ points_after_two_steps n = 82 := by
  sorry

end exists_initial_points_for_82_l3057_305717
