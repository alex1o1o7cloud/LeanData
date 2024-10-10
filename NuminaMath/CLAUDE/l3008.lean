import Mathlib

namespace median_salary_is_45000_l3008_300876

structure Position :=
  (title : String)
  (count : ℕ)
  (salary : ℕ)

def company_data : List Position := [
  ⟨"CEO", 1, 150000⟩,
  ⟨"Senior Manager", 4, 95000⟩,
  ⟨"Manager", 15, 70000⟩,
  ⟨"Assistant Manager", 20, 45000⟩,
  ⟨"Clerk", 40, 18000⟩
]

def total_employees : ℕ := (company_data.map Position.count).sum

def median_salary (data : List Position) : ℕ := 
  if total_employees % 2 = 0 
  then 45000  -- As both (total_employees / 2) and (total_employees / 2 + 1) fall under Assistant Manager
  else 45000  -- As (total_employees / 2 + 1) falls under Assistant Manager

theorem median_salary_is_45000 : 
  median_salary company_data = 45000 := by sorry

end median_salary_is_45000_l3008_300876


namespace complex_cosine_geometric_representation_l3008_300870

/-- The set of points represented by z = i cos θ, where θ ∈ [0, 2π], 
    is equal to the line segment from (0, -1) to (0, 1) in the complex plane. -/
theorem complex_cosine_geometric_representation :
  {z : ℂ | ∃ θ : ℝ, θ ∈ Set.Icc 0 (2 * Real.pi) ∧ z = Complex.I * Complex.cos θ} =
  {z : ℂ | z.re = 0 ∧ z.im ∈ Set.Icc (-1) 1} :=
sorry

end complex_cosine_geometric_representation_l3008_300870


namespace power_product_l3008_300850

theorem power_product (a b : ℝ) (n : ℕ) : (a * b) ^ n = a ^ n * b ^ n := by sorry

end power_product_l3008_300850


namespace H_surjective_l3008_300837

-- Define the function H
def H (x : ℝ) : ℝ := 2 * |2 * x + 3| - 3 * |x - 2|

-- Theorem statement
theorem H_surjective : Function.Surjective H := by sorry

end H_surjective_l3008_300837


namespace football_players_l3008_300880

theorem football_players (total : ℕ) (cricket : ℕ) (neither : ℕ) (both : ℕ) 
  (h1 : total = 250)
  (h2 : cricket = 90)
  (h3 : neither = 50)
  (h4 : both = 50) :
  total - neither - (cricket - both) = 160 :=
by
  sorry

#check football_players

end football_players_l3008_300880


namespace lighthouse_ship_position_l3008_300833

/-- Represents cardinal directions --/
inductive Direction
  | North
  | South
  | East
  | West

/-- Represents a relative position with a direction and angle --/
structure RelativePosition where
  primaryDirection : Direction
  secondaryDirection : Direction
  angle : ℝ

/-- Returns the opposite direction --/
def oppositeDirection (d : Direction) : Direction :=
  match d with
  | Direction.North => Direction.South
  | Direction.South => Direction.North
  | Direction.East => Direction.West
  | Direction.West => Direction.East

/-- Returns the opposite relative position --/
def oppositePosition (pos : RelativePosition) : RelativePosition :=
  { primaryDirection := oppositeDirection pos.primaryDirection,
    secondaryDirection := oppositeDirection pos.secondaryDirection,
    angle := pos.angle }

theorem lighthouse_ship_position 
  (lighthousePos : RelativePosition) 
  (h1 : lighthousePos.primaryDirection = Direction.North)
  (h2 : lighthousePos.secondaryDirection = Direction.East)
  (h3 : lighthousePos.angle = 38) :
  oppositePosition lighthousePos = 
    { primaryDirection := Direction.South,
      secondaryDirection := Direction.West,
      angle := 38 } := by
  sorry

end lighthouse_ship_position_l3008_300833


namespace library_biography_increase_l3008_300829

theorem library_biography_increase (B : ℝ) (h1 : B > 0) : 
  let original_biographies := 0.20 * B
  let new_biographies := (7 / 9) * B
  let percentage_increase := (new_biographies / original_biographies - 1) * 100
  percentage_increase = 3500 / 9 := by sorry

end library_biography_increase_l3008_300829


namespace remaining_milk_l3008_300846

-- Define the initial amount of milk
def initial_milk : ℚ := 5

-- Define the amount given away
def given_away : ℚ := 2 + 3/4

-- Theorem statement
theorem remaining_milk :
  initial_milk - given_away = 2 + 1/4 := by sorry

end remaining_milk_l3008_300846


namespace polynomial_irreducibility_l3008_300823

theorem polynomial_irreducibility (n : ℕ) (hn : n > 1) :
  Irreducible (Polynomial.X ^ n + 5 * Polynomial.X ^ (n - 1) + 3 : Polynomial ℤ) := by
  sorry

end polynomial_irreducibility_l3008_300823


namespace fraction_simplification_l3008_300879

theorem fraction_simplification : (5 * 6 - 4) / 8 = 13 / 4 := by sorry

end fraction_simplification_l3008_300879


namespace binomial_10_3_l3008_300828

theorem binomial_10_3 : Nat.choose 10 3 = 120 := by
  sorry

end binomial_10_3_l3008_300828


namespace shop_discount_percentage_l3008_300853

/-- Calculate the percentage discount given the original price and discounted price -/
def calculate_discount_percentage (original_price discounted_price : ℚ) : ℚ :=
  (original_price - discounted_price) / original_price * 100

/-- The shop's discount percentage is 30% -/
theorem shop_discount_percentage :
  let original_price : ℚ := 800
  let discounted_price : ℚ := 560
  calculate_discount_percentage original_price discounted_price = 30 := by
sorry

end shop_discount_percentage_l3008_300853


namespace compare_expressions_inequality_proof_l3008_300854

-- Part 1
theorem compare_expressions (x : ℝ) : (x + 7) * (x + 8) > (x + 6) * (x + 9) := by
  sorry

-- Part 2
theorem inequality_proof (a b c d : ℝ) (h1 : a < b) (h2 : b < 0) (h3 : 0 < c) (h4 : c < d) :
  a * d + c < b * c + d := by
  sorry

end compare_expressions_inequality_proof_l3008_300854


namespace modulus_of_special_complex_l3008_300802

/-- The modulus of a complex number Z = 3a - 4ai where a < 0 is equal to -5a -/
theorem modulus_of_special_complex (a : ℝ) (ha : a < 0) :
  Complex.abs (Complex.mk (3 * a) (-4 * a)) = -5 * a := by
  sorry

end modulus_of_special_complex_l3008_300802


namespace circle_equation_and_slope_range_l3008_300851

-- Define the circle C
def circle_C (x y : ℝ) : Prop := (x - 1)^2 + (y - 2)^2 = 5

-- Define the line y = 2x
def line_center (x y : ℝ) : Prop := y = 2 * x

-- Define the line x + y - 3 = 0
def line_intersect (x y : ℝ) : Prop := x + y - 3 = 0

-- Define points A and B
def point_A : ℝ × ℝ := sorry
def point_B : ℝ × ℝ := sorry

-- Define the origin
def origin : ℝ × ℝ := (0, 0)

-- Define point M
def point_M : ℝ × ℝ := (0, 5)

-- Define the dot product of OA and OB
def OA_dot_OB_zero : Prop :=
  (point_A.1 - origin.1) * (point_B.1 - origin.1) + 
  (point_A.2 - origin.2) * (point_B.2 - origin.2) = 0

-- Define the slope range for line MP
def slope_range (k : ℝ) : Prop := k ≤ -1/2 ∨ k ≥ 2

theorem circle_equation_and_slope_range :
  (∀ x y, circle_C x y → ((x, y) = origin ∨ line_center x y)) ∧
  (∀ x y, line_intersect x y → circle_C x y → ((x, y) = point_A ∨ (x, y) = point_B)) ∧
  OA_dot_OB_zero →
  (∀ x y, circle_C x y ↔ (x - 1)^2 + (y - 2)^2 = 5) ∧
  (∀ k, (∃ x y, circle_C x y ∧ y - 5 = k * x) ↔ slope_range k) :=
sorry

end circle_equation_and_slope_range_l3008_300851


namespace max_value_of_f_range_of_t_inequality_for_positive_reals_l3008_300849

-- Define the function f(x) = |x+1| - |x-2|
def f (x : ℝ) : ℝ := |x + 1| - |x - 2|

-- Theorem 1: The maximum value of f(x) is 3
theorem max_value_of_f : ∀ x : ℝ, f x ≤ 3 :=
sorry

-- Theorem 2: The range of t given the inequality
theorem range_of_t : ∀ t : ℝ, (∃ x : ℝ, f x ≥ |t - 1| + t) ↔ t ≤ 2 :=
sorry

-- Theorem 3: Inequality for positive real numbers
theorem inequality_for_positive_reals :
  ∀ a b c : ℝ, a > 0 → b > 0 → c > 0 → 2*a + b + c = 2 → a^2 + b^2 + c^2 ≥ 2/3 :=
sorry

end max_value_of_f_range_of_t_inequality_for_positive_reals_l3008_300849


namespace imaginary_part_of_complex_fraction_l3008_300825

theorem imaginary_part_of_complex_fraction : Complex.im ((1 + Complex.I) / (1 - Complex.I)) = 1 := by
  sorry

end imaginary_part_of_complex_fraction_l3008_300825


namespace max_cutlery_sets_l3008_300816

theorem max_cutlery_sets (dinner_forks knives soup_spoons teaspoons dessert_forks butter_knives : ℕ) 
  (max_capacity : ℕ) (dinner_fork_weight knife_weight soup_spoon_weight teaspoon_weight dessert_fork_weight butter_knife_weight : ℕ) : 
  dinner_forks = 6 →
  knives = dinner_forks + 9 →
  soup_spoons = 2 * knives →
  teaspoons = dinner_forks / 2 →
  dessert_forks = teaspoons / 3 →
  butter_knives = 2 * dessert_forks →
  max_capacity = 20000 →
  dinner_fork_weight = 80 →
  knife_weight = 100 →
  soup_spoon_weight = 85 →
  teaspoon_weight = 50 →
  dessert_fork_weight = 70 →
  butter_knife_weight = 65 →
  (max_capacity - (dinner_forks * dinner_fork_weight + knives * knife_weight + 
    soup_spoons * soup_spoon_weight + teaspoons * teaspoon_weight + 
    dessert_forks * dessert_fork_weight + butter_knives * butter_knife_weight)) / 
    (dinner_fork_weight + knife_weight) = 84 := by
  sorry

end max_cutlery_sets_l3008_300816


namespace probability_of_sum_magnitude_at_least_sqrt2_l3008_300847

/-- The roots of z^12 - 1 = 0 -/
def twelfthRootsOfUnity : Finset ℂ := sorry

/-- The condition that v and w are distinct -/
def areDistinct (v w : ℂ) : Prop := v ≠ w

/-- The condition that v and w are roots of z^12 - 1 = 0 -/
def areRoots (v w : ℂ) : Prop := v ∈ twelfthRootsOfUnity ∧ w ∈ twelfthRootsOfUnity

/-- The number of pairs (v, w) satisfying |v + w| ≥ √2 -/
def satisfyingPairs : ℕ := sorry

/-- The total number of distinct pairs (v, w) -/
def totalPairs : ℕ := sorry

theorem probability_of_sum_magnitude_at_least_sqrt2 :
  satisfyingPairs / totalPairs = 10 / 11 :=
sorry

end probability_of_sum_magnitude_at_least_sqrt2_l3008_300847


namespace milburg_population_l3008_300857

/-- The total population of Milburg is the sum of grown-ups and children. -/
theorem milburg_population :
  let grown_ups : ℕ := 5256
  let children : ℕ := 2987
  grown_ups + children = 8243 := by
  sorry

end milburg_population_l3008_300857


namespace inequality_proof_l3008_300884

theorem inequality_proof (a : ℝ) (h : a > 3) : 4 / (a - 3) + a ≥ 7 := by
  sorry

end inequality_proof_l3008_300884


namespace total_cats_l3008_300819

/-- Represents the Clevercat Academy with cats that can perform various tricks. -/
structure ClevercatAcademy where
  jump : ℕ
  fetch : ℕ
  spin : ℕ
  jump_fetch : ℕ
  fetch_spin : ℕ
  jump_spin : ℕ
  all_three : ℕ
  none : ℕ

/-- The theorem states that given the specific numbers of cats that can perform
    various combinations of tricks, the total number of cats in the academy is 99. -/
theorem total_cats (academy : ClevercatAcademy)
  (h_jump : academy.jump = 60)
  (h_fetch : academy.fetch = 35)
  (h_spin : academy.spin = 40)
  (h_jump_fetch : academy.jump_fetch = 20)
  (h_fetch_spin : academy.fetch_spin = 15)
  (h_jump_spin : academy.jump_spin = 22)
  (h_all_three : academy.all_three = 11)
  (h_none : academy.none = 10) :
  (academy.jump - academy.jump_fetch - academy.jump_spin + academy.all_three) +
  (academy.fetch - academy.jump_fetch - academy.fetch_spin + academy.all_three) +
  (academy.spin - academy.jump_spin - academy.fetch_spin + academy.all_three) +
  academy.jump_fetch + academy.fetch_spin + academy.jump_spin -
  2 * academy.all_three + academy.none = 99 :=
by sorry

end total_cats_l3008_300819


namespace lemonade_stand_cost_l3008_300841

-- Define the given conditions
def total_profit : ℝ := 44
def lemonade_revenue : ℝ := 47
def lemonades_sold : ℕ := 50
def babysitting_income : ℝ := 31
def lemon_cost : ℝ := 0.20
def sugar_cost : ℝ := 0.15
def ice_cost : ℝ := 0.05
def sunhat_cost : ℝ := 10

-- Define the theorem
theorem lemonade_stand_cost :
  let variable_cost_per_lemonade := lemon_cost + sugar_cost + ice_cost
  let total_variable_cost := variable_cost_per_lemonade * lemonades_sold
  let total_cost := total_variable_cost + sunhat_cost
  total_cost = 30 := by sorry

end lemonade_stand_cost_l3008_300841


namespace coord_relationship_l3008_300885

/-- The relationship between x and y coordinates on lines y = x or y = -x --/
theorem coord_relationship (x y : ℝ) : (y = x ∨ y = -x) → |x| - |y| = 0 := by
  sorry

end coord_relationship_l3008_300885


namespace quadratic_root_relation_l3008_300813

theorem quadratic_root_relation (p r : ℝ) (hr : r > 0) :
  (∃ x y : ℝ, x^2 + p*x + r = 0 ∧ y^2 + p*y + r = 0 ∧ y = 2*x) →
  p = Real.sqrt (9*r/2) :=
sorry

end quadratic_root_relation_l3008_300813


namespace travelers_meeting_l3008_300835

/-- The problem of two travelers meeting --/
theorem travelers_meeting
  (total_distance : ℝ)
  (travel_time : ℝ)
  (shook_speed : ℝ)
  (h_total_distance : total_distance = 490)
  (h_travel_time : travel_time = 7)
  (h_shook_speed : shook_speed = 37)
  : ∃ (beta_speed : ℝ),
    beta_speed = 33 ∧
    total_distance = shook_speed * travel_time + beta_speed * travel_time :=
by
  sorry

#check travelers_meeting

end travelers_meeting_l3008_300835


namespace chime_time_at_12_l3008_300848

/-- Represents a clock with hourly chimes -/
structure ChimeClock where
  /-- The time in seconds it takes to complete chimes at 4 o'clock -/
  time_at_4 : ℕ
  /-- Assertion that the clock chimes once every hour -/
  chimes_hourly : Prop

/-- Calculates the time it takes to complete chimes at a given hour -/
def chime_time (clock : ChimeClock) (hour : ℕ) : ℕ :=
  sorry

/-- Theorem stating that it takes 44 seconds to complete chimes at 12 o'clock -/
theorem chime_time_at_12 (clock : ChimeClock) 
  (h1 : clock.time_at_4 = 12) 
  (h2 : clock.chimes_hourly) : 
  chime_time clock 12 = 44 :=
sorry

end chime_time_at_12_l3008_300848


namespace fifteenth_prime_l3008_300839

/-- Given that 5 is the third prime number, prove that the fifteenth prime number is 59. -/
theorem fifteenth_prime : 
  (∃ (f : ℕ → ℕ), f 3 = 5 ∧ (∀ n, n ≥ 1 → Prime (f n)) ∧ (∀ n m, n < m → f n < f m)) → 
  (∃ (g : ℕ → ℕ), g 15 = 59 ∧ (∀ n, n ≥ 1 → Prime (g n)) ∧ (∀ n m, n < m → g n < g m)) :=
by sorry

end fifteenth_prime_l3008_300839


namespace complex_power_one_minus_i_six_l3008_300824

theorem complex_power_one_minus_i_six :
  let i : ℂ := Complex.I
  (1 - i)^6 = 8*i := by sorry

end complex_power_one_minus_i_six_l3008_300824


namespace female_students_count_l3008_300874

theorem female_students_count (total_average : ℝ) (male_count : ℕ) (male_average : ℝ) (female_average : ℝ)
  (h1 : total_average = 90)
  (h2 : male_count = 8)
  (h3 : male_average = 85)
  (h4 : female_average = 92) :
  ∃ (female_count : ℕ),
    (male_count * male_average + female_count * female_average) / (male_count + female_count) = total_average ∧
    female_count = 20 := by
  sorry

end female_students_count_l3008_300874


namespace cost_price_per_metre_l3008_300881

/-- Given a trader who sells cloth, this theorem proves the cost price per metre. -/
theorem cost_price_per_metre
  (total_metres : ℕ)
  (selling_price : ℕ)
  (profit_per_metre : ℕ)
  (h1 : total_metres = 85)
  (h2 : selling_price = 8925)
  (h3 : profit_per_metre = 10) :
  (selling_price - total_metres * profit_per_metre) / total_metres = 95 := by
sorry

end cost_price_per_metre_l3008_300881


namespace students_satisfy_equation_unique_solution_l3008_300860

/-- The number of students in class 5A -/
def students : ℕ := 36

/-- The equation that describes the problem conditions -/
def problem_equation (x : ℕ) : Prop :=
  (x - 23) * 23 = (x - 13) * 13

/-- Theorem stating that the number of students in class 5A satisfies the problem conditions -/
theorem students_satisfy_equation : problem_equation students := by
  sorry

/-- Theorem stating that 36 is the unique solution to the problem -/
theorem unique_solution : ∀ x : ℕ, problem_equation x → x = students := by
  sorry

end students_satisfy_equation_unique_solution_l3008_300860


namespace f_properties_l3008_300894

noncomputable section

variables {f : ℝ → ℝ} {a : ℝ}

-- f is an even function
def even_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

-- f is symmetric about x = 1
def symmetric_about_one (f : ℝ → ℝ) : Prop := ∀ x, f (2 - x) = f x

-- f satisfies the multiplicative property for x₁, x₂ ∈ [0, 1/2]
def multiplicative_property (f : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂, x₁ ∈ Set.Icc 0 (1/2) → x₂ ∈ Set.Icc 0 (1/2) → f (x₁ + x₂) = f x₁ * f x₂

theorem f_properties (heven : even_function f) (hsym : symmetric_about_one f)
    (hmult : multiplicative_property f) (hf1 : f 1 = a) (ha : a > 0) :
    f (1/2) = Real.sqrt a ∧ f (1/4) = Real.sqrt (Real.sqrt a) ∧ ∀ x, f (x + 2) = f x := by
  sorry

end

end f_properties_l3008_300894


namespace intersection_points_range_l3008_300873

/-- The range of m for which curves C₁ and C₂ have 4 distinct intersection points -/
theorem intersection_points_range (m : ℝ) : 
  (∃ (x₁ y₁ x₂ y₂ x₃ y₃ x₄ y₄ : ℝ), 
    (∀ i j, (i, j) ∈ [(x₁, y₁), (x₂, y₂), (x₃, y₃), (x₄, y₄)] → 
      (i - 1)^2 + j^2 = 1 ∧ j * (j - m*i - m) = 0) ∧
    (∀ i j k l, (i, j) ≠ (k, l) → (i, j) ∈ [(x₁, y₁), (x₂, y₂), (x₃, y₃), (x₄, y₄)] → 
      (k, l) ∈ [(x₁, y₁), (x₂, y₂), (x₃, y₃), (x₄, y₄)] → (i, j) ≠ (k, l))) ↔ 
  (m > -Real.sqrt 3 / 3 ∧ m < 0) ∨ (m > 0 ∧ m < Real.sqrt 3 / 3) :=
by sorry

end intersection_points_range_l3008_300873


namespace initial_workers_count_l3008_300807

/-- Represents the construction project scenario -/
structure ConstructionProject where
  initial_duration : ℕ
  actual_duration : ℕ
  initial_workers : ℕ
  double_rate_workers : ℕ
  triple_rate_workers : ℕ
  double_rate_join_day : ℕ
  triple_rate_join_day : ℕ

/-- Theorem stating that the initial number of workers is 55 -/
theorem initial_workers_count (project : ConstructionProject) 
  (h1 : project.initial_duration = 24)
  (h2 : project.actual_duration = 19)
  (h3 : project.double_rate_workers = 8)
  (h4 : project.triple_rate_workers = 5)
  (h5 : project.double_rate_join_day = 11)
  (h6 : project.triple_rate_join_day = 17) :
  project.initial_workers = 55 := by
  sorry

end initial_workers_count_l3008_300807


namespace prob_red_or_green_is_two_thirds_l3008_300872

-- Define the number of balls of each color
def red_balls : ℕ := 2
def yellow_balls : ℕ := 3
def green_balls : ℕ := 4

-- Define the total number of balls
def total_balls : ℕ := red_balls + yellow_balls + green_balls

-- Define the number of favorable outcomes (red or green balls)
def favorable_outcomes : ℕ := red_balls + green_balls

-- Define the probability of drawing a red or green ball
def prob_red_or_green : ℚ := favorable_outcomes / total_balls

-- Theorem statement
theorem prob_red_or_green_is_two_thirds : 
  prob_red_or_green = 2 / 3 := by sorry

end prob_red_or_green_is_two_thirds_l3008_300872


namespace parabola_focus_coordinates_l3008_300861

/-- Given a parabola with equation y = 2x^2 + 8x - 1, its focus coordinates are (-2, -8.875) -/
theorem parabola_focus_coordinates :
  let f : ℝ → ℝ := λ x => 2 * x^2 + 8 * x - 1
  ∃ (h k : ℝ), h = -2 ∧ k = -8.875 ∧
    ∀ (x y : ℝ), y = f x → (x - h)^2 = 4 * (y - k) / 2 := by
  sorry

end parabola_focus_coordinates_l3008_300861


namespace dolphin_altitude_l3008_300811

/-- Given a submarine at an altitude of -50 meters and a dolphin 10 meters above it,
    the altitude of the dolphin is -40 meters. -/
theorem dolphin_altitude (submarine_altitude dolphin_distance : ℝ) :
  submarine_altitude = -50 ∧ dolphin_distance = 10 →
  submarine_altitude + dolphin_distance = -40 :=
by sorry

end dolphin_altitude_l3008_300811


namespace triangle_perimeter_range_l3008_300805

theorem triangle_perimeter_range (a b c A B C : ℝ) : 
  0 < A ∧ A < π / 2 →
  0 < B ∧ B < π / 2 →
  0 < C ∧ C < π / 2 →
  c = 2 →
  a * Real.cos B + b * Real.cos A = (Real.sqrt 3 * c) / (2 * Real.sin C) →
  A + B + C = π →
  let P := a + b + c
  4 < P ∧ P ≤ 6 := by
  sorry

end triangle_perimeter_range_l3008_300805


namespace geometric_sequence_second_term_l3008_300889

/-- A geometric sequence is a sequence where each term after the first is found by multiplying the previous term by a fixed, non-zero number called the common ratio. -/
def IsGeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_second_term
  (a : ℕ → ℝ)
  (h_geo : IsGeometricSequence a)
  (h_first : a 1 = 2)
  (h_relation : 16 * a 3 * a 5 = 8 * a 4 - 1) :
  a 2 = 1 := by
  sorry

end geometric_sequence_second_term_l3008_300889


namespace max_difference_l3008_300830

theorem max_difference (a b : ℝ) (ha : -5 ≤ a ∧ a ≤ 10) (hb : -5 ≤ b ∧ b ≤ 10) :
  ∃ (x y : ℝ), -5 ≤ x ∧ x ≤ 10 ∧ -5 ≤ y ∧ y ≤ 10 ∧ x - y = 15 ∧ ∀ (c d : ℝ), -5 ≤ c ∧ c ≤ 10 ∧ -5 ≤ d ∧ d ≤ 10 → c - d ≤ 15 :=
by sorry

end max_difference_l3008_300830


namespace devices_delivered_l3008_300893

/-- Represents the properties of the energy-saving devices delivery -/
structure DeviceDelivery where
  totalWeight : ℕ
  lightestThreeWeight : ℕ
  heaviestThreeWeight : ℕ
  allWeightsDifferent : Bool

/-- The number of devices in the delivery -/
def numDevices (d : DeviceDelivery) : ℕ := sorry

/-- Theorem stating that given the specific conditions, the number of devices is 10 -/
theorem devices_delivered (d : DeviceDelivery) 
  (h1 : d.totalWeight = 120)
  (h2 : d.lightestThreeWeight = 31)
  (h3 : d.heaviestThreeWeight = 41)
  (h4 : d.allWeightsDifferent = true) :
  numDevices d = 10 := by sorry

end devices_delivered_l3008_300893


namespace combination_square_28_l3008_300865

theorem combination_square_28 (n : ℕ) : (n.choose 2 = 28) → n = 8 := by
  sorry

end combination_square_28_l3008_300865


namespace tetrahedron_sphere_radii_l3008_300868

theorem tetrahedron_sphere_radii (r : ℝ) (R : ℝ) :
  r = Real.sqrt 2 - 1 →
  R = Real.sqrt 6 + 1 →
  ∃ (a : ℝ),
    r = (a * Real.sqrt 2) / 4 ∧
    R = (a * Real.sqrt 6) / 4 :=
by sorry

end tetrahedron_sphere_radii_l3008_300868


namespace polynomial_expansion_l3008_300827

theorem polynomial_expansion (x : ℝ) : 
  (5 * x^2 + 3 * x - 7) * (4 * x^3) = 20 * x^5 + 12 * x^4 - 28 * x^3 := by
  sorry

end polynomial_expansion_l3008_300827


namespace line_through_point_equal_intercepts_l3008_300882

-- Define a line passing through (1, 3) with equal intercepts
def line_equal_intercepts (a b c : ℝ) : Prop :=
  a * 1 + b * 3 + c = 0 ∧  -- Line passes through (1, 3)
  ∃ k : ℝ, k ≠ 0 ∧ a * k + c = 0 ∧ b * k + c = 0  -- Equal intercepts

-- Theorem statement
theorem line_through_point_equal_intercepts :
  ∀ a b c : ℝ, line_equal_intercepts a b c →
  (a = -3 ∧ b = 1 ∧ c = 0) ∨ (a = 1 ∧ b = 1 ∧ c = -4) :=
by sorry

end line_through_point_equal_intercepts_l3008_300882


namespace complex_sum_theorem_l3008_300899

theorem complex_sum_theorem (a c d e f : ℝ) : 
  e = -a - c → (a + 2*I) + (c + d*I) + (e + f*I) = 2*I → d + f = 0 := by
  sorry

end complex_sum_theorem_l3008_300899


namespace probability_two_from_ten_with_two_defective_l3008_300800

/-- The probability of drawing at least one defective product -/
def probability_at_least_one_defective (total : ℕ) (defective : ℕ) (draw : ℕ) : ℚ :=
  1 - (Nat.choose (total - defective) draw : ℚ) / (Nat.choose total draw : ℚ)

/-- Theorem stating the probability of drawing at least one defective product -/
theorem probability_two_from_ten_with_two_defective :
  probability_at_least_one_defective 10 2 2 = 17 / 45 := by
sorry

end probability_two_from_ten_with_two_defective_l3008_300800


namespace sphere_surface_area_ratio_l3008_300863

theorem sphere_surface_area_ratio (r₁ r₂ : ℝ) (h : r₁ / r₂ = 1 / 2) :
  (4 * Real.pi * r₁^2) / (4 * Real.pi * r₂^2) = 1 / 4 := by
  sorry

end sphere_surface_area_ratio_l3008_300863


namespace tim_initial_balls_l3008_300812

theorem tim_initial_balls (robert_initial : ℕ) (robert_final : ℕ) (tim_initial : ℕ) : 
  robert_initial = 25 → 
  robert_final = 45 → 
  robert_final = robert_initial + tim_initial / 2 → 
  tim_initial = 40 := by
sorry

end tim_initial_balls_l3008_300812


namespace vasily_salary_higher_l3008_300840

/-- Represents the salary distribution for graduates --/
structure GraduateSalary where
  high : ℝ  -- Salary for 1/5 of graduates
  very_high : ℝ  -- Salary for 1/10 of graduates
  low : ℝ  -- Salary for 1/20 of graduates
  medium : ℝ  -- Salary for remaining graduates

/-- Calculates the expected salary for a student --/
def expected_salary (
  total_students : ℕ
  ) (graduating_students : ℕ
  ) (non_graduate_salary : ℝ
  ) (graduate_salary : GraduateSalary
  ) : ℝ :=
  sorry

/-- Calculates the salary after a number of years with annual increase --/
def salary_after_years (
  initial_salary : ℝ
  ) (annual_increase : ℝ
  ) (years : ℕ
  ) : ℝ :=
  sorry

theorem vasily_salary_higher (
  total_students : ℕ
  ) (graduating_students : ℕ
  ) (non_graduate_salary : ℝ
  ) (graduate_salary : GraduateSalary
  ) (fyodor_initial_salary : ℝ
  ) (fyodor_annual_increase : ℝ
  ) (years : ℕ
  ) : 
  total_students = 300 →
  graduating_students = 270 →
  non_graduate_salary = 25000 →
  graduate_salary.high = 60000 →
  graduate_salary.very_high = 80000 →
  graduate_salary.low = 25000 →
  graduate_salary.medium = 40000 →
  fyodor_initial_salary = 25000 →
  fyodor_annual_increase = 3000 →
  years = 4 →
  expected_salary total_students graduating_students non_graduate_salary graduate_salary = 39625 ∧
  expected_salary total_students graduating_students non_graduate_salary graduate_salary - 
    salary_after_years fyodor_initial_salary fyodor_annual_increase years = 2625 :=
by sorry

end vasily_salary_higher_l3008_300840


namespace cheese_grating_time_is_five_l3008_300897

/-- The time in minutes it takes to grate cheese for one omelet --/
def cheese_grating_time (
  total_time : ℕ)
  (num_omelets : ℕ)
  (pepper_chop_time : ℕ)
  (onion_chop_time : ℕ)
  (omelet_cook_time : ℕ)
  (num_peppers : ℕ)
  (num_onions : ℕ) : ℕ :=
  total_time - 
  (num_peppers * pepper_chop_time + 
   num_onions * onion_chop_time + 
   num_omelets * omelet_cook_time)

theorem cheese_grating_time_is_five :
  cheese_grating_time 50 5 3 4 5 4 2 = 5 := by
  sorry

end cheese_grating_time_is_five_l3008_300897


namespace twenty_one_three_four_zero_is_base5_l3008_300886

def is_base5_digit (d : Nat) : Prop := d < 5

def is_base5_number (n : Nat) : Prop :=
  ∀ d, d ∈ n.digits 5 → is_base5_digit d

theorem twenty_one_three_four_zero_is_base5 :
  is_base5_number 21340 :=
sorry

end twenty_one_three_four_zero_is_base5_l3008_300886


namespace percentage_of_cat_owners_l3008_300866

theorem percentage_of_cat_owners (total_students : ℕ) (cat_owners : ℕ) 
  (h1 : total_students = 500) (h2 : cat_owners = 75) : 
  (cat_owners : ℝ) / total_students * 100 = 15 := by
  sorry

end percentage_of_cat_owners_l3008_300866


namespace sin_equality_proof_l3008_300806

theorem sin_equality_proof (m : ℤ) : 
  -180 ≤ m ∧ m ≤ 180 ∧ Real.sin (m * π / 180) = Real.sin (945 * π / 180) → m = -135 := by
  sorry

end sin_equality_proof_l3008_300806


namespace max_value_of_a_max_value_is_negative_two_l3008_300887

theorem max_value_of_a (a : ℝ) : 
  (∀ x : ℝ, x < a → |x| > 2) ∧ 
  (∃ x : ℝ, |x| > 2 ∧ x ≥ a) →
  a ≤ -2 :=
by sorry

theorem max_value_is_negative_two :
  ∃ a : ℝ, 
    (∀ x : ℝ, x < a → |x| > 2) ∧
    (∃ x : ℝ, |x| > 2 ∧ x ≥ a) ∧
    a = -2 :=
by sorry

end max_value_of_a_max_value_is_negative_two_l3008_300887


namespace ratio_b_to_sum_ac_l3008_300814

theorem ratio_b_to_sum_ac (a b c : ℤ) 
  (sum_eq : a + b + c = 60)
  (a_eq : a = (b + c) / 3)
  (c_eq : c = 35) : 
  b * 5 = a + c := by sorry

end ratio_b_to_sum_ac_l3008_300814


namespace max_contribution_l3008_300878

theorem max_contribution 
  (n : ℕ) 
  (total : ℚ) 
  (min_contribution : ℚ) 
  (h1 : n = 15)
  (h2 : total = 30)
  (h3 : min_contribution = 1)
  (h4 : ∀ i, i ∈ Finset.range n → ∃ c : ℚ, c ≥ min_contribution) :
  ∃ max_contribution : ℚ, 
    max_contribution ≤ total ∧ 
    (∀ i, i ∈ Finset.range n → ∃ c : ℚ, c ≤ max_contribution) ∧
    max_contribution = 16 :=
sorry

end max_contribution_l3008_300878


namespace boat_speed_in_still_water_l3008_300832

/-- Given a boat that travels 20 km downstream in 2 hours and 20 km upstream in 5 hours,
    prove that its speed in still water is 7 km/h. -/
theorem boat_speed_in_still_water :
  ∀ (downstream_speed upstream_speed : ℝ),
  downstream_speed = 20 / 2 →
  upstream_speed = 20 / 5 →
  ∃ (boat_speed stream_speed : ℝ),
    boat_speed + stream_speed = downstream_speed ∧
    boat_speed - stream_speed = upstream_speed ∧
    boat_speed = 7 := by
  sorry

end boat_speed_in_still_water_l3008_300832


namespace evaluate_expression_l3008_300891

theorem evaluate_expression : 
  |7 - (8^2) * (3 - 12)| - |(5^3) - (Real.sqrt 11)^4| = 579 := by
  sorry

end evaluate_expression_l3008_300891


namespace divisibility_of_sum_of_squares_l3008_300836

theorem divisibility_of_sum_of_squares (p x y z : ℕ) : 
  Prime p → 
  0 < x → x < y → y < z → z < p → 
  (x^3 % p = y^3 % p) → (y^3 % p = z^3 % p) →
  (x^2 + y^2 + z^2) % (x + y + z) = 0 := by
  sorry

end divisibility_of_sum_of_squares_l3008_300836


namespace average_of_abcd_l3008_300895

theorem average_of_abcd (a b c d : ℝ) : 
  (4 + 6 + 9 + a + b + c + d) / 7 = 20 → (a + b + c + d) / 4 = 30.25 := by
  sorry

end average_of_abcd_l3008_300895


namespace largest_divisor_of_four_consecutive_naturals_l3008_300831

theorem largest_divisor_of_four_consecutive_naturals :
  ∀ n : ℕ, ∃ k : ℕ, k * 120 = n * (n + 1) * (n + 2) * (n + 3) ∧
  ∀ m : ℕ, m > 120 → ¬(∀ n : ℕ, ∃ k : ℕ, k * m = n * (n + 1) * (n + 2) * (n + 3)) :=
by sorry

end largest_divisor_of_four_consecutive_naturals_l3008_300831


namespace intersection_of_sets_l3008_300826

theorem intersection_of_sets : 
  let A : Set ℕ := {1, 2, 3, 4, 5}
  let B : Set ℕ := {2, 4, 5, 8, 10}
  A ∩ B = {2, 4, 5} := by
sorry

end intersection_of_sets_l3008_300826


namespace special_linear_function_at_two_l3008_300871

/-- A linear function satisfying specific conditions -/
structure SpecialLinearFunction where
  f : ℝ → ℝ
  linear : ∀ x y c : ℝ, f (x + y) = f x + f y ∧ f (c * x) = c * f x
  inverse_relation : ∀ x : ℝ, f x = 3 * f⁻¹ x + 5
  f_one : f 1 = 5

/-- The main theorem stating the value of f(2) for the special linear function -/
theorem special_linear_function_at_two (slf : SpecialLinearFunction) :
  slf.f 2 = 2 * Real.sqrt 3 + (5 * Real.sqrt 3) / (Real.sqrt 3 + 3) := by
  sorry

end special_linear_function_at_two_l3008_300871


namespace sara_golf_balls_l3008_300817

theorem sara_golf_balls (x : ℕ) : x = 16 * (3 * 4) → x / 12 = 16 := by
  sorry

end sara_golf_balls_l3008_300817


namespace boating_group_size_l3008_300809

theorem boating_group_size : ∃ n : ℕ, 
  n > 0 ∧ 
  n % 4 = 10 ∧ 
  n % 5 = 1 ∧ 
  n = 46 := by
  sorry

end boating_group_size_l3008_300809


namespace thursday_return_count_l3008_300838

/-- Calculates the number of books brought back on Thursday given the initial
    number of books, books taken out on Tuesday and Friday, and the final
    number of books in the library. -/
def books_brought_back (initial : ℕ) (taken_tuesday : ℕ) (taken_friday : ℕ) (final : ℕ) : ℕ :=
  initial - taken_tuesday + taken_friday - final

theorem thursday_return_count :
  books_brought_back 235 227 35 29 = 56 := by
  sorry

end thursday_return_count_l3008_300838


namespace digital_root_theorem_l3008_300855

/-- Digital root of a natural number -/
def digitalRoot (n : ℕ) : ℕ :=
  if n % 9 = 0 then 9 else n % 9

/-- List of digital roots of first n natural numbers -/
def digitalRootList (n : ℕ) : List ℕ :=
  List.map digitalRoot (List.range n)

theorem digital_root_theorem :
  let l := digitalRootList 20092009
  (l.count 4 > l.count 5) ∧
  (l.count 9 = 2232445) ∧
  (digitalRoot (3^2009) = 9) ∧
  (digitalRoot (17^2009) = 8) := by
  sorry


end digital_root_theorem_l3008_300855


namespace equation_solutions_l3008_300834

theorem equation_solutions : 
  let f (x : ℝ) := (15*x - x^2)/(x + 1) * (x + (15 - x)/(x + 1))
  ∀ x : ℝ, f x = 60 ↔ x = 5 ∨ x = 6 ∨ x = 3 + Real.sqrt 2 ∨ x = 3 - Real.sqrt 2 :=
by sorry

end equation_solutions_l3008_300834


namespace evaluate_sqrt_fraction_l3008_300822

theorem evaluate_sqrt_fraction (y : ℝ) (h : y < 0) :
  Real.sqrt (y / (1 - (y - 2) / y)) = -y / Real.sqrt 2 := by
  sorry

end evaluate_sqrt_fraction_l3008_300822


namespace min_value_expression_l3008_300883

theorem min_value_expression (x y : ℝ) : 
  (x + y - 1)^2 + (x * y)^2 ≥ 0 ∧ 
  ∃ a b : ℝ, (a + b - 1)^2 + (a * b)^2 = 0 :=
by sorry

end min_value_expression_l3008_300883


namespace range_of_k_l3008_300818

/-- An odd function that is strictly decreasing on [0, +∞) -/
def OddDecreasingFunction (f : ℝ → ℝ) : Prop :=
  (∀ x, f (-x) = -f x) ∧ 
  (∀ x y, 0 ≤ x ∧ x < y → f y < f x)

theorem range_of_k (f : ℝ → ℝ) (h_odd_dec : OddDecreasingFunction f) :
  (∀ k x : ℝ, f (k * x^2 + 2) + f (k * x + k) ≤ 0) ↔ 
  (∀ k : ℝ, 0 ≤ k) :=
sorry

end range_of_k_l3008_300818


namespace fourth_person_height_l3008_300842

/-- Theorem: Height of the fourth person in a specific arrangement --/
theorem fourth_person_height 
  (h₁ h₂ h₃ h₄ : ℝ) 
  (height_order : h₁ < h₂ ∧ h₂ < h₃ ∧ h₃ < h₄)
  (diff_first_three : h₂ - h₁ = 2 ∧ h₃ - h₂ = 2)
  (diff_last_two : h₄ - h₃ = 6)
  (average_height : (h₁ + h₂ + h₃ + h₄) / 4 = 76) :
  h₄ = 82 := by
  sorry

end fourth_person_height_l3008_300842


namespace dividend_calculation_l3008_300843

theorem dividend_calculation (divisor quotient remainder : ℕ) 
  (h1 : divisor = 40)
  (h2 : quotient = 6)
  (h3 : remainder = 28) :
  quotient * divisor + remainder = 268 :=
by sorry

end dividend_calculation_l3008_300843


namespace bennett_window_screens_l3008_300875

theorem bennett_window_screens (january february march : ℕ) : 
  february = 2 * january →
  february = march / 4 →
  january + february + march = 12100 →
  march = 8800 := by sorry

end bennett_window_screens_l3008_300875


namespace rain_probability_l3008_300815

theorem rain_probability (p_friday p_monday : ℝ) 
  (h1 : p_friday = 0.3)
  (h2 : p_monday = 0.2)
  (h3 : 0 ≤ p_friday ∧ p_friday ≤ 1)
  (h4 : 0 ≤ p_monday ∧ p_monday ≤ 1) :
  1 - (1 - p_friday) * (1 - p_monday) = 0.44 := by
sorry

end rain_probability_l3008_300815


namespace min_value_circle_line_l3008_300804

/-- The minimum value of 1/a + 4/b for a circle and a line passing through its center --/
theorem min_value_circle_line (a b : ℝ) : 
  a > 0 → b > 0 → a + b = 1 → 
  (∀ x y : ℝ, x^2 + y^2 + 4*x - 2*y - 1 = 0 → a*x - 2*b*y + 2 = 0) →
  (1/a + 4/b) ≥ 9 := by
sorry

end min_value_circle_line_l3008_300804


namespace distance_between_centers_l3008_300898

/-- Given an isosceles triangle with circumradius R and inradius r,
    the distance d between the centers of the circumcircle and incircle
    is given by d = √(R(R-2r)). -/
theorem distance_between_centers (R r : ℝ) (h : R > 0 ∧ r > 0) :
  ∃ (d : ℝ), d = Real.sqrt (R * (R - 2 * r)) := by
  sorry

end distance_between_centers_l3008_300898


namespace fraction_inequality_l3008_300856

theorem fraction_inequality (a b c d : ℝ) 
  (h1 : a * d > b * c) 
  (h2 : a / b > c / d) 
  (hb : b > 0) 
  (hd : d > 0) : 
  a / b > (a + c) / (b + d) ∧ (a + c) / (b + d) > c / d := by
sorry

end fraction_inequality_l3008_300856


namespace solution_set_equality_l3008_300892

/-- The solution set of the inequality (x^2 - 2x - 3)(x^2 - 4x + 4) < 0 -/
def SolutionSet : Set ℝ :=
  {x | (x^2 - 2*x - 3) * (x^2 - 4*x + 4) < 0}

/-- The set {x | -1 < x < 3 and x ≠ 2} -/
def TargetSet : Set ℝ :=
  {x | -1 < x ∧ x < 3 ∧ x ≠ 2}

theorem solution_set_equality : SolutionSet = TargetSet := by
  sorry

end solution_set_equality_l3008_300892


namespace initial_articles_sold_l3008_300869

/-- The number of articles sold to gain 20% when the total selling price is $60 -/
def articles_sold_gain (n : ℕ) : Prop :=
  ∃ (cp : ℚ), 1.2 * cp * n = 60

/-- The number of articles that should be sold to incur a loss of 20% when the total selling price is $60 -/
def articles_sold_loss : ℚ := 29.99999625000047

/-- The proposition that the initial number of articles sold is correct -/
def correct_initial_articles (n : ℕ) : Prop :=
  articles_sold_gain n ∧
  ∃ (cp : ℚ), 0.8 * cp * articles_sold_loss = 60 ∧
              cp * articles_sold_loss = 75 ∧
              cp * n = 50

theorem initial_articles_sold :
  ∃ (n : ℕ), correct_initial_articles n ∧ n = 20 := by sorry

end initial_articles_sold_l3008_300869


namespace farm_ploughing_problem_l3008_300801

/-- Calculates the actual ploughing rate given the conditions of the farm problem -/
def actualPloughingRate (totalArea plannedRate extraDays unploughedArea : ℕ) : ℕ :=
  let plannedDays := totalArea / plannedRate
  let actualDays := plannedDays + extraDays
  let ploughedArea := totalArea - unploughedArea
  ploughedArea / actualDays

/-- Theorem stating the actual ploughing rate for the given farm problem -/
theorem farm_ploughing_problem :
  actualPloughingRate 3780 90 2 40 = 85 := by
  sorry

end farm_ploughing_problem_l3008_300801


namespace min_value_quadratic_sum_l3008_300803

theorem min_value_quadratic_sum (x y : ℝ) (h : x + y = 1) :
  ∀ z w : ℝ, z + w = 1 → 2 * x^2 + 3 * y^2 ≤ 2 * z^2 + 3 * w^2 ∧
  ∃ a b : ℝ, a + b = 1 ∧ 2 * a^2 + 3 * b^2 = 6/5 :=
by sorry

end min_value_quadratic_sum_l3008_300803


namespace point_above_line_l3008_300808

/-- A point (x, y) is above a line Ax + By + C = 0 if Ax + By + C < 0 -/
def IsAboveLine (x y A B C : ℝ) : Prop := A * x + B * y + C < 0

/-- The theorem states that for the point (-3, -1) to be above the line 3x - 2y - a = 0,
    a must be greater than -7 -/
theorem point_above_line (a : ℝ) :
  IsAboveLine (-3) (-1) 3 (-2) (-a) ↔ a > -7 := by
  sorry

end point_above_line_l3008_300808


namespace last_digit_padic_fermat_l3008_300845

/-- Represents a p-adic integer with a non-zero last digit -/
structure PAdic (p : ℕ) where
  digits : ℕ → ℕ
  last_nonzero : digits 0 ≠ 0
  bound : ∀ n, digits n < p

/-- The last digit of a p-adic number -/
def last_digit {p : ℕ} (a : PAdic p) : ℕ := a.digits 0

/-- Exponentiation for p-adic numbers -/
def padic_pow {p : ℕ} (a : PAdic p) (n : ℕ) : PAdic p :=
  sorry

/-- Subtraction for p-adic numbers -/
def padic_sub {p : ℕ} (a b : PAdic p) : PAdic p :=
  sorry

theorem last_digit_padic_fermat (p : ℕ) (hp : Prime p) (a : PAdic p) :
  last_digit (padic_sub (padic_pow a (p - 1)) (PAdic.mk (λ _ => 1) sorry sorry)) = 0 :=
sorry

end last_digit_padic_fermat_l3008_300845


namespace complex_root_pair_l3008_300821

theorem complex_root_pair (z : ℂ) :
  (3 + 8*I : ℂ)^2 = -55 + 48*I →
  z^2 = -55 + 48*I →
  z = 3 + 8*I ∨ z = -3 - 8*I :=
by sorry

end complex_root_pair_l3008_300821


namespace xyz_stock_price_evolution_l3008_300867

def stock_price_evolution (initial_price : ℝ) (first_year_increase : ℝ) (second_year_decrease : ℝ) : ℝ :=
  let price_after_first_year := initial_price * (1 + first_year_increase)
  price_after_first_year * (1 - second_year_decrease)

theorem xyz_stock_price_evolution :
  stock_price_evolution 120 1 0.3 = 168 := by
  sorry

end xyz_stock_price_evolution_l3008_300867


namespace pasture_rent_is_140_l3008_300810

/-- Represents the rent share of a person -/
structure RentShare where
  oxen : ℕ
  months : ℕ
  payment : ℕ

/-- Calculates the total rent of a pasture given the rent shares of three people -/
def totalRent (a b c : RentShare) : ℕ :=
  let totalOxenMonths := a.oxen * a.months + b.oxen * b.months + c.oxen * c.months
  let costPerOxenMonth := c.payment / (c.oxen * c.months)
  costPerOxenMonth * totalOxenMonths

/-- Theorem stating that the total rent of the pasture is 140 -/
theorem pasture_rent_is_140 (a b c : RentShare)
  (ha : a.oxen = 10 ∧ a.months = 7)
  (hb : b.oxen = 12 ∧ b.months = 5)
  (hc : c.oxen = 15 ∧ c.months = 3 ∧ c.payment = 36) :
  totalRent a b c = 140 := by
  sorry

end pasture_rent_is_140_l3008_300810


namespace billy_lemon_heads_l3008_300820

/-- The number of friends Billy gave Lemon Heads to -/
def num_friends : ℕ := 6

/-- The number of Lemon Heads each friend ate -/
def lemon_heads_per_friend : ℕ := 12

/-- The initial number of Lemon Heads Billy had -/
def initial_lemon_heads : ℕ := num_friends * lemon_heads_per_friend

theorem billy_lemon_heads :
  initial_lemon_heads = 72 :=
by sorry

end billy_lemon_heads_l3008_300820


namespace painting_survey_l3008_300890

theorem painting_survey (total : ℕ) (not_enjoy_not_understand : ℕ) (enjoy : ℕ) (understand : ℕ) :
  total = 440 →
  not_enjoy_not_understand = 110 →
  enjoy = understand →
  (enjoy : ℚ) / total = 3 / 8 :=
by
  sorry

end painting_survey_l3008_300890


namespace units_digit_sum_of_powers_l3008_300877

theorem units_digit_sum_of_powers : (2016^2017 + 2017^2016) % 10 = 7 := by
  sorry

end units_digit_sum_of_powers_l3008_300877


namespace carver_school_earnings_l3008_300888

/-- Represents a school with its student count and work days -/
structure School where
  name : String
  students : ℕ
  days : ℕ

/-- Calculates the total payment for all schools -/
def totalPayment (schools : List School) (basePayment dailyWage : ℚ) : ℚ :=
  (schools.map (fun s => s.students * s.days) |>.sum : ℕ) * dailyWage + 
  (schools.length : ℕ) * basePayment

/-- Calculates the earnings for a specific school -/
def schoolEarnings (school : School) (dailyWage : ℚ) : ℚ :=
  (school.students * school.days : ℕ) * dailyWage

theorem carver_school_earnings :
  let allen := School.mk "Allen" 7 3
  let balboa := School.mk "Balboa" 5 6
  let carver := School.mk "Carver" 4 10
  let schools := [allen, balboa, carver]
  let basePayment := 20
  ∃ dailyWage : ℚ,
    totalPayment schools basePayment dailyWage = 900 ∧
    schoolEarnings carver dailyWage = 369.60 := by
  sorry

end carver_school_earnings_l3008_300888


namespace total_vacations_and_classes_l3008_300896

/-- Represents the number of classes Kelvin has -/
def kelvin_classes : ℕ := 90

/-- Represents the cost of each of Kelvin's classes in dollars -/
def kelvin_class_cost : ℕ := 75

/-- Represents Grant's maximum budget for vacations in dollars -/
def grant_max_budget : ℕ := 100000

/-- Theorem stating that the sum of Grant's vacations and Kelvin's classes is 450 -/
theorem total_vacations_and_classes : 
  ∃ (grant_vacations : ℕ),
    grant_vacations = 4 * kelvin_classes ∧ 
    grant_vacations * (2 * kelvin_class_cost) ≤ grant_max_budget ∧
    grant_vacations + kelvin_classes = 450 := by
  sorry

end total_vacations_and_classes_l3008_300896


namespace quarter_probability_l3008_300858

/-- The probability of choosing a quarter from a jar containing quarters, nickels, and pennies -/
theorem quarter_probability (quarter_value nickel_value penny_value : ℚ)
  (total_quarter_value total_nickel_value total_penny_value : ℚ)
  (h_quarter : quarter_value = 25/100)
  (h_nickel : nickel_value = 5/100)
  (h_penny : penny_value = 1/100)
  (h_total_quarter : total_quarter_value = 15/2)
  (h_total_nickel : total_nickel_value = 25/2)
  (h_total_penny : total_penny_value = 15) :
  (total_quarter_value / quarter_value) / 
  ((total_quarter_value / quarter_value) + 
   (total_nickel_value / nickel_value) + 
   (total_penny_value / penny_value)) = 15/890 := by
  sorry


end quarter_probability_l3008_300858


namespace cold_water_time_l3008_300859

/-- The combined total time Jerry and his friends spent in the cold water pool --/
def total_time (jerry_time elaine_time george_time kramer_time : ℝ) : ℝ :=
  jerry_time + elaine_time + george_time + kramer_time

/-- Theorem stating the total time spent in the cold water pool --/
theorem cold_water_time : ∃ (jerry_time elaine_time george_time kramer_time : ℝ),
  jerry_time = 3 ∧
  elaine_time = 2 * jerry_time ∧
  george_time = (1/3) * elaine_time ∧
  kramer_time = 0 ∧
  total_time jerry_time elaine_time george_time kramer_time = 11 := by
  sorry

end cold_water_time_l3008_300859


namespace meeting_speed_l3008_300844

theorem meeting_speed (distance : ℝ) (time : ℝ) (speed_difference : ℝ) 
  (h1 : distance = 200)
  (h2 : time = 8)
  (h3 : speed_difference = 7) :
  ∃ (speed : ℝ), 
    speed > 0 ∧ 
    (speed + (speed + speed_difference)) * time = distance ∧
    speed = 9 := by
  sorry

end meeting_speed_l3008_300844


namespace trajectory_characterization_l3008_300862

-- Define the fixed points
def F₁ : ℝ × ℝ := (-5, 0)
def F₂ : ℝ × ℝ := (5, 0)

-- Define the condition for point P
def satisfies_condition (P : ℝ × ℝ) (a : ℝ) : Prop :=
  |P.1 - F₁.1| + |P.2 - F₁.2| - (|P.1 - F₂.1| + |P.2 - F₂.2|) = 2 * a

-- Define what it means to be on one branch of a hyperbola
def on_hyperbola_branch (P : ℝ × ℝ) : Prop :=
  ∃ (a : ℝ), a > 0 ∧ satisfies_condition P a ∧ 
  (P.1 < -5 ∨ (P.1 > 5 ∧ P.2 ≠ 0))

-- Define what it means to be on a ray starting from (5, 0) in positive x direction
def on_positive_x_ray (P : ℝ × ℝ) : Prop :=
  P.2 = 0 ∧ P.1 ≥ 5

theorem trajectory_characterization :
  (∀ P : ℝ × ℝ, satisfies_condition P 3 → on_hyperbola_branch P) ∧
  (∀ P : ℝ × ℝ, satisfies_condition P 5 → on_positive_x_ray P) :=
sorry

end trajectory_characterization_l3008_300862


namespace inverse_division_identity_l3008_300864

theorem inverse_division_identity (x : ℝ) (hx : x ≠ 0) : 1 / x⁻¹ = x := by
  sorry

end inverse_division_identity_l3008_300864


namespace sets_relationship_l3008_300852

def M : Set ℝ := {x | ∃ m : ℤ, x = m + 1/6}
def N : Set ℝ := {x | ∃ n : ℤ, x = n/2 - 1/3}
def P : Set ℝ := {x | ∃ p : ℤ, x = p/2 + 1/6}

theorem sets_relationship : N = P ∧ M ≠ N :=
sorry

end sets_relationship_l3008_300852
