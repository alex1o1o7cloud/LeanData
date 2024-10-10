import Mathlib

namespace polynomial_evaluation_l1445_144575

def P (x : ℤ) : ℤ :=
  x^15 - 2008*x^14 + 2008*x^13 - 2008*x^12 + 2008*x^11 - 2008*x^10 + 2008*x^9 - 2008*x^8 + 2008*x^7 - 2008*x^6 + 2008*x^5 - 2008*x^4 + 2008*x^3 - 2008*x^2 + 2008*x

theorem polynomial_evaluation : P 2007 = 2007 := by
  sorry

end polynomial_evaluation_l1445_144575


namespace intersection_of_A_and_B_l1445_144542

def A : Set ℝ := {x | x^2 - 2*x > 0}

def B : Set ℝ := {x | (x+1)/(x-1) ≤ 0}

theorem intersection_of_A_and_B : A ∩ B = {x : ℝ | -1 ≤ x ∧ x < 0} := by sorry

end intersection_of_A_and_B_l1445_144542


namespace infinitely_many_primes_mod_3_eq_2_l1445_144508

theorem infinitely_many_primes_mod_3_eq_2 : 
  Set.Infinite {p : ℕ | Nat.Prime p ∧ p % 3 = 2} := by
  sorry

end infinitely_many_primes_mod_3_eq_2_l1445_144508


namespace probability_one_more_red_eq_three_eighths_l1445_144549

/-- Represents the color of a ball -/
inductive BallColor
  | Red
  | White

/-- Represents the outcome of three draws -/
def ThreeDraw := (BallColor × BallColor × BallColor)

/-- The set of all possible outcomes when drawing a ball three times with replacement -/
def allOutcomes : Finset ThreeDraw := sorry

/-- Predicate to check if an outcome has one more red ball than white balls -/
def hasOneMoreRed (draw : ThreeDraw) : Prop := sorry

/-- The set of favorable outcomes (one more red than white) -/
def favorableOutcomes : Finset ThreeDraw := sorry

/-- The probability of drawing the red ball one more time than the white ball -/
def probabilityOneMoreRed : ℚ := (favorableOutcomes.card : ℚ) / (allOutcomes.card : ℚ)

/-- Theorem: The probability of drawing the red ball one more time than the white ball is 3/8 -/
theorem probability_one_more_red_eq_three_eighths : 
  probabilityOneMoreRed = 3 / 8 := by sorry

end probability_one_more_red_eq_three_eighths_l1445_144549


namespace secret_spread_theorem_l1445_144503

/-- The number of people who know the secret after n days -/
def secret_spread (n : ℕ) : ℕ :=
  (3^(n+1) - 1) / 2

/-- The day of the week given the number of days since Monday -/
def day_of_week (n : ℕ) : String :=
  match n % 7 with
  | 0 => "Monday"
  | 1 => "Tuesday"
  | 2 => "Wednesday"
  | 3 => "Thursday"
  | 4 => "Friday"
  | 5 => "Saturday"
  | _ => "Sunday"

theorem secret_spread_theorem :
  ∃ n : ℕ, secret_spread n ≥ 2186 ∧
           ∀ m : ℕ, m < n → secret_spread m < 2186 ∧
           day_of_week n = "Sunday" :=
by
  sorry

end secret_spread_theorem_l1445_144503


namespace f_max_min_on_interval_l1445_144518

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sqrt 3 * Real.sin x * Real.cos x + 2 * (Real.cos x)^2 - 1

theorem f_max_min_on_interval :
  let a := 0
  let b := Real.pi / 2
  ∃ (x_max x_min : ℝ), x_max ∈ Set.Icc a b ∧ x_min ∈ Set.Icc a b ∧
    (∀ x ∈ Set.Icc a b, f x ≤ f x_max) ∧
    (∀ x ∈ Set.Icc a b, f x_min ≤ f x) ∧
    f x_max = 2 ∧ f x_min = -1 :=
sorry

end f_max_min_on_interval_l1445_144518


namespace inequality_of_means_l1445_144510

theorem inequality_of_means (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  (a + a^2) / 2 > a^(3/2) ∧ a^(3/2) > 2 * a^2 / (1 + a) := by
  sorry

end inequality_of_means_l1445_144510


namespace trains_crossing_time_trains_crossing_time_approx_9_seconds_l1445_144506

/-- Time for trains to cross each other -/
theorem trains_crossing_time (train1_length train2_length : ℝ) 
  (train1_speed train2_speed : ℝ) : ℝ :=
  let total_length := train1_length + train2_length
  let relative_speed_kmh := train1_speed + train2_speed
  let relative_speed_ms := relative_speed_kmh * 1000 / 3600
  total_length / relative_speed_ms

/-- Proof that the time for the trains to cross is approximately 9 seconds -/
theorem trains_crossing_time_approx_9_seconds : 
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.1 ∧ 
  |trains_crossing_time 120.00001 380.03999 120 80 - 9| < ε :=
by
  sorry

end trains_crossing_time_trains_crossing_time_approx_9_seconds_l1445_144506


namespace treasure_chest_value_l1445_144543

def base7_to_base10 (n : List Nat) : Nat :=
  n.enum.foldr (fun (i, d) acc => acc + d * (7 ^ i)) 0

theorem treasure_chest_value : 
  let coins := [6, 4, 3, 5]
  let gems := [1, 2, 5, 6]
  let maps := [0, 2, 3]
  base7_to_base10 coins + base7_to_base10 gems + base7_to_base10 maps = 4305 := by
sorry

#eval base7_to_base10 [6, 4, 3, 5] + base7_to_base10 [1, 2, 5, 6] + base7_to_base10 [0, 2, 3]

end treasure_chest_value_l1445_144543


namespace sum_of_series_equals_two_l1445_144546

/-- The sum of the infinite series ∑(n=1 to ∞) (4n-1)/3^n is equal to 2 -/
theorem sum_of_series_equals_two :
  let series := fun n : ℕ => (4 * n - 1) / (3 ^ n : ℝ)
  (∑' n, series n) = 2 := by
  sorry

end sum_of_series_equals_two_l1445_144546


namespace orange_savings_percentage_l1445_144538

/-- Calculates the percentage of money saved when receiving free items instead of buying them -/
theorem orange_savings_percentage 
  (family_size : ℕ) 
  (planned_spending : ℝ) 
  (orange_price : ℝ) : 
  family_size = 4 → 
  planned_spending = 15 → 
  orange_price = 1.5 → 
  (family_size * orange_price / planned_spending) * 100 = 40 := by
sorry

end orange_savings_percentage_l1445_144538


namespace robotics_club_neither_cs_nor_electronics_l1445_144571

/-- The number of students in the robotics club who take neither computer science nor electronics -/
theorem robotics_club_neither_cs_nor_electronics :
  let total_students : ℕ := 60
  let cs_students : ℕ := 40
  let electronics_students : ℕ := 35
  let both_cs_and_electronics : ℕ := 25
  let neither_cs_nor_electronics : ℕ := total_students - (cs_students + electronics_students - both_cs_and_electronics)
  neither_cs_nor_electronics = 10 := by
sorry

end robotics_club_neither_cs_nor_electronics_l1445_144571


namespace pancake_cooking_theorem_l1445_144591

/-- Represents the minimum time needed to cook a given number of pancakes -/
def min_cooking_time (num_pancakes : ℕ) : ℕ :=
  sorry

/-- The pancake cooking theorem -/
theorem pancake_cooking_theorem :
  let pan_capacity : ℕ := 2
  let cooking_time_per_pancake : ℕ := 2
  let num_pancakes : ℕ := 3
  min_cooking_time num_pancakes = 3 :=
sorry

end pancake_cooking_theorem_l1445_144591


namespace minimum_selling_price_chocolate_manufacturer_l1445_144595

/-- Calculates the minimum selling price per unit to achieve a desired monthly profit -/
def minimum_selling_price (units : ℕ) (cost_per_unit : ℚ) (desired_profit : ℚ) : ℚ :=
  (units * cost_per_unit + desired_profit) / units

theorem minimum_selling_price_chocolate_manufacturer :
  let units : ℕ := 400
  let cost_per_unit : ℚ := 40
  let desired_profit : ℚ := 40000
  minimum_selling_price units cost_per_unit desired_profit = 140 := by
  sorry

end minimum_selling_price_chocolate_manufacturer_l1445_144595


namespace slope_one_points_l1445_144540

theorem slope_one_points (a : ℝ) : 
  let A : ℝ × ℝ := (-a, 3)
  let B : ℝ × ℝ := (5, -a)
  let slope := (B.2 - A.2) / (B.1 - A.1)
  slope = 1 → a = -4 := by
sorry

end slope_one_points_l1445_144540


namespace equivalent_expression_l1445_144585

theorem equivalent_expression (x : ℝ) (h : x < 0) : 
  Real.sqrt (x / (1 - (x - 2) / x)) = -x / Real.sqrt 2 := by
  sorry

end equivalent_expression_l1445_144585


namespace three_propositions_are_true_l1445_144592

-- Define the concept of a line
def Line : Type := sorry

-- Define the concept of a point
def Point : Type := sorry

-- Define the relation of two lines being skew
def are_skew (a b : Line) : Prop := sorry

-- Define the relation of a line intersecting another line at a point
def intersects_at (l1 l2 : Line) (p : Point) : Prop := sorry

-- Define the relation of two lines being parallel
def are_parallel (l1 l2 : Line) : Prop := sorry

-- Define the concept of a plane
def Plane : Type := sorry

-- Define the relation of two lines determining a plane
def determine_plane (l1 l2 : Line) (p : Plane) : Prop := sorry

theorem three_propositions_are_true :
  -- Proposition 1
  (∀ (a b c d : Line) (E F G H : Point),
    are_skew a b ∧
    intersects_at c a E ∧ intersects_at c b F ∧
    intersects_at d a G ∧ intersects_at d b H ∧
    E ≠ F ∧ E ≠ G ∧ E ≠ H ∧ F ≠ G ∧ F ≠ H ∧ G ≠ H →
    are_skew c d) ∧
  -- Proposition 2
  (∀ (a b l : Line),
    are_skew a b →
    are_parallel l a →
    ¬(are_parallel l b)) ∧
  -- Proposition 3
  (∀ (a b l : Line),
    are_skew a b →
    (∃ (P Q : Point), intersects_at l a P ∧ intersects_at l b Q) →
    ∃ (p1 p2 : Plane), determine_plane a l p1 ∧ determine_plane b l p2) :=
by sorry

end three_propositions_are_true_l1445_144592


namespace age_difference_is_fifty_l1445_144504

/-- Represents the ages of family members in the year 2000 -/
structure FamilyAges where
  daughter : ℕ
  son : ℕ
  mother : ℕ
  father : ℕ

/-- Conditions given in the problem -/
def familyConditions (ages : FamilyAges) : Prop :=
  ages.mother = 4 * ages.daughter ∧
  ages.father = 6 * ages.son ∧
  ages.son = (3 * ages.daughter) / 2 ∧
  ages.father + 10 = 2 * (ages.mother + 10)

/-- The theorem to be proved -/
theorem age_difference_is_fifty (ages : FamilyAges) 
  (h : familyConditions ages) : ages.father - ages.mother = 50 := by
  sorry

#check age_difference_is_fifty

end age_difference_is_fifty_l1445_144504


namespace fractional_equation_solution_l1445_144588

theorem fractional_equation_solution :
  ∃ x : ℚ, (3 / 2 : ℚ) - (2 * x) / (3 * x - 1) = 7 / (6 * x - 2) ∧ x = 2 := by
  sorry

end fractional_equation_solution_l1445_144588


namespace gcd_lcm_product_90_150_l1445_144545

theorem gcd_lcm_product_90_150 : Nat.gcd 90 150 * Nat.lcm 90 150 = 13500 := by
  sorry

end gcd_lcm_product_90_150_l1445_144545


namespace f_composition_inequality_l1445_144519

-- Define the piecewise function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≥ 2 then x^2 + 2*a*x else 2^x + 1

-- State the theorem
theorem f_composition_inequality (a : ℝ) :
  (f a (f a 1) > 3 * a^2) ↔ (-1 < a ∧ a < 3) := by
  sorry

end f_composition_inequality_l1445_144519


namespace average_difference_l1445_144541

theorem average_difference (x : ℝ) : 
  (10 + 70 + x) / 3 = (20 + 40 + 60) / 3 - 7 → x = 19 := by
  sorry

end average_difference_l1445_144541


namespace problem_solution_l1445_144539

theorem problem_solution (k : ℝ) (h1 : k ≠ 0) :
  (∀ x : ℝ, (x^2 - k) * (x + k) = x^3 + k * (x^2 - x - 8)) → k = 8 := by
  sorry

end problem_solution_l1445_144539


namespace integer_set_range_l1445_144533

theorem integer_set_range (a : ℝ) : 
  a ≤ 1 →
  (∃ (x y z : ℤ), x ≠ y ∧ y ≠ z ∧ x ≠ z ∧
    (↑x : ℝ) ∈ Set.Icc a (2 - a) ∧
    (↑y : ℝ) ∈ Set.Icc a (2 - a) ∧
    (↑z : ℝ) ∈ Set.Icc a (2 - a) ∧
    (∀ (w : ℤ), (↑w : ℝ) ∈ Set.Icc a (2 - a) → w = x ∨ w = y ∨ w = z)) →
  -1 < a ∧ a ≤ 0 :=
by sorry

end integer_set_range_l1445_144533


namespace base_b_not_divisible_by_five_l1445_144590

theorem base_b_not_divisible_by_five (b : ℤ) : b ∈ ({4, 5, 7, 8, 10} : Set ℤ) →
  (3 * b^3 - b^2 + b - 1) % 5 ≠ 0 ↔ b = 4 ∨ b = 7 := by
  sorry

end base_b_not_divisible_by_five_l1445_144590


namespace tom_final_balance_l1445_144547

def calculate_final_balance (initial_allowance : ℚ) (extra_earning : ℚ) (final_spending : ℚ) : ℚ :=
  let week1_balance := initial_allowance - initial_allowance / 3
  let week2_balance := week1_balance - week1_balance / 4
  let week3_balance_before_spending := week2_balance + extra_earning
  let week3_balance_after_spending := week3_balance_before_spending / 2
  week3_balance_after_spending - final_spending

theorem tom_final_balance :
  calculate_final_balance 12 5 3 = (5/2 : ℚ) := by sorry

end tom_final_balance_l1445_144547


namespace uf_games_before_championship_l1445_144582

/-- The number of games UF played before the championship game -/
def n : ℕ := sorry

/-- The total points UF scored in previous games -/
def total_points : ℕ := 720

/-- UF's score in the championship game -/
def championship_score : ℕ := total_points / (2 * n) - 2

/-- UF's opponent's score in the championship game -/
def opponent_score : ℕ := 11

theorem uf_games_before_championship : 
  (total_points / n = championship_score + 2) ∧ 
  (championship_score = opponent_score + 2) ∧
  (n = 24) := by sorry

end uf_games_before_championship_l1445_144582


namespace geometric_series_sum_l1445_144502

theorem geometric_series_sum : ∑' i, (2/3:ℝ)^i = 2 := by sorry

end geometric_series_sum_l1445_144502


namespace blue_surface_area_fraction_l1445_144596

theorem blue_surface_area_fraction (edge_length : ℕ) (small_cube_count : ℕ) 
  (green_count : ℕ) (blue_count : ℕ) :
  edge_length = 4 →
  small_cube_count = 64 →
  green_count = 44 →
  blue_count = 20 →
  (∃ (blue_exposed : ℕ), 
    blue_exposed ≤ blue_count ∧ 
    blue_exposed * 1 = (edge_length ^ 2 * 6) / 8) :=
by sorry

end blue_surface_area_fraction_l1445_144596


namespace fraction_difference_squared_l1445_144563

theorem fraction_difference_squared (a b : ℝ) (h : a ≠ 0 ∧ b ≠ 0) 
  (h1 : ∀ x y : ℝ, x ≠ 0 → y ≠ 0 → 1 / x - 1 / y = 1 / (x + y)) : 
  1 / a^2 - 1 / b^2 = 1 / (a * b) := by
  sorry

end fraction_difference_squared_l1445_144563


namespace glen_animals_theorem_l1445_144586

theorem glen_animals_theorem (f t c r : ℕ) : 
  f = (5 * t) / 2 → 
  c = 3 * f → 
  r = 4 * c → 
  ∀ t : ℕ, f + t + c + r ≠ 108 :=
by
  sorry

end glen_animals_theorem_l1445_144586


namespace zoo_feeding_theorem_l1445_144565

/-- Represents the number of bread and treats brought by each person -/
structure BreadAndTreats :=
  (bread : ℕ)
  (treats : ℕ)

/-- Calculates the total number of bread and treats -/
def totalItems (items : List BreadAndTreats) : ℕ :=
  (items.map (λ i => i.bread + i.treats)).sum

/-- Calculates the cost per pet -/
def costPerPet (totalBread totalTreats : ℕ) (x y : ℚ) (z : ℕ) : ℚ :=
  (totalBread * x + totalTreats * y) / z

theorem zoo_feeding_theorem 
  (jane_bread : ℕ) (jane_treats : ℕ)
  (wanda_bread : ℕ) (wanda_treats : ℕ)
  (carla_bread : ℕ) (carla_treats : ℕ)
  (peter_bread : ℕ) (peter_treats : ℕ)
  (x y : ℚ) (z : ℕ) :
  jane_bread = (75 * jane_treats) / 100 →
  wanda_treats = jane_treats / 2 →
  wanda_bread = 3 * wanda_treats →
  wanda_bread = 90 →
  carla_treats = (5 * carla_bread) / 2 →
  carla_bread = 40 →
  peter_bread = 2 * peter_treats →
  peter_bread + peter_treats = 140 →
  let items := [
    BreadAndTreats.mk jane_bread jane_treats,
    BreadAndTreats.mk wanda_bread wanda_treats,
    BreadAndTreats.mk carla_bread carla_treats,
    BreadAndTreats.mk peter_bread peter_treats
  ]
  totalItems items = 427 ∧
  costPerPet 235 192 x y z = (235 * x + 192 * y) / z :=
by sorry


end zoo_feeding_theorem_l1445_144565


namespace lateralEdgeAngle_specific_pyramid_l1445_144536

/-- A regular truncated quadrangular pyramid -/
structure TruncatedPyramid where
  upperBaseSide : ℝ
  lowerBaseSide : ℝ
  height : ℝ
  lateralSurfaceArea : ℝ

/-- The angle between the lateral edge and the base plane of a truncated pyramid -/
def lateralEdgeAngle (p : TruncatedPyramid) : ℝ := sorry

/-- Theorem: The angle between the lateral edge and the base plane of a specific truncated pyramid -/
theorem lateralEdgeAngle_specific_pyramid :
  ∀ (p : TruncatedPyramid),
    p.lowerBaseSide = 5 * p.upperBaseSide →
    p.lateralSurfaceArea = p.height ^ 2 →
    lateralEdgeAngle p = Real.arctan (Real.sqrt (9 + 3 * Real.sqrt 10)) := by
  sorry

end lateralEdgeAngle_specific_pyramid_l1445_144536


namespace arithmetic_sequence_sum_divisibility_l1445_144559

-- Define the arithmetic sequence
def arithmeticSequence (a₁ aₙ d : ℕ) : List ℕ :=
  let n := (aₙ - a₁) / d + 1
  List.range n |>.map (λ i => a₁ + i * d)

-- Define the sum of a list of natural numbers
def sumList (l : List ℕ) : ℕ :=
  l.foldl (· + ·) 0

-- State the theorem
theorem arithmetic_sequence_sum_divisibility :
  let seq := arithmeticSequence 3 251 8
  let sum := sumList seq
  sum % 8 = 0 := by sorry

end arithmetic_sequence_sum_divisibility_l1445_144559


namespace equation_solution_l1445_144554

theorem equation_solution (x y : ℝ) (h : x / (x - 1) = (y^2 + 2*y - 1) / (y^2 + 2*y - 2)) :
  x = y^2 + 2*y - 1 := by
  sorry

end equation_solution_l1445_144554


namespace nested_fraction_equality_l1445_144527

theorem nested_fraction_equality : 
  2 + 1 / (2 + 1 / (2 + 1 / 2)) = 29 / 12 := by
  sorry

end nested_fraction_equality_l1445_144527


namespace son_age_is_22_l1445_144584

/-- Given a man and his son, where:
    1. The man is 24 years older than his son
    2. In two years, the man's age will be twice the age of his son
    This theorem proves that the present age of the son is 22 years. -/
theorem son_age_is_22 (man_age son_age : ℕ) : 
  man_age = son_age + 24 →
  man_age + 2 = 2 * (son_age + 2) →
  son_age = 22 := by
  sorry

end son_age_is_22_l1445_144584


namespace parabola_translation_l1445_144537

/-- A parabola in the Cartesian coordinate system. -/
structure Parabola where
  a : ℝ
  h : ℝ
  k : ℝ

/-- The equation of a parabola in the form y = a(x-h)^2 + k. -/
def parabola_equation (p : Parabola) (x : ℝ) : ℝ :=
  p.a * (x - p.h)^2 + p.k

/-- The translation of a parabola. -/
def translate (p : Parabola) (dx dy : ℝ) : Parabola :=
  { a := p.a, h := p.h + dx, k := p.k + dy }

theorem parabola_translation (p1 p2 : Parabola) :
  p1.a = 1/2 ∧ p1.h = 0 ∧ p1.k = -1 ∧
  p2.a = 1/2 ∧ p2.h = 4 ∧ p2.k = 2 →
  p2 = translate p1 4 3 :=
sorry

end parabola_translation_l1445_144537


namespace correct_statements_l1445_144557

theorem correct_statements :
  (∀ x : ℝ, x < 0 → x^3 < x) ∧
  (∀ x : ℝ, x^3 > 0 → x > 0) ∧
  (∀ x : ℝ, x > 1 → x^3 > x) :=
by sorry

end correct_statements_l1445_144557


namespace u_closed_under_multiplication_l1445_144534

def u : Set ℕ := {n : ℕ | ∃ m : ℕ, n = m * m ∧ m > 0}

theorem u_closed_under_multiplication :
  ∀ x y : ℕ, x ∈ u → y ∈ u → (x * y) ∈ u :=
by
  sorry

end u_closed_under_multiplication_l1445_144534


namespace ryan_project_average_funding_l1445_144505

/-- The average amount each person funds to Ryan's project -/
def average_funding (total_goal : ℕ) (people : ℕ) (initial_funds : ℕ) : ℚ :=
  (total_goal - initial_funds : ℚ) / people

/-- Theorem: The average funding per person for Ryan's project is $10 -/
theorem ryan_project_average_funding :
  average_funding 1000 80 200 = 10 := by
  sorry

end ryan_project_average_funding_l1445_144505


namespace intersection_points_determine_a_l1445_144587

def curve_C₁ (a : ℝ) (x y : ℝ) : Prop :=
  (x - a)^2 + y^2 = a^2 ∧ 0 ≤ x ∧ x ≤ a

def curve_C₂ (x y : ℝ) : Prop :=
  x^2 / 4 + y^2 = 1

theorem intersection_points_determine_a :
  ∀ a : ℝ, a > 0 →
  ∃ A B : ℝ × ℝ,
    curve_C₁ a A.1 A.2 ∧
    curve_C₁ a B.1 B.2 ∧
    curve_C₂ A.1 A.2 ∧
    curve_C₂ B.1 B.2 ∧
    (A.1 - B.1)^2 + (A.2 - B.2)^2 = (4 * Real.sqrt 2 / 3)^2 →
    a = 1 := by
  sorry

end intersection_points_determine_a_l1445_144587


namespace intersection_A_complement_B_necessary_not_sufficient_condition_l1445_144515

-- Define sets A and B
def A : Set ℝ := {x | x^2 - 2*x - 8 ≤ 0}
def B (m : ℝ) : Set ℝ := {x | x^2 - 2*x + 1 - m^2 ≤ 0}

-- Theorem 1: Intersection of A and complement of B when m = 2
theorem intersection_A_complement_B :
  A ∩ (Set.univ \ B 2) = {x | -2 ≤ x ∧ x < -1 ∨ 3 < x ∧ x ≤ 4} := by sorry

-- Theorem 2: Necessary but not sufficient condition
theorem necessary_not_sufficient_condition :
  (∀ m > 0, (∀ x, x ∈ B m → x ∈ A) ∧ (∃ x, x ∈ A ∧ x ∉ B m)) ↔ 0 < m ∧ m ≤ 3 := by sorry

end intersection_A_complement_B_necessary_not_sufficient_condition_l1445_144515


namespace square_area_from_perimeter_l1445_144511

/-- Theorem: The area of a square with perimeter 32 feet is 64 square feet. -/
theorem square_area_from_perimeter (perimeter : ℝ) (area : ℝ) : 
  perimeter = 32 → area = (perimeter / 4) ^ 2 → area = 64 := by
  sorry

end square_area_from_perimeter_l1445_144511


namespace part_i_part_ii_l1445_144521

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x - 1| + |x + a|

-- Define the function g
def g (a : ℝ) (x : ℝ) : ℝ := f a x - |3 + a|

-- Part I
theorem part_i : 
  {x : ℝ | f 3 x > 6} = {x : ℝ | x < -4 ∨ x > 2} :=
sorry

-- Part II
theorem part_ii :
  (∃ x : ℝ, g a x = 0) → a ≥ -2 :=
sorry

end part_i_part_ii_l1445_144521


namespace resultant_polyhedron_edges_l1445_144576

-- Define the convex polyhedron S
structure ConvexPolyhedron :=
  (vertices : ℕ)
  (edges : ℕ)

-- Define the operation of intersecting S with planes
def intersect_with_planes (S : ConvexPolyhedron) (num_planes : ℕ) : ℕ :=
  S.edges * 2 + S.edges

-- Theorem statement
theorem resultant_polyhedron_edges 
  (S : ConvexPolyhedron) 
  (h1 : S.vertices = S.vertices) 
  (h2 : S.edges = 150) :
  intersect_with_planes S S.vertices = 450 := by
  sorry

end resultant_polyhedron_edges_l1445_144576


namespace taxi_charge_per_segment_l1445_144570

/-- Proves that the additional charge per 2/5 of a mile is $0.35 -/
theorem taxi_charge_per_segment (initial_fee : ℝ) (trip_distance : ℝ) (total_charge : ℝ) 
  (h1 : initial_fee = 2.35)
  (h2 : trip_distance = 3.6)
  (h3 : total_charge = 5.5) :
  (total_charge - initial_fee) / (trip_distance / (2/5)) = 0.35 := by
  sorry

end taxi_charge_per_segment_l1445_144570


namespace angle_count_in_plane_l1445_144524

/-- Given n points in a plane, this theorem proves the number of 0° and 180° angles formed. -/
theorem angle_count_in_plane (n : ℕ) : 
  let zero_angles := n * (n - 1) * (n - 2) / 3
  let straight_angles := n * (n - 1) * (n - 2) / 6
  let total_angles := n * (n - 1) * (n - 2) / 2
  (zero_angles : ℚ) + (straight_angles : ℚ) = (total_angles : ℚ) :=
by sorry

/-- The total number of angles formed by n points in a plane. -/
def N (n : ℕ) : ℕ := n * (n - 1) * (n - 2) / 2

/-- The number of 0° angles formed by n points in a plane. -/
def zero_angles (n : ℕ) : ℕ := n * (n - 1) * (n - 2) / 3

/-- The number of 180° angles formed by n points in a plane. -/
def straight_angles (n : ℕ) : ℕ := n * (n - 1) * (n - 2) / 6

end angle_count_in_plane_l1445_144524


namespace max_value_of_f_l1445_144520

noncomputable def f (x : ℝ) := 3 + Real.log x + 4 / Real.log x

theorem max_value_of_f :
  (∀ x : ℝ, 0 < x → x < 1 → f x ≤ -1) ∧
  (∃ x : ℝ, 0 < x ∧ x < 1 ∧ f x = -1) :=
sorry

end max_value_of_f_l1445_144520


namespace geometric_sequence_middle_term_l1445_144516

/-- Given a geometric sequence of real numbers 1, a₁, a₂, a₃, 4, prove that a₂ = 2 -/
theorem geometric_sequence_middle_term 
  (a₁ a₂ a₃ : ℝ) 
  (h : ∃ (r : ℝ), r ≠ 0 ∧ a₁ = r ∧ a₂ = r^2 ∧ a₃ = r^3 ∧ 4 = r^4) : 
  a₂ = 2 := by
sorry

end geometric_sequence_middle_term_l1445_144516


namespace candy_bar_cost_l1445_144558

theorem candy_bar_cost (total_bars : ℕ) (dave_bars : ℕ) (john_paid : ℚ) : 
  total_bars = 20 → 
  dave_bars = 6 → 
  john_paid = 21 → 
  (john_paid / (total_bars - dave_bars : ℚ)) = 1.5 := by
sorry

end candy_bar_cost_l1445_144558


namespace mountain_elevation_l1445_144560

/-- The relative elevation of a mountain given temperature information -/
theorem mountain_elevation (temp_decrease_rate : ℝ) (temp_summit temp_foot : ℝ) 
  (h1 : temp_decrease_rate = 0.7)
  (h2 : temp_summit = 14.1)
  (h3 : temp_foot = 26) :
  (temp_foot - temp_summit) / temp_decrease_rate * 100 = 1700 := by
  sorry

end mountain_elevation_l1445_144560


namespace subtracted_amount_l1445_144552

theorem subtracted_amount (x : ℝ) (h : x = 2.625) : 8 * x - 17 = 4 := by
  sorry

end subtracted_amount_l1445_144552


namespace acid_dilution_l1445_144551

/-- Proves that adding 15 ounces of pure water to 30 ounces of a 30% acid solution yields a 20% acid solution. -/
theorem acid_dilution (initial_volume : ℝ) (initial_concentration : ℝ) (water_added : ℝ) (final_concentration : ℝ) :
  initial_volume = 30 →
  initial_concentration = 0.3 →
  water_added = 15 →
  final_concentration = 0.2 →
  (initial_volume * initial_concentration) / (initial_volume + water_added) = final_concentration :=
by sorry

end acid_dilution_l1445_144551


namespace jackson_pbj_sandwiches_l1445_144501

/-- Calculates the number of peanut butter and jelly sandwiches eaten in a school year --/
def pbj_sandwiches_eaten (weeks : ℕ) (wed_holidays : ℕ) (fri_holidays : ℕ) 
  (ham_cheese_interval : ℕ) (wed_missed : ℕ) (fri_missed : ℕ) : ℕ :=
  let total_wed := weeks
  let total_fri := weeks
  let wed_after_holidays := total_wed - wed_holidays
  let fri_after_holidays := total_fri - fri_holidays
  let wed_after_missed := wed_after_holidays - wed_missed
  let fri_after_missed := fri_after_holidays - fri_missed
  let ham_cheese_weeks := weeks / ham_cheese_interval
  let pbj_wed := wed_after_missed - ham_cheese_weeks
  let pbj_fri := fri_after_missed - (2 * ham_cheese_weeks)
  pbj_wed + pbj_fri

theorem jackson_pbj_sandwiches :
  pbj_sandwiches_eaten 36 2 3 4 1 2 = 37 := by
  sorry

#eval pbj_sandwiches_eaten 36 2 3 4 1 2

end jackson_pbj_sandwiches_l1445_144501


namespace units_digit_47_power_47_l1445_144535

theorem units_digit_47_power_47 : 47^47 ≡ 3 [ZMOD 10] := by sorry

end units_digit_47_power_47_l1445_144535


namespace f_neg_two_eq_neg_twenty_two_l1445_144578

/-- The function f(x) = x^3 - 3x^2 + x -/
def f (x : ℝ) : ℝ := x^3 - 3*x^2 + x

/-- Theorem: The value of f(-2) is -22 -/
theorem f_neg_two_eq_neg_twenty_two : f (-2) = -22 := by
  sorry

end f_neg_two_eq_neg_twenty_two_l1445_144578


namespace parallel_to_y_axis_second_quadrant_equal_distance_l1445_144566

-- Define point P
def P (a : ℝ) : ℝ × ℝ := (2*a - 2, a + 5)

-- Define point Q
def Q : ℝ × ℝ := (4, 5)

-- Part 1
theorem parallel_to_y_axis (a : ℝ) :
  (P a).1 = Q.1 → P a = (4, 8) := by sorry

-- Part 2
theorem second_quadrant_equal_distance (a : ℝ) :
  (P a).1 < 0 ∧ (P a).2 > 0 ∧ -(P a).1 = (P a).2 → a^2023 + a^(1/3) = -2 := by sorry

end parallel_to_y_axis_second_quadrant_equal_distance_l1445_144566


namespace probability_two_kings_or_at_least_two_aces_l1445_144574

def standard_deck : ℕ := 52
def num_aces : ℕ := 4
def num_kings : ℕ := 4
def cards_drawn : ℕ := 3

def prob_two_kings : ℚ := (Nat.choose num_kings 2 * Nat.choose (standard_deck - num_kings) 1) / Nat.choose standard_deck cards_drawn

def prob_two_aces : ℚ := (Nat.choose num_aces 2 * Nat.choose (standard_deck - num_aces) 1) / Nat.choose standard_deck cards_drawn

def prob_three_aces : ℚ := Nat.choose num_aces 3 / Nat.choose standard_deck cards_drawn

def prob_at_least_two_aces : ℚ := prob_two_aces + prob_three_aces

theorem probability_two_kings_or_at_least_two_aces :
  prob_two_kings + prob_at_least_two_aces = 1090482 / 40711175 := by
  sorry

end probability_two_kings_or_at_least_two_aces_l1445_144574


namespace euler_totient_power_of_two_l1445_144580

theorem euler_totient_power_of_two (n : ℕ) : 
  Odd n → 
  ∃ k m : ℕ, Nat.totient n = 2^k ∧ Nat.totient (n+1) = 2^m → 
  ∃ p : ℕ, n + 1 = 2^p ∨ n = 5 := by
  sorry

end euler_totient_power_of_two_l1445_144580


namespace bridge_length_calculation_l1445_144597

/-- Calculates the length of a bridge given train parameters --/
theorem bridge_length_calculation (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time : ℝ) : 
  train_length = 130 →
  train_speed_kmh = 45 →
  crossing_time = 30 →
  (train_speed_kmh * 1000 / 3600) * crossing_time - train_length = 245 := by
  sorry

#check bridge_length_calculation

end bridge_length_calculation_l1445_144597


namespace parabola_focus_directrix_distance_l1445_144599

/-- The distance from the focus to the directrix of a parabola y^2 = 8x is 4 -/
theorem parabola_focus_directrix_distance :
  ∀ (x y : ℝ), y^2 = 8*x → 
  ∃ (f d : ℝ × ℝ), 
    (f.1 - d.1)^2 + (f.2 - d.2)^2 = 4^2 ∧
    (∀ (p : ℝ × ℝ), p.2^2 = 8*p.1 → 
      (p.1 - f.1)^2 + (p.2 - f.2)^2 = (p.1 - d.1)^2 + (p.2 - d.2)^2) :=
by sorry

end parabola_focus_directrix_distance_l1445_144599


namespace lock_problem_l1445_144561

def num_buttons : ℕ := 10
def buttons_to_press : ℕ := 3
def time_per_attempt : ℕ := 2

def total_combinations : ℕ := (num_buttons.choose buttons_to_press)

theorem lock_problem :
  let total_time : ℕ := total_combinations * time_per_attempt
  let avg_attempts : ℚ := (1 + total_combinations : ℚ) / 2
  let avg_time : ℚ := avg_attempts * time_per_attempt
  let max_attempts_in_minute : ℕ := 60 / time_per_attempt
  (total_time = 240) ∧
  (avg_time = 121) ∧
  (max_attempts_in_minute : ℚ) / total_combinations = 29 / 120 := by
  sorry

end lock_problem_l1445_144561


namespace squirrel_walnuts_l1445_144572

/-- The number of walnuts the boy squirrel effectively adds to the burrow -/
def boy_walnuts : ℕ := 5

/-- The number of walnuts the girl squirrel effectively adds to the burrow -/
def girl_walnuts : ℕ := 3

/-- The final number of walnuts in the burrow -/
def final_walnuts : ℕ := 20

/-- The initial number of walnuts in the burrow -/
def initial_walnuts : ℕ := 12

theorem squirrel_walnuts :
  initial_walnuts + boy_walnuts + girl_walnuts = final_walnuts :=
by
  sorry

end squirrel_walnuts_l1445_144572


namespace geometric_arithmetic_sequence_problem_l1445_144507

theorem geometric_arithmetic_sequence_problem (a b c : ℝ) : 
  a + b + c = 114 →
  b / a = c / b →
  b / a ≠ 1 →
  b - a = c - b →
  c - a = 24 * (b - a) →
  a = 2 ∧ b = 14 ∧ c = 98 := by sorry

end geometric_arithmetic_sequence_problem_l1445_144507


namespace square_difference_given_system_l1445_144593

theorem square_difference_given_system (x y : ℝ) 
  (eq1 : 3 * x + 2 * y = 20) 
  (eq2 : 4 * x + 3 * y = 29) : 
  x^2 - y^2 = -45 := by
  sorry

end square_difference_given_system_l1445_144593


namespace parallel_planes_sufficient_not_necessary_l1445_144562

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (contains : Plane → Line → Prop)
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Plane → Plane → Prop)
variable (perpendicularLines : Line → Line → Prop)

-- State the theorem
theorem parallel_planes_sufficient_not_necessary
  (a b : Line) (α β : Plane)
  (h1 : contains α a)
  (h2 : perpendicular b β) :
  (∀ a b α β, parallel α β → perpendicularLines a b) ∧
  (∃ a b α β, perpendicularLines a b ∧ ¬ parallel α β) :=
sorry

end parallel_planes_sufficient_not_necessary_l1445_144562


namespace nonshaded_perimeter_is_64_l1445_144589

/-- A structure representing the geometric configuration described in the problem -/
structure GeometricConfig where
  outer_length : ℝ
  outer_width : ℝ
  inner_length : ℝ
  inner_width : ℝ
  extension : ℝ
  shaded_area : ℝ

/-- The perimeter of the non-shaded region given the geometric configuration -/
def nonshaded_perimeter (config : GeometricConfig) : ℝ :=
  2 * (config.outer_width + (config.outer_length + config.extension - config.inner_length))

/-- Theorem stating that given the specific geometric configuration, 
    the perimeter of the non-shaded region is 64 inches -/
theorem nonshaded_perimeter_is_64 (config : GeometricConfig) 
  (h1 : config.outer_length = 12)
  (h2 : config.outer_width = 10)
  (h3 : config.inner_length = 3)
  (h4 : config.inner_width = 4)
  (h5 : config.extension = 3)
  (h6 : config.shaded_area = 120) :
  nonshaded_perimeter config = 64 := by
  sorry

end nonshaded_perimeter_is_64_l1445_144589


namespace tangent_triangle_area_l1445_144514

/-- The area of the triangle formed by the tangent line at (1, e^(-1)) on y = e^(-x) and the axes is 2/e -/
theorem tangent_triangle_area :
  let f : ℝ → ℝ := fun x ↦ Real.exp (-x)
  let M : ℝ × ℝ := (1, Real.exp (-1))
  let tangent_line (x : ℝ) : ℝ := -Real.exp (-1) * (x - 1) + Real.exp (-1)
  let x_intercept : ℝ := 2
  let y_intercept : ℝ := 2 * Real.exp (-1)
  let triangle_area : ℝ := (1/2) * x_intercept * y_intercept
  triangle_area = 2 / Real.exp 1 :=
by sorry


end tangent_triangle_area_l1445_144514


namespace snow_probability_l1445_144500

theorem snow_probability (p : ℝ) (h : p = 3 / 4) :
  1 - (1 - p)^4 = 255 / 256 := by
  sorry

end snow_probability_l1445_144500


namespace least_number_divisible_up_to_28_l1445_144525

def is_divisible_up_to (n : ℕ) (m : ℕ) : Prop :=
  ∀ k : ℕ, 1 ≤ k ∧ k ≤ m → n % k = 0

theorem least_number_divisible_up_to_28 :
  ∃ n : ℕ, n > 0 ∧ is_divisible_up_to n 28 ∧
  (∀ m : ℕ, 0 < m ∧ m < n → ¬is_divisible_up_to m 28) ∧
  n = 5348882400 := by
  sorry

end least_number_divisible_up_to_28_l1445_144525


namespace square_area_ratio_l1445_144523

theorem square_area_ratio (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_perimeter : 4 * a = 4 * (4 * b)) : a^2 = 16 * b^2 := by
  sorry

end square_area_ratio_l1445_144523


namespace five_fourths_of_fifteen_fourths_l1445_144553

theorem five_fourths_of_fifteen_fourths (x : ℚ) : 
  x = 15 / 4 → (5 / 4 : ℚ) * x = 75 / 16 := by
  sorry

end five_fourths_of_fifteen_fourths_l1445_144553


namespace product_of_three_numbers_l1445_144569

theorem product_of_three_numbers (x y z : ℝ) : 
  x > 0 → y > 0 → z > 0 → 
  x * y * z = 1 → 
  x + 1 / z = 8 → 
  y + 1 / x = 20 → 
  z + 1 / y = 10 / 53 := by
sorry

end product_of_three_numbers_l1445_144569


namespace village_population_problem_l1445_144598

theorem village_population_problem (original : ℕ) : 
  (original : ℝ) * 0.9 * 0.75 = 5130 → original = 7600 := by
  sorry

end village_population_problem_l1445_144598


namespace temp_increase_pressure_decrease_sea_water_heat_engine_possible_l1445_144556

-- Define an ideal gas
structure IdealGas where
  temperature : ℝ
  pressure : ℝ
  volume : ℝ
  particle_count : ℕ

-- Define the ideal gas law
axiom ideal_gas_law (gas : IdealGas) : gas.pressure * gas.volume = gas.particle_count * gas.temperature

-- Define average kinetic energy of molecules
def avg_kinetic_energy (gas : IdealGas) : ℝ := gas.temperature

-- Define a heat engine
structure HeatEngine where
  hot_reservoir : ℝ
  cold_reservoir : ℝ

-- Theorem 1: Temperature increase can lead to increased kinetic energy but decreased pressure
theorem temp_increase_pressure_decrease (gas1 gas2 : IdealGas) 
  (h_temp : gas2.temperature > gas1.temperature)
  (h_volume : gas2.volume = gas1.volume)
  (h_particles : gas2.particle_count = gas1.particle_count) :
  avg_kinetic_energy gas2 > avg_kinetic_energy gas1 ∧ 
  ∃ (p : ℝ), gas2.pressure = p ∧ p < gas1.pressure :=
sorry

-- Theorem 2: Heat engine using sea water temperature difference is theoretically possible
theorem sea_water_heat_engine_possible (shallow_temp deep_temp : ℝ) 
  (h_temp_diff : shallow_temp > deep_temp) :
  ∃ (engine : HeatEngine), engine.hot_reservoir = shallow_temp ∧ 
    engine.cold_reservoir = deep_temp ∧
    (∃ (work : ℝ), work > 0) :=
sorry

end temp_increase_pressure_decrease_sea_water_heat_engine_possible_l1445_144556


namespace larger_number_proof_l1445_144531

theorem larger_number_proof (L S : ℕ) (h1 : L - S = 1395) (h2 : L = 6 * S + 15) : L = 1671 := by
  sorry

end larger_number_proof_l1445_144531


namespace prop_equivalence_l1445_144544

theorem prop_equivalence (p q : Prop) 
  (h1 : p ∨ q) 
  (h2 : ¬(p ∧ q)) : 
  p ↔ ¬q := by
  sorry

end prop_equivalence_l1445_144544


namespace triangle_solution_l1445_144530

noncomputable def triangle_problem (a b c A B C : ℝ) : Prop :=
  (2 * b - c) * Real.cos A = a * Real.cos C ∧
  a = Real.sqrt 13 ∧
  (1 / 2) * b * c * Real.sin A = 3 * Real.sqrt 3

theorem triangle_solution (a b c A B C : ℝ) 
  (h : triangle_problem a b c A B C) :
  A = π / 3 ∧ a + b + c = 7 + Real.sqrt 13 := by
  sorry

end triangle_solution_l1445_144530


namespace paint_remaining_after_three_days_paint_problem_solution_l1445_144579

/-- Represents the amount of paint remaining after a certain number of days -/
def paint_remaining (initial_amount : ℚ) (days : ℕ) : ℚ :=
  initial_amount * (1 / 2) ^ days

/-- Theorem stating that after 3 days of using half the remaining paint each day, 
    1/4 of the original amount remains -/
theorem paint_remaining_after_three_days (initial_amount : ℚ) :
  paint_remaining initial_amount 3 = initial_amount / 4 := by
  sorry

/-- Theorem proving that starting with 2 gallons and using half the remaining paint 
    for three consecutive days leaves 1/4 of the original amount -/
theorem paint_problem_solution :
  paint_remaining 2 3 = 1 / 2 := by
  sorry

end paint_remaining_after_three_days_paint_problem_solution_l1445_144579


namespace carly_to_lisa_jeans_ratio_l1445_144522

/-- Represents the spending of a person on different items --/
structure Spending :=
  (tshirts : ℚ)
  (jeans : ℚ)
  (coats : ℚ)

/-- Calculate the total spending of a person --/
def totalSpending (s : Spending) : ℚ :=
  s.tshirts + s.jeans + s.coats

/-- Lisa's spending based on the given conditions --/
def lisa : Spending :=
  { tshirts := 40
  , jeans := 40 / 2
  , coats := 40 * 2 }

/-- Carly's spending based on the given conditions --/
def carly : Spending :=
  { tshirts := lisa.tshirts / 4
  , jeans := lisa.jeans * (230 - totalSpending lisa - (lisa.tshirts / 4) - (lisa.coats / 4)) / lisa.jeans
  , coats := lisa.coats / 4 }

/-- The main theorem to prove --/
theorem carly_to_lisa_jeans_ratio :
  carly.jeans / lisa.jeans = 3 := by sorry

end carly_to_lisa_jeans_ratio_l1445_144522


namespace barbara_candies_l1445_144526

/-- The number of candies Barbara has in total is 27, given her initial candies and additional purchase. -/
theorem barbara_candies : 
  ∀ (initial_candies additional_candies : ℕ),
    initial_candies = 9 →
    additional_candies = 18 →
    initial_candies + additional_candies = 27 :=
by
  sorry

end barbara_candies_l1445_144526


namespace candy_box_price_increase_l1445_144577

theorem candy_box_price_increase (P : ℝ) : P + 0.25 * P = 10 → P = 8 := by
  sorry

end candy_box_price_increase_l1445_144577


namespace container_capacity_l1445_144517

theorem container_capacity (C : ℝ) (h : 0.35 * C + 48 = 0.75 * C) : C = 120 := by
  sorry

end container_capacity_l1445_144517


namespace triangle_with_unequal_angle_l1445_144594

theorem triangle_with_unequal_angle (a b c : ℝ) : 
  a + b + c = 180 →  -- Sum of angles in a triangle is 180°
  a = b →            -- Two angles are equal
  c = a - 10 →       -- Third angle is 10° less than the others
  c = 53.33 :=       -- Measure of the smallest angle
by sorry

end triangle_with_unequal_angle_l1445_144594


namespace front_view_length_l1445_144532

/-- Given a line segment with length 5√2, side view 5, and top view √34, 
    its front view has length √41. -/
theorem front_view_length 
  (segment_length : ℝ) 
  (side_view : ℝ) 
  (top_view : ℝ) 
  (h1 : segment_length = 5 * Real.sqrt 2)
  (h2 : side_view = 5)
  (h3 : top_view = Real.sqrt 34) : 
  Real.sqrt (side_view^2 + top_view^2 + (Real.sqrt 41)^2) = segment_length :=
by sorry

end front_view_length_l1445_144532


namespace expression_evaluation_l1445_144529

theorem expression_evaluation :
  let x : ℚ := 1/2
  let y : ℚ := -3
  (15 * x^3 * y - 10 * x^2 * y^2) / (5 * x * y) - (3*x + y) * (x - 3*y) = 18 :=
by sorry

end expression_evaluation_l1445_144529


namespace sin_2alpha_minus_pi_6_l1445_144513

theorem sin_2alpha_minus_pi_6 (α : Real) 
  (h : 2 * Real.sin α = 1 + 2 * Real.sqrt 3 * Real.cos α) : 
  Real.sin (2 * α - Real.pi / 6) = 7 / 8 := by
  sorry

end sin_2alpha_minus_pi_6_l1445_144513


namespace row_sum_is_odd_square_l1445_144509

/-- The sum of an arithmetic progression -/
def arithmetic_sum (a₁ : ℕ) (d : ℕ) (n : ℕ) : ℕ :=
  n * (2 * a₁ + (n - 1) * d) / 2

/-- The statement to be proved -/
theorem row_sum_is_odd_square (n : ℕ) (h : n > 0) :
  arithmetic_sum n 1 (2 * n - 1) = (2 * n - 1)^2 := by
  sorry

end row_sum_is_odd_square_l1445_144509


namespace polynomial_with_geometric_zeros_l1445_144548

/-- A polynomial of the form x^4 + jx^2 + kx + 256 with four distinct real zeros in geometric progression has j = -32 -/
theorem polynomial_with_geometric_zeros (j k : ℝ) : 
  (∃ (a r : ℝ) (hr : r ≠ 1) (ha : a ≠ 0), 
    (∀ x : ℝ, x^4 + j*x^2 + k*x + 256 = (x - a*r^3) * (x - a*r^2) * (x - a*r) * (x - a)) ∧ 
    (a*r^3 ≠ a*r^2) ∧ (a*r^2 ≠ a*r) ∧ (a*r ≠ a)) → 
  j = -32 := by
sorry

end polynomial_with_geometric_zeros_l1445_144548


namespace amount_increases_to_approx_87030_l1445_144567

/-- The amount after two years given an initial amount and yearly increase rate. -/
def amount_after_two_years (initial_amount : ℝ) (yearly_increase_rate : ℝ) : ℝ :=
  initial_amount * (1 + yearly_increase_rate)^2

/-- Theorem stating that given an initial amount of 64000 that increases by 1/6th each year,
    the amount after two years is approximately 87030.40. -/
theorem amount_increases_to_approx_87030 :
  let initial_amount := 64000
  let yearly_increase_rate := 1 / 6
  let final_amount := amount_after_two_years initial_amount yearly_increase_rate
  ∃ ε > 0, |final_amount - 87030.40| < ε :=
sorry

end amount_increases_to_approx_87030_l1445_144567


namespace fred_onions_l1445_144564

theorem fred_onions (sara_onions sally_onions total_onions : ℕ) 
  (h1 : sara_onions = 4)
  (h2 : sally_onions = 5)
  (h3 : total_onions = 18) :
  total_onions - (sara_onions + sally_onions) = 9 := by
sorry

end fred_onions_l1445_144564


namespace coprime_2013_in_32nd_group_l1445_144583

def is_coprime (a b : ℕ) : Prop := Nat.gcd a b = 1

def group_size (n : ℕ) : ℕ := 2 * n - 1

def cumulative_group_size (n : ℕ) : ℕ := n^2

def coprime_count (n : ℕ) : ℕ := n - (n.div 2 + n.div 503 - n.div 1006)

theorem coprime_2013_in_32nd_group :
  ∃ k : ℕ, k = 32 ∧
    coprime_count 2012 < cumulative_group_size (k - 1) ∧
    coprime_count 2012 + 1 ≤ cumulative_group_size k ∧
    is_coprime 2013 2012 := by
  sorry

end coprime_2013_in_32nd_group_l1445_144583


namespace triangle_xy_length_l1445_144555

-- Define the triangle
def Triangle (X Y Z : ℝ × ℝ) : Prop :=
  -- Right angle at X
  (Y.1 - X.1) * (Z.1 - X.1) + (Y.2 - X.2) * (Z.2 - X.2) = 0 ∧
  -- 45° angle at Y
  (Z.1 - Y.1) * (X.1 - Y.1) + (Z.2 - Y.2) * (X.2 - Y.2) = 
    Real.sqrt ((Z.1 - Y.1)^2 + (Z.2 - Y.2)^2) * Real.sqrt ((X.1 - Y.1)^2 + (X.2 - Y.2)^2) / 2 ∧
  -- XZ = 12√2
  (Z.1 - X.1)^2 + (Z.2 - X.2)^2 = 288

-- Theorem statement
theorem triangle_xy_length (X Y Z : ℝ × ℝ) (h : Triangle X Y Z) :
  (Y.1 - X.1)^2 + (Y.2 - X.2)^2 = 288 := by
  sorry

end triangle_xy_length_l1445_144555


namespace work_days_calculation_l1445_144568

theorem work_days_calculation (total_days : ℕ) (work_pay : ℕ) (no_work_deduction : ℕ) (total_earnings : ℤ) :
  total_days = 30 ∧ 
  work_pay = 80 ∧ 
  no_work_deduction = 40 ∧ 
  total_earnings = 1600 →
  ∃ (days_not_worked : ℕ),
    days_not_worked = 20 ∧
    total_earnings = work_pay * (total_days - days_not_worked) - no_work_deduction * days_not_worked :=
by sorry


end work_days_calculation_l1445_144568


namespace inequality_solution_condition_l1445_144512

theorem inequality_solution_condition (a : ℝ) : 
  (∀ x : ℝ, (a + 1) * x > a + 1 ↔ x < 1) → a < -1 := by
  sorry

end inequality_solution_condition_l1445_144512


namespace inscribed_polyhedron_radius_gt_three_l1445_144550

/-- A polyhedron inscribed in a sphere -/
structure InscribedPolyhedron where
  radius : ℝ
  volume : ℝ
  surface_area : ℝ
  volume_eq_surface_area : volume = surface_area

/-- Theorem: For any polyhedron inscribed in a sphere, if its volume equals its surface area, then the radius of the sphere is greater than 3 -/
theorem inscribed_polyhedron_radius_gt_three (p : InscribedPolyhedron) : p.radius > 3 := by
  sorry

end inscribed_polyhedron_radius_gt_three_l1445_144550


namespace perp_planes_parallel_perp_plane_line_perp_l1445_144573

-- Define the types for lines and planes
variable (L : Type) [LinearOrderedField L]
variable (P : Type)

-- Define the relations
variable (parallel : L → L → Prop)
variable (perp : L → L → Prop)
variable (perp_plane : L → P → Prop)
variable (parallel_plane : P → P → Prop)
variable (contained : L → P → Prop)

-- Theorem 1
theorem perp_planes_parallel
  (m : L) (α β : P)
  (h1 : perp_plane m α)
  (h2 : perp_plane m β)
  : parallel_plane α β :=
sorry

-- Theorem 2
theorem perp_plane_line_perp
  (m n : L) (α : P)
  (h1 : perp_plane m α)
  (h2 : contained n α)
  : perp m n :=
sorry

end perp_planes_parallel_perp_plane_line_perp_l1445_144573


namespace parallelepiped_volume_solution_l1445_144528

/-- The volume of a parallelepiped defined by vectors (3,4,5), (2,k,3), and (2,3,k) -/
def parallelepipedVolume (k : ℝ) : ℝ := |3 * k^2 - 15 * k + 27|

/-- Theorem stating that k = 5 is the positive solution for the parallelepiped volume equation -/
theorem parallelepiped_volume_solution :
  ∃! k : ℝ, k > 0 ∧ parallelepipedVolume k = 27 ∧ k = 5 := by sorry

end parallelepiped_volume_solution_l1445_144528


namespace parallelogram_side_sum_l1445_144581

/-- A parallelogram with given side lengths -/
structure Parallelogram where
  side1 : ℝ
  side2 : ℝ
  side3 : ℝ → ℝ
  side4 : ℝ → ℝ

/-- The specific parallelogram from the problem -/
def problem_parallelogram : Parallelogram where
  side1 := 12
  side2 := 15
  side3 := fun y => 10 * y - 3
  side4 := fun x => 3 * x + 6

/-- The theorem stating the solution to the problem -/
theorem parallelogram_side_sum (p : Parallelogram) 
  (h1 : p.side1 = p.side4 1)
  (h2 : p.side2 = p.side3 2)
  (h3 : p = problem_parallelogram) :
  ∃ (x y : ℝ), x + y = 3.8 ∧ p.side3 y = 15 ∧ p.side4 x = 12 := by
  sorry

end parallelogram_side_sum_l1445_144581
