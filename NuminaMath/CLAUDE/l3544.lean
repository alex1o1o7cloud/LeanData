import Mathlib

namespace problem_1_problem_2_l3544_354483

-- Problem 1
theorem problem_1 : (π - 2023) ^ 0 - 3 * Real.tan (π / 6) + |1 - Real.sqrt 3| = 0 := by
  sorry

-- Problem 2
theorem problem_2 (x : ℝ) (hx : x ≠ 0 ∧ x ≠ 1) :
  ((2 * x + 1) / (x - 1) - 1) / ((2 * x + x^2) / (x^2 - 2 * x + 1)) = (x - 1) / x := by
  sorry

end problem_1_problem_2_l3544_354483


namespace work_completion_theorem_l3544_354451

theorem work_completion_theorem (initial_men : ℕ) (initial_days : ℕ) (new_days : ℕ) (new_men : ℕ) :
  initial_men = 36 →
  initial_days = 18 →
  new_days = 8 →
  initial_men * initial_days = new_men * new_days →
  new_men = 81 := by
sorry

end work_completion_theorem_l3544_354451


namespace geometric_sequence_ratio_l3544_354429

theorem geometric_sequence_ratio (a : ℕ → ℝ) (q : ℝ) :
  (∀ n : ℕ, a (n + 1) = q * a n) →  -- geometric sequence condition
  q = -3 →                         -- given common ratio
  (a 1 + a 3 + a 5 + a 7) / (a 2 + a 4 + a 6 + a 8) = -1/3 := by
  sorry

end geometric_sequence_ratio_l3544_354429


namespace cooking_and_weaving_count_l3544_354469

/-- Represents the number of people in various curriculum combinations -/
structure CurriculumParticipation where
  yoga : ℕ
  cooking : ℕ
  weaving : ℕ
  cookingOnly : ℕ
  cookingAndYoga : ℕ
  allCurriculums : ℕ

/-- Theorem stating the number of people studying both cooking and weaving -/
theorem cooking_and_weaving_count (cp : CurriculumParticipation)
  (h1 : cp.yoga = 35)
  (h2 : cp.cooking = 20)
  (h3 : cp.weaving = 15)
  (h4 : cp.cookingOnly = 7)
  (h5 : cp.cookingAndYoga = 5)
  (h6 : cp.allCurriculums = 3) :
  cp.cooking - cp.cookingOnly - cp.cookingAndYoga + cp.allCurriculums = 5 := by
  sorry


end cooking_and_weaving_count_l3544_354469


namespace school_pupils_l3544_354463

theorem school_pupils (girls : ℕ) (boys : ℕ) (h1 : girls = 692) (h2 : girls = boys + 458) :
  girls + boys = 926 := by
  sorry

end school_pupils_l3544_354463


namespace smallest_k_divisible_by_power_of_three_l3544_354485

theorem smallest_k_divisible_by_power_of_three : ∃ k : ℕ, 
  (∀ m : ℕ, m < k → ¬(3^67 ∣ 2016^m)) ∧ (3^67 ∣ 2016^k) ∧ k = 34 := by
  sorry

end smallest_k_divisible_by_power_of_three_l3544_354485


namespace igor_lied_l3544_354433

-- Define the set of boys
inductive Boy : Type
| andrey : Boy
| maxim : Boy
| igor : Boy
| kolya : Boy

-- Define the possible positions in the race
inductive Position : Type
| first : Position
| second : Position
| third : Position
| fourth : Position

-- Define a function to represent the actual position of each boy
def actual_position : Boy → Position := sorry

-- Define a function to represent whether a boy is telling the truth
def is_truthful : Boy → Prop := sorry

-- State the conditions of the problem
axiom three_truthful : ∃ (a b c : Boy), a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ 
  is_truthful a ∧ is_truthful b ∧ is_truthful c ∧ 
  ∀ (d : Boy), d ≠ a ∧ d ≠ b ∧ d ≠ c → ¬is_truthful d

axiom andrey_claim : is_truthful Boy.andrey ↔ 
  actual_position Boy.andrey ≠ Position.first ∧ 
  actual_position Boy.andrey ≠ Position.fourth

axiom maxim_claim : is_truthful Boy.maxim ↔ 
  actual_position Boy.maxim ≠ Position.fourth

axiom igor_claim : is_truthful Boy.igor ↔ 
  actual_position Boy.igor = Position.first

axiom kolya_claim : is_truthful Boy.kolya ↔ 
  actual_position Boy.kolya = Position.fourth

-- Theorem to prove
theorem igor_lied : ¬is_truthful Boy.igor := by sorry

end igor_lied_l3544_354433


namespace boat_trip_time_l3544_354482

theorem boat_trip_time (v : ℝ) :
  (90 = (v - 3) * (T + 0.5)) →
  (90 = (v + 3) * T) →
  (T > 0) →
  T = 2.5 := by
  sorry

end boat_trip_time_l3544_354482


namespace rectangular_cards_are_squares_l3544_354448

/-- Represents a rectangular card with dimensions width and height -/
structure Card where
  width : ℕ+
  height : ℕ+

/-- Represents the result of a child cutting their card into squares -/
structure CutResult where
  squareCount : ℕ+

theorem rectangular_cards_are_squares
  (n : ℕ+)
  (h_n : n > 1)
  (cards : Fin n → Card)
  (h_identical : ∀ i j : Fin n, cards i = cards j)
  (cuts : Fin n → CutResult)
  (h_prime_total : Nat.Prime (Finset.sum (Finset.range n) (λ i => (cuts i).squareCount))) :
  ∀ i : Fin n, (cards i).width = (cards i).height :=
sorry

end rectangular_cards_are_squares_l3544_354448


namespace problem_solution_l3544_354444

open Real

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := log (exp x + a)

def is_odd (g : ℝ → ℝ) : Prop := ∀ x, g (-x) = -g x

theorem problem_solution :
  (∃ a, is_odd (f a)) →
  (∃ a, a = 0 ∧ is_odd (f a)) ∧
  (∀ m : ℝ,
    (m > 1/exp 1 + exp 2 → ¬∃ x, (log x) / x = x^2 - 2 * (exp 1) * x + m) ∧
    (m = 1/exp 1 + exp 2 → ∃! x, x = exp 1 ∧ (log x) / x = x^2 - 2 * (exp 1) * x + m) ∧
    (m < 1/exp 1 + exp 2 → ∃ x y, x ≠ y ∧ (log x) / x = x^2 - 2 * (exp 1) * x + m ∧ (log y) / y = y^2 - 2 * (exp 1) * y + m))
    := by sorry

end problem_solution_l3544_354444


namespace radical_simplification_l3544_354409

theorem radical_simplification (p : ℝ) (hp : p > 0) :
  Real.sqrt (15 * p^2) * Real.sqrt (8 * p) * Real.sqrt (27 * p^5) = 18 * p^4 * Real.sqrt 10 := by
  sorry

end radical_simplification_l3544_354409


namespace girls_from_maple_grove_l3544_354438

-- Define the total number of students
def total_students : ℕ := 150

-- Define the number of boys
def num_boys : ℕ := 82

-- Define the number of girls
def num_girls : ℕ := 68

-- Define the number of students from Pine Ridge School
def pine_ridge_students : ℕ := 70

-- Define the number of students from Maple Grove School
def maple_grove_students : ℕ := 80

-- Define the number of boys from Pine Ridge School
def pine_ridge_boys : ℕ := 36

-- Theorem to prove
theorem girls_from_maple_grove :
  total_students = num_boys + num_girls ∧
  total_students = pine_ridge_students + maple_grove_students ∧
  num_boys = pine_ridge_boys + (num_boys - pine_ridge_boys) →
  maple_grove_students - (num_boys - pine_ridge_boys) = 34 :=
by sorry

end girls_from_maple_grove_l3544_354438


namespace picnic_attendance_theorem_l3544_354426

/-- The percentage of men who attended the picnic -/
def men_attendance_rate : ℝ := 0.20

/-- The percentage of women who attended the picnic -/
def women_attendance_rate : ℝ := 0.40

/-- The percentage of employees who are men -/
def men_employee_rate : ℝ := 0.55

/-- The percentage of all employees who attended the picnic -/
def total_attendance_rate : ℝ := men_employee_rate * men_attendance_rate + (1 - men_employee_rate) * women_attendance_rate

theorem picnic_attendance_theorem :
  total_attendance_rate = 0.29 := by sorry

end picnic_attendance_theorem_l3544_354426


namespace cool_double_l3544_354443

def is_cool (n : ℕ) : Prop := ∃ a b : ℕ, n = a^2 + b^2

theorem cool_double {k : ℕ} (h : is_cool k) : is_cool (2 * k) := by
  sorry

end cool_double_l3544_354443


namespace shared_rest_days_count_l3544_354464

/-- Chris's work cycle in days -/
def chris_cycle : ℕ := 7

/-- Dana's work cycle in days -/
def dana_cycle : ℕ := 7

/-- Total number of days -/
def total_days : ℕ := 1200

/-- Number of rest days Chris has in a cycle -/
def chris_rest_days : ℕ := 2

/-- Number of rest days Dana has in a cycle -/
def dana_rest_days : ℕ := 1

/-- The day in the cycle when both Chris and Dana rest -/
def common_rest_day : ℕ := 7

/-- The number of times Chris and Dana share a rest day in the given period -/
def shared_rest_days : ℕ := total_days / chris_cycle

theorem shared_rest_days_count :
  shared_rest_days = 171 :=
sorry

end shared_rest_days_count_l3544_354464


namespace largest_number_l3544_354462

def a : ℚ := 8.23455
def b : ℚ := 8 + 234 / 1000 + 5 / 9000
def c : ℚ := 8 + 23 / 100 + 45 / 9900
def d : ℚ := 8 + 2 / 10 + 345 / 999
def e : ℚ := 8 + 2345 / 9999

theorem largest_number : b > a ∧ b > c ∧ b > d ∧ b > e := by
  sorry

end largest_number_l3544_354462


namespace triangle_ambiguous_case_l3544_354402

theorem triangle_ambiguous_case (a b : ℝ) (A : ℝ) : 
  a = 12 → A = π / 3 → (b * Real.sin A < a ∧ a < b) ↔ (12 < b ∧ b < 8 * Real.sqrt 3) := by
  sorry

end triangle_ambiguous_case_l3544_354402


namespace wall_length_height_ratio_l3544_354400

/-- Represents the dimensions and volume of a rectangular wall. -/
structure Wall where
  breadth : ℝ
  height : ℝ
  length : ℝ
  volume : ℝ

/-- Theorem stating the ratio of length to height for a specific wall. -/
theorem wall_length_height_ratio (w : Wall) 
  (h_volume : w.volume = 12.8)
  (h_breadth : w.breadth = 0.4)
  (h_height : w.height = 5 * w.breadth)
  (h_volume_calc : w.volume = w.breadth * w.height * w.length) :
  w.length / w.height = 4 := by
  sorry

#check wall_length_height_ratio

end wall_length_height_ratio_l3544_354400


namespace traditionalist_progressive_ratio_l3544_354423

/-- Represents a country with provinces, progressives, and traditionalists -/
structure Country where
  num_provinces : ℕ
  total_population : ℝ
  fraction_traditionalist : ℝ
  progressives : ℝ
  traditionalists_per_province : ℝ

/-- The theorem stating the ratio of traditionalists in one province to total progressives -/
theorem traditionalist_progressive_ratio (c : Country) 
  (h1 : c.num_provinces = 4)
  (h2 : c.fraction_traditionalist = 0.75)
  (h3 : c.total_population = c.progressives + c.num_provinces * c.traditionalists_per_province)
  (h4 : c.fraction_traditionalist * c.total_population = c.num_provinces * c.traditionalists_per_province) :
  c.traditionalists_per_province / c.progressives = 3 / 4 := by
  sorry


end traditionalist_progressive_ratio_l3544_354423


namespace triangle_cosine_identity_l3544_354460

-- Define a triangle ABC
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  a : ℝ
  b : ℝ
  c : ℝ
  -- Ensure A + B + C = π (180 degrees)
  angle_sum : A + B + C = Real.pi
  -- Ensure all angles are positive
  A_pos : A > 0
  B_pos : B > 0
  C_pos : C > 0
  -- Ensure the given condition 2b = a + c
  side_condition : 2 * b = a + c

-- Theorem statement
theorem triangle_cosine_identity (t : Triangle) :
  5 * Real.cos t.A - 4 * Real.cos t.A * Real.cos t.C + 5 * Real.cos t.C = 4 := by
  sorry

end triangle_cosine_identity_l3544_354460


namespace computers_probability_l3544_354484

def CAMPUS : Finset Char := {'C', 'A', 'M', 'P', 'U', 'S'}
def THREADS : Finset Char := {'T', 'H', 'R', 'E', 'A', 'D', 'S'}
def GLOW : Finset Char := {'G', 'L', 'O', 'W'}
def COMPUTERS : Finset Char := {'C', 'O', 'M', 'P', 'U', 'T', 'E', 'R', 'S'}

def probability_CAMPUS : ℚ := 1 / (CAMPUS.card.choose 3)
def probability_THREADS : ℚ := 1 / (THREADS.card.choose 5)
def probability_GLOW : ℚ := (GLOW.card - 1).choose 1 / (GLOW.card.choose 2)

theorem computers_probability :
  probability_CAMPUS * probability_THREADS * probability_GLOW = 1 / 840 := by
  sorry

end computers_probability_l3544_354484


namespace cos_squared_pi_eighth_minus_one_l3544_354447

theorem cos_squared_pi_eighth_minus_one (π : Real) : 2 * Real.cos (π / 8) ^ 2 - 1 = Real.sqrt 2 / 2 := by
  sorry

end cos_squared_pi_eighth_minus_one_l3544_354447


namespace area_two_quarter_circles_l3544_354413

/-- The area of a figure formed by two 90° sectors of a circle with radius 10 -/
theorem area_two_quarter_circles (r : ℝ) (h : r = 10) : 
  2 * (π * r^2 / 4) = 50 * π := by
  sorry

end area_two_quarter_circles_l3544_354413


namespace probability_grape_star_l3544_354435

/-- A tablet shape -/
inductive Shape
| Square
| Triangle
| Star

/-- A tablet flavor -/
inductive Flavor
| Strawberry
| Grape
| Orange

/-- The number of tablets of each shape -/
def tablets_per_shape : ℕ := 60

/-- The number of flavors -/
def num_flavors : ℕ := 3

/-- The total number of tablets -/
def total_tablets : ℕ := tablets_per_shape * 3

/-- The number of grape star tablets -/
def grape_star_tablets : ℕ := tablets_per_shape / num_flavors

theorem probability_grape_star :
  (grape_star_tablets : ℚ) / total_tablets = 1 / 9 := by
  sorry

end probability_grape_star_l3544_354435


namespace role_assignment_combinations_l3544_354477

def number_of_friends : ℕ := 6

theorem role_assignment_combinations (maria_is_cook : Bool) 
  (h1 : maria_is_cook = true) 
  (h2 : number_of_friends = 6) : 
  (Nat.choose (number_of_friends - 1) 1) * (Nat.choose (number_of_friends - 2) 2) = 30 := by
  sorry

end role_assignment_combinations_l3544_354477


namespace photo_shoot_count_l3544_354478

/-- The number of photos taken during a photo shoot, given initial conditions and final count --/
theorem photo_shoot_count (initial : ℕ) (deleted_first : ℕ) (added_first : ℕ)
  (deleted_friend1 : ℕ) (added_friend1 : ℕ)
  (deleted_friend2 : ℕ) (added_friend2 : ℕ)
  (added_friend3 : ℕ)
  (deleted_last : ℕ) (final : ℕ) :
  initial = 63 →
  deleted_first = 7 →
  added_first = 15 →
  deleted_friend1 = 3 →
  added_friend1 = 5 →
  deleted_friend2 = 1 →
  added_friend2 = 4 →
  added_friend3 = 6 →
  deleted_last = 2 →
  final = 112 →
  ∃ x : ℕ, x = 32 ∧
    final = initial - deleted_first + added_first + x - deleted_friend1 + added_friend1 - deleted_friend2 + added_friend2 + added_friend3 - deleted_last :=
by sorry

end photo_shoot_count_l3544_354478


namespace caterer_order_l3544_354407

/-- The number of ice-cream bars ordered by a caterer -/
def num_ice_cream_bars : ℕ := 225

/-- The total price of the order in cents -/
def total_price : ℕ := 20000

/-- The price of each ice-cream bar in cents -/
def price_ice_cream_bar : ℕ := 60

/-- The price of each sundae in cents -/
def price_sundae : ℕ := 52

/-- The number of sundaes ordered -/
def num_sundaes : ℕ := 125

theorem caterer_order :
  num_ice_cream_bars * price_ice_cream_bar + num_sundaes * price_sundae = total_price :=
by sorry

end caterer_order_l3544_354407


namespace rectangular_to_polar_conversion_l3544_354420

theorem rectangular_to_polar_conversion :
  let x : ℝ := 3
  let y : ℝ := -3
  let r : ℝ := Real.sqrt (x^2 + y^2)
  let θ : ℝ := if x > 0 ∧ y < 0 then 2 * π + Real.arctan (y / x) else Real.arctan (y / x)
  r > 0 ∧ 0 ≤ θ ∧ θ < 2 * π →
  r = 3 * Real.sqrt 2 ∧ θ = 7 * π / 4 := by sorry

end rectangular_to_polar_conversion_l3544_354420


namespace intersection_M_N_l3544_354442

def M : Set ℝ := {x | 2 * x - x^2 ≥ 0}

def N : Set ℝ := {x | ∃ y, y = 1 / Real.sqrt (1 - x^2)}

theorem intersection_M_N : M ∩ N = Set.Icc 0 1 := by sorry

end intersection_M_N_l3544_354442


namespace condition_equivalent_to_a_range_l3544_354424

/-- The function f(x) = ax - 1 -/
def f (a : ℝ) (x : ℝ) : ℝ := a * x - 1

/-- The function g(x) = -x^2 + 2x + 1 -/
def g (x : ℝ) : ℝ := -x^2 + 2*x + 1

/-- The theorem stating the equivalence between the condition and the range of a -/
theorem condition_equivalent_to_a_range :
  ∀ a : ℝ, (∀ x₁ ∈ Set.Icc (-1) 1, ∃ x₂ ∈ Set.Icc 0 2, f a x₁ < g x₂) ↔ a ∈ Set.Ioo (-3) 3 :=
by sorry

end condition_equivalent_to_a_range_l3544_354424


namespace hamburger_problem_l3544_354408

theorem hamburger_problem (total_spent : ℚ) (total_burgers : ℕ) 
  (single_cost : ℚ) (double_cost : ℚ) (h1 : total_spent = 68.5) 
  (h2 : total_burgers = 50) (h3 : single_cost = 1) (h4 : double_cost = 1.5) :
  ∃ (single_count double_count : ℕ),
    single_count + double_count = total_burgers ∧
    single_count * single_cost + double_count * double_cost = total_spent ∧
    double_count = 37 := by
  sorry

end hamburger_problem_l3544_354408


namespace sequence_periodicity_l3544_354450

/-- A cubic polynomial with rational coefficients -/
def CubicPolynomial (α : ℚ → ℚ) : Prop :=
  ∃ a b c d : ℚ, ∀ x, α x = a * x^3 + b * x^2 + c * x + d

/-- A sequence of rational numbers satisfying the given condition -/
def SequenceSatisfyingCondition (p : ℚ → ℚ) (q : ℕ → ℚ) : Prop :=
  ∀ n : ℕ, q n = p (q (n + 1))

theorem sequence_periodicity
  (p : ℚ → ℚ) (q : ℕ → ℚ)
  (h_cubic : CubicPolynomial p)
  (h_seq : SequenceSatisfyingCondition p q) :
  ∃ k : ℕ, k ≥ 1 ∧ ∀ n : ℕ, n ≥ 1 → q (n + k) = q n :=
sorry

end sequence_periodicity_l3544_354450


namespace largest_alpha_l3544_354440

theorem largest_alpha : ∃ (α : ℝ), (α = 3) ∧ 
  (∀ (m n : ℕ+), (m : ℝ) / n < Real.sqrt 7 → α / n^2 ≤ 7 - (m : ℝ)^2 / n^2) ∧
  (∀ (β : ℝ), β > α → 
    ∃ (m n : ℕ+), (m : ℝ) / n < Real.sqrt 7 ∧ β / n^2 > 7 - (m : ℝ)^2 / n^2) :=
sorry

end largest_alpha_l3544_354440


namespace stone_123_is_3_l3544_354415

/-- The number of stones in the sequence -/
def num_stones : ℕ := 12

/-- The length of the counting pattern before it repeats -/
def pattern_length : ℕ := 22

/-- The target count we're interested in -/
def target_count : ℕ := 123

/-- The original stone number we claim is counted as the target_count -/
def claimed_stone : ℕ := 3

/-- Function to determine which stone is counted as a given number -/
def stone_at_count (count : ℕ) : ℕ :=
  (count - 1) % pattern_length + 1

theorem stone_123_is_3 : 
  stone_at_count target_count = claimed_stone := by
  sorry

end stone_123_is_3_l3544_354415


namespace jefferson_high_club_overlap_l3544_354491

/-- Represents the number of students in both robotics and science clubs -/
def students_in_both_clubs (total : ℕ) (robotics : ℕ) (science : ℕ) (either : ℕ) : ℕ :=
  robotics + science - either

/-- Theorem: Given the conditions from Jefferson High School, 
    prove that there are 20 students in both robotics and science clubs -/
theorem jefferson_high_club_overlap :
  students_in_both_clubs 300 80 130 190 = 20 := by
  sorry

end jefferson_high_club_overlap_l3544_354491


namespace anthony_initial_pencils_l3544_354452

-- Define the variables
def pencils_given : ℝ := 9.0
def pencils_left : ℕ := 47

-- State the theorem
theorem anthony_initial_pencils : 
  pencils_given + pencils_left = 56 := by sorry

end anthony_initial_pencils_l3544_354452


namespace sugar_substitute_box_cost_l3544_354439

theorem sugar_substitute_box_cost 
  (packets_per_coffee : ℕ)
  (coffees_per_day : ℕ)
  (packets_per_box : ℕ)
  (days_supply : ℕ)
  (total_cost : ℝ)
  (h1 : packets_per_coffee = 1)
  (h2 : coffees_per_day = 2)
  (h3 : packets_per_box = 30)
  (h4 : days_supply = 90)
  (h5 : total_cost = 24) :
  total_cost / (days_supply * coffees_per_day * packets_per_coffee / packets_per_box) = 4 := by
  sorry

end sugar_substitute_box_cost_l3544_354439


namespace semicircles_to_circle_area_ratio_l3544_354495

/-- The ratio of the combined areas of two semicircles with radius r/2 inscribed in a circle with radius r to the area of the circle is 1/4. -/
theorem semicircles_to_circle_area_ratio (r : ℝ) (h : r > 0) : 
  (2 * (π * (r/2)^2 / 2)) / (π * r^2) = 1/4 := by
  sorry

end semicircles_to_circle_area_ratio_l3544_354495


namespace science_club_membership_l3544_354459

theorem science_club_membership (total : ℕ) (math : ℕ) (physics : ℕ) (both : ℕ)
  (h1 : total = 120)
  (h2 : math = 75)
  (h3 : physics = 50)
  (h4 : both = 15) :
  total - (math + physics - both) = 10 := by
sorry

end science_club_membership_l3544_354459


namespace log_sum_equality_l3544_354480

theorem log_sum_equality : Real.log 50 + Real.log 30 = 3 + Real.log 1.5 := by sorry

end log_sum_equality_l3544_354480


namespace tennis_balls_cost_l3544_354410

/-- The number of packs of tennis balls Melissa bought -/
def num_packs : ℕ := 4

/-- The number of balls in each pack -/
def balls_per_pack : ℕ := 3

/-- The cost of each tennis ball in dollars -/
def cost_per_ball : ℕ := 2

/-- The total cost of the tennis balls -/
def total_cost : ℕ := num_packs * balls_per_pack * cost_per_ball

theorem tennis_balls_cost : total_cost = 24 := by
  sorry

end tennis_balls_cost_l3544_354410


namespace window_length_l3544_354427

/-- Given a rectangular window with width 10 feet and area 60 square feet, its length is 6 feet -/
theorem window_length (width : ℝ) (area : ℝ) (length : ℝ) : 
  width = 10 → area = 60 → area = length * width → length = 6 := by
  sorry

end window_length_l3544_354427


namespace negation_of_universal_proposition_l3544_354419

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℕ, x^3 > x^2) ↔ (∃ x : ℕ, x^3 ≤ x^2) :=
by sorry

end negation_of_universal_proposition_l3544_354419


namespace meeting_unexpectedly_is_random_l3544_354488

/-- Represents an event --/
inductive Event
| WinterToSpring
| FishingMoonInWater
| SeekingFishOnTree
| MeetingUnexpectedly

/-- Defines whether an event is certain --/
def isCertain : Event → Prop
| Event.WinterToSpring => True
| _ => False

/-- Defines whether an event is impossible --/
def isImpossible : Event → Prop
| Event.FishingMoonInWater => True
| Event.SeekingFishOnTree => True
| _ => False

/-- Defines a random event --/
def isRandom (e : Event) : Prop :=
  ¬(isCertain e) ∧ ¬(isImpossible e)

/-- Theorem: Meeting unexpectedly is a random event --/
theorem meeting_unexpectedly_is_random :
  isRandom Event.MeetingUnexpectedly :=
by sorry

end meeting_unexpectedly_is_random_l3544_354488


namespace product_of_good_is_good_l3544_354467

/-- A positive integer is good if it can be represented as ax^2 + bxy + cy^2 
    with b^2 - 4ac = -20 for some integers a, b, c, x, y -/
def is_good (n : ℕ+) : Prop :=
  ∃ (a b c x y : ℤ), (n : ℤ) = a * x^2 + b * x * y + c * y^2 ∧ b^2 - 4 * a * c = -20

/-- The product of two good numbers is also a good number -/
theorem product_of_good_is_good (n1 n2 : ℕ+) (h1 : is_good n1) (h2 : is_good n2) :
  is_good (n1 * n2) :=
sorry

end product_of_good_is_good_l3544_354467


namespace max_triangles_theorem_l3544_354475

/-- Represents a convex n-gon with diagonals drawn such that no three or more intersect at a single point. -/
structure ConvexPolygonWithDiagonals where
  n : ℕ
  is_convex : Bool
  no_triple_intersection : Bool

/-- Calculates the maximum number of triangles formed by diagonals in a convex n-gon. -/
def max_triangles (polygon : ConvexPolygonWithDiagonals) : ℕ :=
  if polygon.n % 2 = 0 then
    2 * polygon.n - 4
  else
    2 * polygon.n - 5

/-- Theorem stating the maximum number of triangles formed by diagonals in a convex n-gon. -/
theorem max_triangles_theorem (polygon : ConvexPolygonWithDiagonals) :
  polygon.is_convex ∧ polygon.no_triple_intersection →
  max_triangles polygon = if polygon.n % 2 = 0 then 2 * polygon.n - 4 else 2 * polygon.n - 5 :=
by
  sorry

end max_triangles_theorem_l3544_354475


namespace parallel_line_equation_l3544_354499

/-- Given a line y = (5/3)x + 10, prove that a parallel line L
    that is 5 units away from it has the equation
    y = (5/3)x + (10 ± (5√34)/3) -/
theorem parallel_line_equation (x y : ℝ) :
  let original_line := fun x => (5/3) * x + 10
  let distance := 5
  let slope := 5/3
  let perpendicular_slope := -3/5
  let c := 10
  ∃ L : ℝ → ℝ,
    (∀ x, L x = slope * x + (c + distance * Real.sqrt (slope^2 + 1))) ∨
    (∀ x, L x = slope * x + (c - distance * Real.sqrt (slope^2 + 1))) ∧
    (∀ x, |L x - original_line x| / Real.sqrt (1 + perpendicular_slope^2) = distance) :=
by sorry

end parallel_line_equation_l3544_354499


namespace sphere_surface_area_l3544_354428

/-- Given a sphere whose surface area increases by 4π cm² when cut in half,
    prove that its original surface area was 8π cm². -/
theorem sphere_surface_area (R : ℝ) (h : 2 * Real.pi * R^2 = 4 * Real.pi) :
  4 * Real.pi * R^2 = 8 * Real.pi :=
by sorry

end sphere_surface_area_l3544_354428


namespace range_of_m_for_inequality_l3544_354437

theorem range_of_m_for_inequality (m : ℝ) : 
  (∀ x : ℕ+, (x = 1 ∨ x = 2 ∨ x = 3 ∨ x = 4) ↔ (3 * (x : ℝ) - 3 * m ≤ -2 * m)) → 
  (12 ≤ m ∧ m < 15) := by
sorry

end range_of_m_for_inequality_l3544_354437


namespace third_day_temperature_l3544_354472

/-- Given three temperatures in Fahrenheit, calculates their average -/
def average (t1 t2 t3 : ℚ) : ℚ := (t1 + t2 + t3) / 3

/-- Proves that given an average temperature of -7°F for three days, 
    with temperatures of -8°F and +1°F on two of the days, 
    the temperature on the third day must be -14°F -/
theorem third_day_temperature 
  (t1 t2 t3 : ℚ) 
  (h1 : t1 = -8)
  (h2 : t2 = 1)
  (h_avg : average t1 t2 t3 = -7) :
  t3 = -14 := by
  sorry

#eval average (-8) 1 (-14) -- Should output -7

end third_day_temperature_l3544_354472


namespace notebook_cost_l3544_354425

theorem notebook_cost (total_cost cover_cost notebook_cost : ℝ) : 
  total_cost = 3.60 →
  notebook_cost = 1.5 * cover_cost →
  total_cost = notebook_cost + cover_cost →
  notebook_cost = 2.16 := by
sorry

end notebook_cost_l3544_354425


namespace solve_pq_system_l3544_354458

theorem solve_pq_system (p q : ℝ) (hp : p > 1) (hq : q > 1) 
  (h1 : 1/p + 1/q = 1) (h2 : p * q = 9) : 
  q = (9 + 3 * Real.sqrt 5) / 2 ∨ q = (9 - 3 * Real.sqrt 5) / 2 := by
  sorry

end solve_pq_system_l3544_354458


namespace power_difference_square_sum_l3544_354493

theorem power_difference_square_sum (m n : ℕ+) : 
  2^(m : ℕ) - 2^(n : ℕ) = 1792 → m^2 + n^2 = 185 := by
  sorry

end power_difference_square_sum_l3544_354493


namespace rectangular_box_surface_area_l3544_354412

theorem rectangular_box_surface_area
  (x y z : ℝ)
  (h1 : 4 * x + 4 * y + 4 * z = 240)
  (h2 : Real.sqrt (x^2 + y^2 + z^2) = 31) :
  2 * (x * y + y * z + z * x) = 2639 := by
  sorry

end rectangular_box_surface_area_l3544_354412


namespace janes_leave_days_l3544_354479

theorem janes_leave_days (jane_rate ashley_rate total_days extra_days : ℝ) 
  (h1 : jane_rate = 1 / 10)
  (h2 : ashley_rate = 1 / 40)
  (h3 : total_days = 15.2)
  (h4 : extra_days = 4) : 
  ∃ leave_days : ℝ, 
    (jane_rate + ashley_rate) * (total_days - leave_days) + 
    ashley_rate * leave_days + 
    jane_rate * extra_days = 1 ∧ 
    leave_days = 13 := by
sorry

end janes_leave_days_l3544_354479


namespace subset_implies_m_leq_5_l3544_354404

/-- Given sets A and B, prove that if B is a subset of A, then m ≤ 5 -/
theorem subset_implies_m_leq_5 (m : ℝ) : 
  let A : Set ℝ := {x | 4 ≤ x ∧ x ≤ 8}
  let B : Set ℝ := {x | m + 1 < x ∧ x < 2*m - 2}
  B ⊆ A → m ≤ 5 := by
  sorry

end subset_implies_m_leq_5_l3544_354404


namespace fraction_zero_implies_x_equals_three_l3544_354476

theorem fraction_zero_implies_x_equals_three (x : ℝ) :
  (x^2 - 9) / (x + 3) = 0 ∧ x + 3 ≠ 0 → x = 3 := by
  sorry

end fraction_zero_implies_x_equals_three_l3544_354476


namespace train_crossing_time_l3544_354457

/-- Proves that a train with given length and speed takes the calculated time to cross an electric pole -/
theorem train_crossing_time (train_length : ℝ) (train_speed_km_hr : ℝ) (crossing_time : ℝ) : 
  train_length = 400 →
  train_speed_km_hr = 144 →
  crossing_time = train_length / (train_speed_km_hr * 1000 / 3600) →
  crossing_time = 10 := by
  sorry

#check train_crossing_time

end train_crossing_time_l3544_354457


namespace inequality_proof_l3544_354473

theorem inequality_proof (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  x^2 + 8/(x*y) + y^2 ≥ 8 ∧
  (x^2 + 8/(x*y) + y^2 = 8 ↔ x = Real.sqrt 2 ∧ y = Real.sqrt 2) :=
by sorry

end inequality_proof_l3544_354473


namespace first_interest_rate_is_five_percent_l3544_354422

/-- Proves that the first interest rate is 5% given the problem conditions --/
theorem first_interest_rate_is_five_percent
  (total_amount : ℝ)
  (first_part : ℝ)
  (second_part : ℝ)
  (second_interest_rate : ℝ)
  (total_income : ℝ)
  (h1 : total_amount = 2500)
  (h2 : first_part = 1000)
  (h3 : second_part = total_amount - first_part)
  (h4 : second_interest_rate = 6)
  (h5 : total_income = 140)
  (h6 : total_income = (first_part * first_interest_rate / 100) + (second_part * second_interest_rate / 100)) :
  first_interest_rate = 5 := by
  sorry

#check first_interest_rate_is_five_percent

end first_interest_rate_is_five_percent_l3544_354422


namespace rectangle_existence_l3544_354417

theorem rectangle_existence (s d : ℝ) (hs : s > 0) (hd : d > 0) :
  ∃ (a b : ℝ), 2 * (a + b) = s ∧ a^2 + b^2 = d^2 ∧ a > 0 ∧ b > 0 := by
  sorry

end rectangle_existence_l3544_354417


namespace vector_multiplication_and_addition_l3544_354454

theorem vector_multiplication_and_addition :
  (3 : ℝ) • ((-3 : ℝ), (2 : ℝ), (-5 : ℝ)) + ((4 : ℝ), (10 : ℝ), (-6 : ℝ)) = 
  ((-5 : ℝ), (16 : ℝ), (-21 : ℝ)) := by sorry

end vector_multiplication_and_addition_l3544_354454


namespace shopping_change_calculation_l3544_354406

def book_price : ℝ := 25
def pen_price : ℝ := 4
def ruler_price : ℝ := 1
def notebook_price : ℝ := 8
def pencil_case_price : ℝ := 6
def book_discount : ℝ := 0.1
def pen_discount : ℝ := 0.05
def sales_tax_rate : ℝ := 0.06
def payment : ℝ := 100

theorem shopping_change_calculation :
  let discounted_book_price := book_price * (1 - book_discount)
  let discounted_pen_price := pen_price * (1 - pen_discount)
  let total_before_tax := discounted_book_price + discounted_pen_price + ruler_price + notebook_price + pencil_case_price
  let tax_amount := total_before_tax * sales_tax_rate
  let total_with_tax := total_before_tax + tax_amount
  let change := payment - total_with_tax
  change = 56.22 := by sorry

end shopping_change_calculation_l3544_354406


namespace tangent_fraction_equality_l3544_354497

theorem tangent_fraction_equality (α : Real) (h : Real.tan α = 3) :
  (2 * Real.sin α - Real.cos α) / (Real.sin α + 3 * Real.cos α) = 5/6 := by
  sorry

end tangent_fraction_equality_l3544_354497


namespace solve_textbook_problems_l3544_354466

/-- The number of days it takes to solve all problems -/
def solve_duration (total_problems : ℕ) (problems_left_day3 : ℕ) : ℕ :=
  let problems_solved_day3 := total_problems - problems_left_day3
  let z := problems_solved_day3 / 3
  let daily_problems := List.range 7 |>.map (fun i => z + 1 - i)
  daily_problems.length

/-- Theorem stating that it takes 7 days to solve all problems under given conditions -/
theorem solve_textbook_problems :
  solve_duration 91 46 = 7 := by
  sorry

end solve_textbook_problems_l3544_354466


namespace cookie_batches_for_workshop_l3544_354416

/-- Calculates the minimum number of full batches of cookies needed for a math competition workshop --/
def min_cookie_batches (base_students : ℕ) (additional_students : ℕ) (cookies_per_student : ℕ) (cookies_per_batch : ℕ) : ℕ :=
  let total_students := base_students + additional_students
  let total_cookies_needed := total_students * cookies_per_student
  (total_cookies_needed + cookies_per_batch - 1) / cookies_per_batch

/-- Proves that 16 batches are needed for the given conditions --/
theorem cookie_batches_for_workshop : 
  min_cookie_batches 90 15 3 20 = 16 := by
sorry

end cookie_batches_for_workshop_l3544_354416


namespace inequality_solution_l3544_354494

theorem inequality_solution (x : ℝ) : 
  (1 / (x * (x + 1)) - 1 / ((x + 1) * (x + 2)) < 1 / 4) ↔ 
  (x < -2 ∨ (-1 < x ∧ x < 0) ∨ 2 < x) :=
sorry

end inequality_solution_l3544_354494


namespace local_minimum_at_two_l3544_354453

-- Define the function f
def f (c : ℝ) (x : ℝ) : ℝ := x * (x - c)^2

-- State the theorem
theorem local_minimum_at_two (c : ℝ) : 
  (∀ h, h > 0 → ∃ δ > 0, ∀ x, 0 < |x - 2| ∧ |x - 2| < δ → f c x ≥ f c 2) → 
  c = 2 := by
  sorry

end local_minimum_at_two_l3544_354453


namespace max_divisors_sympathetic_l3544_354421

/-- A number is sympathetic if for each of its divisors d, d+2 is prime. -/
def Sympathetic (n : ℕ) : Prop :=
  ∀ d : ℕ, d ∣ n → Nat.Prime (d + 2)

/-- The maximum number of divisors a sympathetic number can have is 8. -/
theorem max_divisors_sympathetic :
  ∃ n : ℕ, Sympathetic n ∧ (∀ m : ℕ, Sympathetic m → Nat.card (Nat.divisors m) ≤ Nat.card (Nat.divisors n)) ∧
    Nat.card (Nat.divisors n) = 8 :=
sorry

end max_divisors_sympathetic_l3544_354421


namespace smallest_positive_t_value_l3544_354498

theorem smallest_positive_t_value (p q r s t : ℤ) : 
  (∀ x : ℝ, p * x^4 + q * x^3 + r * x^2 + s * x + t = 0 ↔ x = -3 ∨ x = 4 ∨ x = 6 ∨ x = 1/2) →
  t > 0 →
  (∀ t' : ℤ, t' > 0 ∧ (∀ x : ℝ, p * x^4 + q * x^3 + r * x^2 + s * x + t' = 0 ↔ x = -3 ∨ x = 4 ∨ x = 6 ∨ x = 1/2) → t' ≥ t) →
  t = 72 := by
sorry

end smallest_positive_t_value_l3544_354498


namespace masha_floor_number_l3544_354468

/-- Represents a multi-story apartment building -/
structure ApartmentBuilding where
  floors : ℕ
  entrances : ℕ
  apartments_per_floor : ℕ

/-- Calculates the floor number given an apartment number and building structure -/
def floor_number (building : ApartmentBuilding) (apartment_number : ℕ) : ℕ :=
  sorry

theorem masha_floor_number :
  let building := ApartmentBuilding.mk 17 4 5
  let masha_apartment := 290
  floor_number building masha_apartment = 7 := by
  sorry

end masha_floor_number_l3544_354468


namespace union_of_sets_l3544_354430

theorem union_of_sets : 
  let A : Set ℕ := {1, 2, 3}
  let B : Set ℕ := {2, 3, 4}
  A ∪ B = {1, 2, 3, 4} := by
sorry

end union_of_sets_l3544_354430


namespace william_bottle_caps_l3544_354481

/-- The number of bottle caps William has in total -/
def total_bottle_caps (initial : ℕ) (additional : ℕ) : ℕ :=
  initial + additional

/-- Theorem stating that William has 43 bottle caps in total -/
theorem william_bottle_caps : 
  total_bottle_caps 2 41 = 43 := by
  sorry

end william_bottle_caps_l3544_354481


namespace base4_1302_equals_base5_424_l3544_354474

/-- Converts a base 4 number to base 10 --/
def base4ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (4 ^ i)) 0

/-- Converts a base 10 number to base 5 --/
def base10ToBase5 (n : Nat) : List Nat :=
  if n = 0 then [0] else
    let rec go (m : Nat) (acc : List Nat) : List Nat :=
      if m = 0 then acc
      else go (m / 5) ((m % 5) :: acc)
    go n []

theorem base4_1302_equals_base5_424 :
  base10ToBase5 (base4ToBase10 [2, 0, 3, 1]) = [4, 2, 4] := by
  sorry

end base4_1302_equals_base5_424_l3544_354474


namespace car_speed_second_hour_l3544_354446

/-- Given a car's speed in the first hour and its average speed over two hours,
    calculate its speed in the second hour. -/
theorem car_speed_second_hour
  (speed_first_hour : ℝ)
  (average_speed : ℝ)
  (h1 : speed_first_hour = 90)
  (h2 : average_speed = 65)
  : ∃ (speed_second_hour : ℝ),
    speed_second_hour = 40 ∧
    (speed_first_hour + speed_second_hour) / 2 = average_speed :=
by
  sorry

#check car_speed_second_hour

end car_speed_second_hour_l3544_354446


namespace chris_age_l3544_354436

theorem chris_age (a b c : ℕ) : 
  (a + b + c) / 3 = 9 →  -- The average of their ages is 9
  c - 4 = a →            -- Four years ago, Chris was Amy's current age
  b + 3 = 2 * (a + 3) / 3 →  -- In 3 years, Ben's age will be 2/3 of Amy's age
  c = 13 :=               -- Chris's current age is 13
by sorry

end chris_age_l3544_354436


namespace additional_round_trips_l3544_354492

/-- Represents the number of passengers on a one-way trip -/
def one_way_passengers : ℕ := 100

/-- Represents the number of passengers on a return trip -/
def return_passengers : ℕ := 60

/-- Represents the total number of passengers transported that day -/
def total_passengers : ℕ := 640

/-- Calculates the number of passengers in one round trip -/
def passengers_per_round_trip : ℕ := one_way_passengers + return_passengers

/-- Theorem: The number of additional round trips is 3 -/
theorem additional_round_trips :
  (total_passengers - passengers_per_round_trip) / passengers_per_round_trip = 3 := by
  sorry

end additional_round_trips_l3544_354492


namespace sequence_properties_l3544_354490

def a (n : ℕ+) : ℤ := n * (n - 8) - 20

theorem sequence_properties :
  (∃ (k : ℕ), k = 9 ∧ ∀ n : ℕ+, a n < 0 ↔ n.val ≤ k) ∧
  (∀ n : ℕ+, n ≥ 4 → a (n + 1) > a n) ∧
  (∀ n : ℕ+, a n ≥ a 4 ∧ a 4 = -36) :=
sorry

end sequence_properties_l3544_354490


namespace factorization_equality_l3544_354456

theorem factorization_equality (x y : ℝ) : y - 2*x*y + x^2*y = y*(1-x)^2 := by
  sorry

end factorization_equality_l3544_354456


namespace ice_cream_arrangements_l3544_354401

theorem ice_cream_arrangements : (Nat.factorial 5) = 120 := by
  sorry

end ice_cream_arrangements_l3544_354401


namespace max_value_of_expression_l3544_354405

theorem max_value_of_expression (x y : ℝ) (h : x + y = 5) :
  (∃ m : ℝ, ∀ a b : ℝ, a + b = 5 → 
    x^5*y + x^4*y^2 + x^3*y^3 + x^2*y^4 + x*y^5 ≤ m ∧
    x^5*y + x^4*y^2 + x^3*y^3 + x^2*y^4 + x*y^5 = m) ∧
  (∀ m : ℝ, (∀ a b : ℝ, a + b = 5 → 
    x^5*y + x^4*y^2 + x^3*y^3 + x^2*y^4 + x*y^5 ≤ m ∧
    x^5*y + x^4*y^2 + x^3*y^3 + x^2*y^4 + x*y^5 = m) → 
  m = 625/4) :=
sorry

end max_value_of_expression_l3544_354405


namespace circle_area_special_condition_l3544_354434

theorem circle_area_special_condition (r : ℝ) (h : (2 * r)^2 = 8 * (2 * π * r)) :
  π * r^2 = 16 * π^3 := by
  sorry

end circle_area_special_condition_l3544_354434


namespace pi_is_irrational_l3544_354455

theorem pi_is_irrational : Irrational Real.pi := by sorry

end pi_is_irrational_l3544_354455


namespace good_goods_sufficient_for_not_cheap_l3544_354445

-- Define the propositions
def good_goods : Prop := sorry
def not_cheap : Prop := sorry

-- Define Sister Qian's statement
def sister_qian_statement : Prop := good_goods → not_cheap

-- Theorem to prove
theorem good_goods_sufficient_for_not_cheap :
  sister_qian_statement → (∃ p q : Prop, (p → q) ∧ (p = good_goods) ∧ (q = not_cheap)) :=
by sorry

end good_goods_sufficient_for_not_cheap_l3544_354445


namespace tree_height_difference_l3544_354411

theorem tree_height_difference :
  let apple_tree_height : ℚ := 53 / 4
  let cherry_tree_height : ℚ := 147 / 8
  cherry_tree_height - apple_tree_height = 41 / 8 := by sorry

end tree_height_difference_l3544_354411


namespace constant_term_expansion_l3544_354465

/-- The constant term in the expansion of (x - 2/x^2)^9 is -672 -/
theorem constant_term_expansion : 
  let f : ℝ → ℝ := fun x ↦ (x - 2 / x^2)^9
  ∃ c : ℝ, ∀ x : ℝ, x ≠ 0 → f x = c + x * (f x - c) / x ∧ c = -672 :=
sorry

end constant_term_expansion_l3544_354465


namespace pattern_paths_count_l3544_354403

/-- Represents a position in the diagram -/
structure Position :=
  (row : ℕ) (col : ℕ)

/-- Represents a letter in the diagram -/
inductive Letter
  | P | A | T | E | R | N | C | O

/-- The diagram of letters -/
def diagram : List (List Letter) := sorry

/-- Checks if two positions are adjacent -/
def adjacent (p1 p2 : Position) : Prop := sorry

/-- Checks if a path spells "PATTERN" -/
def spells_pattern (path : List Position) : Prop := sorry

/-- Counts the number of valid paths spelling "PATTERN" -/
def count_pattern_paths : ℕ := sorry

/-- The main theorem to prove -/
theorem pattern_paths_count :
  count_pattern_paths = 18 := by sorry

end pattern_paths_count_l3544_354403


namespace angle_in_first_quadrant_l3544_354489

theorem angle_in_first_quadrant (α : Real) 
  (h1 : Real.tan α > 0) 
  (h2 : Real.sin α + Real.cos α > 0) : 
  0 < α ∧ α < Real.pi / 2 := by
  sorry

end angle_in_first_quadrant_l3544_354489


namespace lose_condition_win_condition_rattle_count_l3544_354461

/-- The number of rattles Twalley has -/
def t : ℕ := 7

/-- The number of rattles Tweerley has -/
def r : ℕ := 5

/-- If Twalley loses the bet, he will have the same number of rattles as Tweerley -/
theorem lose_condition : t - 1 = r + 1 := by sorry

/-- If Twalley wins the bet, he will have twice as many rattles as Tweerley -/
theorem win_condition : t + 1 = 2 * (r - 1) := by sorry

/-- Prove that given the conditions of the bet, Twalley must have 7 rattles and Tweerley must have 5 rattles -/
theorem rattle_count : t = 7 ∧ r = 5 := by sorry

end lose_condition_win_condition_rattle_count_l3544_354461


namespace contrapositive_equivalence_l3544_354432

theorem contrapositive_equivalence (p q : Prop) :
  (p → q) → (¬q → ¬p) := by sorry

end contrapositive_equivalence_l3544_354432


namespace choir_average_age_l3544_354486

/-- The average age of people in a choir given the number and average age of females and males -/
theorem choir_average_age 
  (num_females : ℕ) 
  (num_males : ℕ) 
  (avg_age_females : ℚ) 
  (avg_age_males : ℚ) 
  (h1 : num_females = 12) 
  (h2 : num_males = 18) 
  (h3 : avg_age_females = 28) 
  (h4 : avg_age_males = 32) :
  let total_people := num_females + num_males
  let total_age := num_females * avg_age_females + num_males * avg_age_males
  total_age / total_people = 30.4 := by
sorry


end choir_average_age_l3544_354486


namespace quadratic_minimum_l3544_354431

theorem quadratic_minimum : 
  (∃ (y : ℝ), y^2 - 6*y + 5 = -4) ∧ 
  (∀ (y : ℝ), y^2 - 6*y + 5 ≥ -4) := by
sorry

end quadratic_minimum_l3544_354431


namespace money_ratio_l3544_354449

theorem money_ratio (bob phil jenna : ℚ) : 
  bob = 60 →
  phil = (1/3) * bob →
  jenna = bob - 20 →
  jenna / phil = 2 := by
sorry

end money_ratio_l3544_354449


namespace laundry_synchronization_l3544_354471

def ronald_cycle : ℕ := 6
def tim_cycle : ℕ := 9
def laura_cycle : ℕ := 12
def dani_cycle : ℕ := 15
def laura_birthday : ℕ := 35

theorem laundry_synchronization (ronald_cycle tim_cycle laura_cycle dani_cycle laura_birthday : ℕ) 
  (h1 : ronald_cycle = 6)
  (h2 : tim_cycle = 9)
  (h3 : laura_cycle = 12)
  (h4 : dani_cycle = 15)
  (h5 : laura_birthday = 35) :
  ∃ (next_sync : ℕ), next_sync - laura_birthday = 145 ∧ 
  next_sync % ronald_cycle = 0 ∧
  next_sync % tim_cycle = 0 ∧
  next_sync % laura_cycle = 0 ∧
  next_sync % dani_cycle = 0 :=
by sorry

end laundry_synchronization_l3544_354471


namespace p_or_q_iff_m_in_range_l3544_354418

def p (m : ℝ) : Prop := ∀ x : ℝ, x^2 - m*x + 3/2 > 0

def q (m : ℝ) : Prop := 
  (m - 1 > 0) ∧ (3 - m > 0) ∧ 
  ∃ c : ℝ, c^2 = (m - 1)*(3 - m) ∧ 
  ∀ x y : ℝ, x^2/(m-1) + y^2/(3-m) = 1 → x^2 + y^2 = (m-1)^2/(m-1) ∨ x^2 + y^2 = (3-m)^2/(3-m)

theorem p_or_q_iff_m_in_range (m : ℝ) : 
  p m ∨ q m ↔ m > -Real.sqrt 6 ∧ m < 3 :=
sorry

end p_or_q_iff_m_in_range_l3544_354418


namespace maryann_working_time_l3544_354487

/-- Maryann's working time calculation -/
theorem maryann_working_time 
  (time_calling : ℕ) 
  (accounting_ratio : ℕ) 
  (h1 : time_calling = 70) 
  (h2 : accounting_ratio = 7) : 
  time_calling + accounting_ratio * time_calling = 560 := by
  sorry

end maryann_working_time_l3544_354487


namespace fifth_score_calculation_l3544_354441

theorem fifth_score_calculation (s1 s2 s3 s4 : ℕ) (avg : ℚ) (h1 : s1 = 65) (h2 : s2 = 67) (h3 : s3 = 76) (h4 : s4 = 80) (h5 : avg = 76.6) :
  ∃ (s5 : ℕ), s5 = 95 ∧ (s1 + s2 + s3 + s4 + s5 : ℚ) / 5 = avg :=
sorry

end fifth_score_calculation_l3544_354441


namespace squats_on_fourth_day_l3544_354496

def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

def squats_on_day (initial_squats : ℕ) (day : ℕ) : ℕ :=
  match day with
  | 0 => initial_squats
  | n + 1 => squats_on_day initial_squats n + factorial n

theorem squats_on_fourth_day (initial_squats : ℕ) :
  initial_squats = 30 → squats_on_day initial_squats 3 = 39 := by
  sorry

end squats_on_fourth_day_l3544_354496


namespace ratio_a_to_c_l3544_354470

theorem ratio_a_to_c (a b c d : ℚ) 
  (hab : a / b = 5 / 4)
  (hcd : c / d = 2 / 1)
  (hdb : d / b = 1 / 8) :
  a / c = 5 / 1 := by
sorry

end ratio_a_to_c_l3544_354470


namespace gasoline_added_l3544_354414

theorem gasoline_added (tank_capacity : ℝ) (initial_fill : ℝ) (final_fill : ℝ) : tank_capacity = 29.999999999999996 → initial_fill = 3/4 → final_fill = 9/10 → (final_fill - initial_fill) * tank_capacity = 4.499999999999999 := by
  sorry

end gasoline_added_l3544_354414
