import Mathlib

namespace NUMINAMATH_CALUDE_no_consecutive_heads_probability_l3626_362631

/-- The number of ways to toss n coins such that no two heads appear consecutively -/
def f : ℕ → ℕ
| 0 => 1  -- Convention for empty sequence
| 1 => 2  -- Base case
| 2 => 3  -- Base case
| (n + 3) => f (n + 2) + f (n + 1)

/-- The probability of no two heads appearing consecutively in 10 coin tosses -/
theorem no_consecutive_heads_probability :
  (f 10 : ℚ) / (2^10 : ℚ) = 9/64 := by sorry

end NUMINAMATH_CALUDE_no_consecutive_heads_probability_l3626_362631


namespace NUMINAMATH_CALUDE_share_investment_interest_rate_l3626_362675

/-- Calculates the interest rate for a share investment -/
theorem share_investment_interest_rate 
  (face_value : ℝ) 
  (dividend_rate : ℝ) 
  (market_value : ℝ) 
  (h1 : face_value = 52) 
  (h2 : dividend_rate = 0.09) 
  (h3 : market_value = 39) : 
  (dividend_rate * face_value) / market_value = 0.12 := by
  sorry

#check share_investment_interest_rate

end NUMINAMATH_CALUDE_share_investment_interest_rate_l3626_362675


namespace NUMINAMATH_CALUDE_min_sum_abc_l3626_362676

def is_min_sum (a b c : ℕ) : Prop :=
  ∀ x y z : ℕ, 
    (Nat.lcm (Nat.lcm x y) z = 48) → 
    (Nat.gcd x y = 4) → 
    (Nat.gcd y z = 3) → 
    a + b + c ≤ x + y + z

theorem min_sum_abc : 
  ∃ a b c : ℕ,
    (Nat.lcm (Nat.lcm a b) c = 48) ∧ 
    (Nat.gcd a b = 4) ∧ 
    (Nat.gcd b c = 3) ∧ 
    (is_min_sum a b c) ∧ 
    (a + b + c = 31) :=
sorry

end NUMINAMATH_CALUDE_min_sum_abc_l3626_362676


namespace NUMINAMATH_CALUDE_circle_radius_l3626_362616

/-- The radius of a circle defined by the equation x^2 + 2x + y^2 = 0 is 1 -/
theorem circle_radius (x y : ℝ) : x^2 + 2*x + y^2 = 0 → ∃ (c : ℝ × ℝ), (x - c.1)^2 + (y - c.2)^2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_circle_radius_l3626_362616


namespace NUMINAMATH_CALUDE_simplified_fraction_sum_l3626_362607

theorem simplified_fraction_sum (a b : ℕ) (h : a = 75 ∧ b = 100) :
  ∃ (c d : ℕ), (c.gcd d = 1) ∧ (a * d = b * c) ∧ (c + d = 7) := by
  sorry

end NUMINAMATH_CALUDE_simplified_fraction_sum_l3626_362607


namespace NUMINAMATH_CALUDE_negation_of_universal_quadratic_inequality_l3626_362657

theorem negation_of_universal_quadratic_inequality :
  ¬(∀ x : ℝ, x^2 - 2*x + 1 ≥ 0) ↔ ∃ x : ℝ, x^2 - 2*x + 1 < 0 :=
by sorry

end NUMINAMATH_CALUDE_negation_of_universal_quadratic_inequality_l3626_362657


namespace NUMINAMATH_CALUDE_line_through_circle_center_l3626_362627

/-- Given a line and a circle, if the line passes through the center of the circle,
    then the value of m in the line equation is 0. -/
theorem line_through_circle_center (m : ℝ) : 
  (∀ x y : ℝ, x^2 + y^2 - 2*x + 4*y = 0 → 
    ∃ h k : ℝ, (h - 1)^2 + (k + 2)^2 = 0 ∧ 2*h + k + m = 0) → 
  m = 0 := by
sorry

end NUMINAMATH_CALUDE_line_through_circle_center_l3626_362627


namespace NUMINAMATH_CALUDE_range_of_a_l3626_362647

-- Define the functions f and g
def f (a : ℝ) (x : ℝ) : ℝ := |x + a| + |x + 3|
def g (x : ℝ) : ℝ := |x - 1| + 2

-- State the theorem
theorem range_of_a (a : ℝ) : 
  (∀ x₁ : ℝ, ∃ x₂ : ℝ, f a x₁ = g x₂) → 
  a ≥ 5 ∨ a ≤ 1 := by sorry

end NUMINAMATH_CALUDE_range_of_a_l3626_362647


namespace NUMINAMATH_CALUDE_blue_packs_bought_l3626_362624

/- Define the problem parameters -/
def white_pack_size : ℕ := 6
def blue_pack_size : ℕ := 9
def white_packs_bought : ℕ := 5
def total_tshirts : ℕ := 57

/- Define the theorem -/
theorem blue_packs_bought :
  ∃ (blue_packs : ℕ),
    blue_packs * blue_pack_size + white_packs_bought * white_pack_size = total_tshirts ∧
    blue_packs = 3 := by
  sorry

end NUMINAMATH_CALUDE_blue_packs_bought_l3626_362624


namespace NUMINAMATH_CALUDE_events_B_C_complementary_l3626_362673

-- Define the sample space (cube faces)
def Ω : Set ℕ := {1, 2, 3, 4, 5, 6}

-- Define events A, B, and C
def A : Set ℕ := {n ∈ Ω | n % 2 = 1}
def B : Set ℕ := {n ∈ Ω | n ≤ 3}
def C : Set ℕ := {n ∈ Ω | n ≥ 4}

-- Theorem to prove
theorem events_B_C_complementary : B ∪ C = Ω ∧ B ∩ C = ∅ := by
  sorry

end NUMINAMATH_CALUDE_events_B_C_complementary_l3626_362673


namespace NUMINAMATH_CALUDE_probability_four_white_balls_l3626_362619

def total_balls : ℕ := 25
def white_balls : ℕ := 10
def black_balls : ℕ := 15
def drawn_balls : ℕ := 4

theorem probability_four_white_balls : 
  (Nat.choose white_balls drawn_balls : ℚ) / (Nat.choose total_balls drawn_balls) = 3 / 181 :=
sorry

end NUMINAMATH_CALUDE_probability_four_white_balls_l3626_362619


namespace NUMINAMATH_CALUDE_equation_solution_l3626_362677

theorem equation_solution : ∃ x : ℝ, 30 - (5 * 2) = 3 + x ∧ x = 17 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3626_362677


namespace NUMINAMATH_CALUDE_unique_solution_star_l3626_362636

/-- The ⋆ operation -/
def star (x y : ℝ) : ℝ := 5*x - 2*y + 3*x*y

/-- Theorem stating that 2 ⋆ y = 10 has a unique solution y = 0 -/
theorem unique_solution_star :
  ∃! y : ℝ, star 2 y = 10 :=
sorry

end NUMINAMATH_CALUDE_unique_solution_star_l3626_362636


namespace NUMINAMATH_CALUDE_equation_solutions_l3626_362615

theorem equation_solutions : 
  {x : ℝ | (2 + x)^(2/3) + 3 * (2 - x)^(2/3) = 4 * (4 - x^2)^(1/3)} = {0, 13/7} := by
  sorry

end NUMINAMATH_CALUDE_equation_solutions_l3626_362615


namespace NUMINAMATH_CALUDE_distribute_seven_balls_four_boxes_l3626_362623

/-- Represents the number of ways to distribute indistinguishable balls into indistinguishable boxes -/
def distribute (balls : ℕ) (boxes : ℕ) (min_per_box : ℕ) : ℕ :=
  sorry

/-- The main theorem stating that there are 3 ways to distribute 7 balls into 4 boxes -/
theorem distribute_seven_balls_four_boxes :
  distribute 7 4 1 = 3 := by sorry

end NUMINAMATH_CALUDE_distribute_seven_balls_four_boxes_l3626_362623


namespace NUMINAMATH_CALUDE_equation_solutions_l3626_362614

theorem equation_solutions (x : ℝ) : 
  (x - 1)^2 * (x - 5)^2 / (x - 5) = 4 ↔ x = 3 ∨ x = -1 :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l3626_362614


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l3626_362678

/-- The line y = kx + 1 and the parabola y^2 = 4x have only one common point -/
def has_one_common_point (k : ℝ) : Prop :=
  ∃! x y, y = k * x + 1 ∧ y^2 = 4 * x

theorem sufficient_not_necessary :
  (∀ k, k = 0 → has_one_common_point k) ∧
  (∃ k, k ≠ 0 ∧ has_one_common_point k) :=
sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l3626_362678


namespace NUMINAMATH_CALUDE_bills_problem_l3626_362649

/-- Represents the bill amount for a person -/
structure Bill where
  amount : ℝ
  tipPercentage : ℝ
  tipAmount : ℝ

/-- The problem statement -/
theorem bills_problem (mike : Bill) (joe : Bill) (bill : Bill)
  (h_mike : mike.tipPercentage = 0.10 ∧ mike.tipAmount = 3)
  (h_joe : joe.tipPercentage = 0.15 ∧ joe.tipAmount = 4.5)
  (h_bill : bill.tipPercentage = 0.25 ∧ bill.tipAmount = 5) :
  bill.amount = 20 := by
  sorry


end NUMINAMATH_CALUDE_bills_problem_l3626_362649


namespace NUMINAMATH_CALUDE_subway_length_l3626_362632

/-- The length of a subway given its speed, time to cross a bridge, and bridge length. -/
theorem subway_length
  (speed : ℝ)  -- Speed of the subway in km/min
  (time : ℝ)   -- Time to cross the bridge in minutes
  (bridge_length : ℝ)  -- Length of the bridge in km
  (h1 : speed = 1.6)  -- The subway speed is 1.6 km/min
  (h2 : time = 3.25)  -- The time to cross the bridge is 3 min and 15 sec (3.25 min)
  (h3 : bridge_length = 4.85)  -- The bridge length is 4.85 km
  : (speed * time - bridge_length) * 1000 = 350 :=
by sorry

end NUMINAMATH_CALUDE_subway_length_l3626_362632


namespace NUMINAMATH_CALUDE_max_salary_is_220000_l3626_362698

/-- Represents a basketball team with salary constraints -/
structure BasketballTeam where
  num_players : ℕ
  min_salary : ℕ
  salary_cap : ℕ

/-- Calculates the maximum possible salary for the highest-paid player -/
def max_highest_salary (team : BasketballTeam) : ℕ :=
  team.salary_cap - (team.num_players - 1) * team.min_salary

/-- Theorem stating the maximum possible salary for the highest-paid player -/
theorem max_salary_is_220000 (team : BasketballTeam) 
  (h1 : team.num_players = 15)
  (h2 : team.min_salary = 20000)
  (h3 : team.salary_cap = 500000) :
  max_highest_salary team = 220000 := by
  sorry

#eval max_highest_salary { num_players := 15, min_salary := 20000, salary_cap := 500000 }

end NUMINAMATH_CALUDE_max_salary_is_220000_l3626_362698


namespace NUMINAMATH_CALUDE_basketball_game_score_l3626_362620

/-- Represents the score of a team in a quarter -/
structure QuarterScore where
  score : ℕ
  valid : score ≤ 25

/-- Represents the scores of a team for all four quarters -/
structure GameScore where
  q1 : QuarterScore
  q2 : QuarterScore
  q3 : QuarterScore
  q4 : QuarterScore
  increasing : q1.score < q2.score ∧ q2.score < q3.score ∧ q3.score < q4.score
  arithmetic : ∃ d : ℕ, q2.score = q1.score + d ∧ q3.score = q2.score + d ∧ q4.score = q3.score + d

def total_score (g : GameScore) : ℕ :=
  g.q1.score + g.q2.score + g.q3.score + g.q4.score

def first_half_score (g : GameScore) : ℕ :=
  g.q1.score + g.q2.score

theorem basketball_game_score :
  ∀ raiders wildcats : GameScore,
    raiders.q1 = wildcats.q1 →  -- First quarter tie
    total_score raiders = total_score wildcats + 2 →  -- Raiders win by 2
    first_half_score raiders + first_half_score wildcats = 38 :=
by sorry

end NUMINAMATH_CALUDE_basketball_game_score_l3626_362620


namespace NUMINAMATH_CALUDE_quadratic_coefficient_l3626_362662

theorem quadratic_coefficient (b : ℝ) :
  (b < 0) →
  (∃ m : ℝ, ∀ x : ℝ, x^2 + b*x + 1/4 = (x + m)^2 + 1/16) →
  b = -Real.sqrt 3 / 2 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_coefficient_l3626_362662


namespace NUMINAMATH_CALUDE_ellipse_sum_l3626_362680

/-- Represents an ellipse with center (h, k) and semi-axes lengths a and b -/
structure Ellipse where
  h : ℝ
  k : ℝ
  a : ℝ
  b : ℝ

/-- The equation of the ellipse -/
def ellipse_equation (e : Ellipse) (x y : ℝ) : Prop :=
  (x - e.h)^2 / e.a^2 + (y - e.k)^2 / e.b^2 = 1

theorem ellipse_sum (e : Ellipse) :
  e.h = 3 ∧ e.k = -5 ∧ e.a = 7 ∧ e.b = 4 →
  e.h + e.k + e.a + e.b = 9 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_sum_l3626_362680


namespace NUMINAMATH_CALUDE_multiplication_of_monomials_l3626_362645

theorem multiplication_of_monomials (a : ℝ) : 3 * a * (4 * a^2) = 12 * a^3 := by
  sorry

end NUMINAMATH_CALUDE_multiplication_of_monomials_l3626_362645


namespace NUMINAMATH_CALUDE_length_BC_is_sqrt_13_l3626_362664

/-- The cosine theorem for a triangle ABC -/
def cosine_theorem (a b c : ℝ) (A : ℝ) : Prop :=
  c^2 = a^2 + b^2 - 2*a*b*(Real.cos A)

/-- Triangle ABC with given side lengths and angle -/
structure Triangle :=
  (AB : ℝ)
  (AC : ℝ)
  (angle_A : ℝ)
  (h_AB_pos : AB > 0)
  (h_AC_pos : AC > 0)
  (h_angle_A_pos : angle_A > 0)
  (h_angle_A_lt_pi : angle_A < π)

theorem length_BC_is_sqrt_13 (t : Triangle) 
  (h_AB : t.AB = 3)
  (h_AC : t.AC = 4)
  (h_angle_A : t.angle_A = π/3) :
  ∃ BC : ℝ, BC > 0 ∧ BC^2 = 13 ∧ cosine_theorem t.AB t.AC BC t.angle_A :=
sorry

end NUMINAMATH_CALUDE_length_BC_is_sqrt_13_l3626_362664


namespace NUMINAMATH_CALUDE_rhombus_area_l3626_362601

theorem rhombus_area (d₁ d₂ : ℝ) : 
  d₁^2 - 5*d₁ + 6 = 0 → 
  d₂^2 - 5*d₂ + 6 = 0 → 
  d₁ ≠ d₂ →
  (d₁ * d₂) / 2 = 3 := by
sorry

end NUMINAMATH_CALUDE_rhombus_area_l3626_362601


namespace NUMINAMATH_CALUDE_min_value_expression_l3626_362661

theorem min_value_expression (n : ℕ) (hn : n > 0) :
  (n : ℝ) / 3 + 27 / n ≥ 6 ∧
  ((n : ℝ) / 3 + 27 / n = 6 ↔ n = 9) :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l3626_362661


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l3626_362639

def A : Set ℕ := {1, 2, 3, 4}
def B : Set ℕ := {1, 3, 5, 7}

theorem intersection_of_A_and_B : A ∩ B = {1, 3} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l3626_362639


namespace NUMINAMATH_CALUDE_torn_pages_sum_not_1990_l3626_362690

/-- Represents a sheet in the notebook -/
structure Sheet :=
  (number : ℕ)
  (h_range : number ≥ 1 ∧ number ≤ 96)

/-- The sum of page numbers on a sheet -/
def sheet_sum (s : Sheet) : ℕ := 4 * s.number - 1

/-- A selection of 25 sheets -/
def SheetSelection := { sel : Finset Sheet // sel.card = 25 }

theorem torn_pages_sum_not_1990 (sel : SheetSelection) :
  (sel.val.sum sheet_sum) ≠ 1990 := by
  sorry


end NUMINAMATH_CALUDE_torn_pages_sum_not_1990_l3626_362690


namespace NUMINAMATH_CALUDE_f_lower_bound_l3626_362692

noncomputable section

variables (a x : ℝ)

def f (a x : ℝ) : ℝ := (1/2) * a * x^2 + (2*a - 1) * x - 2 * Real.log x

theorem f_lower_bound (ha : a > 0) (hx : x > 0) :
  f a x ≥ 4 - (5/(2*a)) := by sorry

end NUMINAMATH_CALUDE_f_lower_bound_l3626_362692


namespace NUMINAMATH_CALUDE_cubic_monotonicity_implies_one_intersection_one_intersection_not_implies_monotonicity_l3626_362686

-- Define the cubic function
def f (a b c d x : ℝ) : ℝ := a * x^3 + b * x^2 + c * x + d

-- Define strict monotonicity
def strictly_monotonic (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x < f y ∨ (∀ x y, x < y → f x > f y)

-- Define the property of intersecting x-axis exactly once
def intersects_x_axis_once (f : ℝ → ℝ) : Prop :=
  ∃! x, f x = 0

-- Main theorem
theorem cubic_monotonicity_implies_one_intersection
  (a b c d : ℝ) (h : a ≠ 0) :
  strictly_monotonic (f a b c d) →
  intersects_x_axis_once (f a b c d) :=
sorry

-- Counterexample theorem
theorem one_intersection_not_implies_monotonicity :
  ∃ a b c d : ℝ,
    intersects_x_axis_once (f a b c d) ∧
    ¬strictly_monotonic (f a b c d) :=
sorry

end NUMINAMATH_CALUDE_cubic_monotonicity_implies_one_intersection_one_intersection_not_implies_monotonicity_l3626_362686


namespace NUMINAMATH_CALUDE_marys_remaining_cards_l3626_362626

theorem marys_remaining_cards (initial_cards promised_cards bought_cards : ℝ) :
  initial_cards + bought_cards - promised_cards =
  initial_cards + bought_cards - promised_cards :=
by sorry

end NUMINAMATH_CALUDE_marys_remaining_cards_l3626_362626


namespace NUMINAMATH_CALUDE_scientific_notation_correct_l3626_362670

/-- Scientific notation representation of a number -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  is_valid : 1 ≤ coefficient ∧ coefficient < 10

/-- The number to be expressed in scientific notation -/
def number : ℕ := 3050000

/-- The scientific notation representation of the number -/
def scientific_representation : ScientificNotation := {
  coefficient := 3.05,
  exponent := 6,
  is_valid := by sorry
}

/-- Theorem stating that the scientific notation representation is correct -/
theorem scientific_notation_correct :
  (scientific_representation.coefficient * (10 : ℝ) ^ scientific_representation.exponent) = number := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_correct_l3626_362670


namespace NUMINAMATH_CALUDE_unique_assignment_l3626_362634

/- Define the girls and colors as enums -/
inductive Girl : Type
  | Katya | Olya | Liza | Rita

inductive Color : Type
  | Pink | Green | Yellow | Blue

/- Define the assignment of colors to girls -/
def assignment : Girl → Color
  | Girl.Katya => Color.Green
  | Girl.Olya => Color.Blue
  | Girl.Liza => Color.Pink
  | Girl.Rita => Color.Yellow

/- Define the circular arrangement of girls -/
def nextGirl : Girl → Girl
  | Girl.Katya => Girl.Olya
  | Girl.Olya => Girl.Liza
  | Girl.Liza => Girl.Rita
  | Girl.Rita => Girl.Katya

/- Define the conditions -/
def conditions (a : Girl → Color) : Prop :=
  (a Girl.Katya ≠ Color.Pink ∧ a Girl.Katya ≠ Color.Blue) ∧
  (∃ g : Girl, a g = Color.Green ∧ 
    ((nextGirl g = Girl.Liza ∧ a (nextGirl (nextGirl g)) = Color.Yellow) ∨
     (nextGirl (nextGirl g) = Girl.Liza ∧ a (nextGirl g) = Color.Yellow))) ∧
  (a Girl.Rita ≠ Color.Green ∧ a Girl.Rita ≠ Color.Blue) ∧
  (∃ g : Girl, nextGirl g = Girl.Olya ∧ nextGirl (nextGirl g) = Girl.Rita ∧ 
    (a g = Color.Pink ∨ a (nextGirl (nextGirl (nextGirl g))) = Color.Pink))

/- Theorem statement -/
theorem unique_assignment : 
  ∀ a : Girl → Color, conditions a → a = assignment :=
sorry

end NUMINAMATH_CALUDE_unique_assignment_l3626_362634


namespace NUMINAMATH_CALUDE_no_bounded_ratio_interval_l3626_362610

theorem no_bounded_ratio_interval (a : ℝ) (ha : a > 0) :
  ¬∃ (b c : ℝ) (hbc : b < c),
    ∀ (x y : ℝ) (hx : b < x ∧ x < c) (hy : b < y ∧ y < c) (hxy : x ≠ y),
      |((x + y) / (x - y))| ≤ a :=
sorry

end NUMINAMATH_CALUDE_no_bounded_ratio_interval_l3626_362610


namespace NUMINAMATH_CALUDE_george_oranges_l3626_362694

def orange_problem (betty sandra emily frank george : ℕ) : Prop :=
  betty = 12 ∧
  sandra = 3 * betty ∧
  emily = 7 * sandra ∧
  frank = 5 * emily ∧
  george = (5/2 : ℚ) * frank

theorem george_oranges :
  ∀ betty sandra emily frank george : ℕ,
  orange_problem betty sandra emily frank george →
  george = 3150 :=
by
  sorry

end NUMINAMATH_CALUDE_george_oranges_l3626_362694


namespace NUMINAMATH_CALUDE_right_triangle_existence_l3626_362665

noncomputable def f (x : ℝ) : ℝ :=
  if x < Real.exp 1 then -x^3 + x^2 else Real.log x

theorem right_triangle_existence (a : ℝ) :
  (∃ t : ℝ, t ≥ Real.exp 1 ∧
    ((-t^2 + f t * (-t^3 + t^2) = 0) ∧
     (∃ P Q : ℝ × ℝ, P = (t, f t) ∧ Q = (-t, f (-t)) ∧
       (P.1 * Q.1 + P.2 * Q.2 = 0) ∧
       ((P.1 + Q.1) / 2 = 0))))
  ↔ (0 < a ∧ a ≤ 1 / (Real.exp 1 * Real.log (Real.exp 1) + 1)) :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_existence_l3626_362665


namespace NUMINAMATH_CALUDE_maria_eggs_l3626_362660

/-- The number of eggs Maria has -/
def total_eggs (num_boxes : ℕ) (eggs_per_box : ℕ) : ℕ :=
  num_boxes * eggs_per_box

/-- Theorem: Maria has 21 eggs in total -/
theorem maria_eggs : total_eggs 3 7 = 21 := by
  sorry

end NUMINAMATH_CALUDE_maria_eggs_l3626_362660


namespace NUMINAMATH_CALUDE_slope_range_theorem_l3626_362684

-- Define a line by its slope and a point it passes through
def Line (k : ℝ) (x₀ y₀ : ℝ) :=
  {(x, y) : ℝ × ℝ | y - y₀ = k * (x - x₀)}

-- Define the translation of a line
def translate (L : Set (ℝ × ℝ)) (dx dy : ℝ) :=
  {(x, y) : ℝ × ℝ | (x - dx, y - dy) ∈ L}

-- Define the fourth quadrant
def fourthQuadrant := {(x, y) : ℝ × ℝ | x > 0 ∧ y < 0}

theorem slope_range_theorem (k : ℝ) :
  let l := Line k 1 (-1)
  let m := translate l 3 (-2)
  (∀ p ∈ m, p ∉ fourthQuadrant) → 0 ≤ k ∧ k ≤ 1/4 := by
  sorry

end NUMINAMATH_CALUDE_slope_range_theorem_l3626_362684


namespace NUMINAMATH_CALUDE_metal_waste_l3626_362621

/-- Given a rectangle with length l and breadth b (where l > b), from which a maximum-sized
    circular piece is cut and then a maximum-sized square piece is cut from that circle,
    the total amount of metal wasted is equal to l × b - b²/2. -/
theorem metal_waste (l b : ℝ) (h : l > b) (b_pos : b > 0) :
  let circle_area := π * (b/2)^2
  let square_side := b / Real.sqrt 2
  let square_area := square_side^2
  l * b - square_area = l * b - b^2/2 :=
by sorry

end NUMINAMATH_CALUDE_metal_waste_l3626_362621


namespace NUMINAMATH_CALUDE_vendor_watermelons_l3626_362630

def watermelons_sold (n : ℕ) (total : ℕ) : ℕ :=
  if n = 0 then total
  else watermelons_sold (n - 1) ((total + 1) / 2)

theorem vendor_watermelons :
  ∃ (initial : ℕ), initial > 0 ∧ watermelons_sold 7 initial = 0 ∧ initial = 127 :=
sorry

end NUMINAMATH_CALUDE_vendor_watermelons_l3626_362630


namespace NUMINAMATH_CALUDE_coefficient_x_squared_in_expansion_l3626_362644

theorem coefficient_x_squared_in_expansion (x : ℝ) : 
  (x - 1)^4 = x^4 - 4*x^3 + 6*x^2 - 4*x + 1 := by
  sorry

#check coefficient_x_squared_in_expansion

end NUMINAMATH_CALUDE_coefficient_x_squared_in_expansion_l3626_362644


namespace NUMINAMATH_CALUDE_subset_arithmetic_result_l3626_362696

theorem subset_arithmetic_result (M : Finset ℕ) 
  (h_card : M.card = 13)
  (h_bounds : ∀ m ∈ M, 100 ≤ m ∧ m ≤ 999) :
  ∃ S : Finset ℕ, S ⊆ M ∧ 
  ∃ a b c d e f : ℕ, 
    a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ d ∈ S ∧ e ∈ S ∧ f ∈ S ∧
    3 < (b : ℚ) / a + (d : ℚ) / c + (f : ℚ) / e ∧
    (b : ℚ) / a + (d : ℚ) / c + (f : ℚ) / e < 4 :=
sorry

end NUMINAMATH_CALUDE_subset_arithmetic_result_l3626_362696


namespace NUMINAMATH_CALUDE_four_composition_odd_l3626_362655

/-- A function f is odd if f(-x) = -f(x) for all x -/
def IsOdd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

theorem four_composition_odd (f : ℝ → ℝ) (h : IsOdd f) : IsOdd (fun x ↦ f (f (f (f x)))) := by
  sorry

end NUMINAMATH_CALUDE_four_composition_odd_l3626_362655


namespace NUMINAMATH_CALUDE_cubic_root_function_l3626_362605

/-- Given a function y = kx^(1/3) where y = 4√3 when x = 64, 
    prove that y = 2√3 when x = 8 -/
theorem cubic_root_function (k : ℝ) :
  (∀ x, x > 0 → k * x^(1/3) = 4 * Real.sqrt 3 → x = 64) →
  k * 8^(1/3) = 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_cubic_root_function_l3626_362605


namespace NUMINAMATH_CALUDE_rectangular_solid_diagonal_l3626_362679

theorem rectangular_solid_diagonal (a b c : ℝ) 
  (surface_area : 2 * (a * b + b * c + a * c) = 34)
  (edge_length : 4 * (a + b + c) = 40) :
  ∃ d : ℝ, d^2 = 66 ∧ d^2 = a^2 + b^2 + c^2 := by sorry

end NUMINAMATH_CALUDE_rectangular_solid_diagonal_l3626_362679


namespace NUMINAMATH_CALUDE_fifth_friend_payment_l3626_362652

/-- Represents the payment made by each friend -/
structure Payment where
  first : ℝ
  second : ℝ
  third : ℝ
  fourth : ℝ
  fifth : ℝ

/-- Conditions for the gift payment problem -/
def GiftPaymentConditions (p : Payment) : Prop :=
  p.first + p.second + p.third + p.fourth + p.fifth = 120 ∧
  p.first = (1/3) * (p.second + p.third + p.fourth + p.fifth) ∧
  p.second = (1/4) * (p.first + p.third + p.fourth + p.fifth) ∧
  p.third = (1/5) * (p.first + p.second + p.fourth + p.fifth)

/-- Theorem stating that under the given conditions, the fifth friend paid $40 -/
theorem fifth_friend_payment (p : Payment) : 
  GiftPaymentConditions p → p.fifth = 40 := by
  sorry

end NUMINAMATH_CALUDE_fifth_friend_payment_l3626_362652


namespace NUMINAMATH_CALUDE_complete_residue_system_l3626_362659

theorem complete_residue_system (m : ℕ) (x : Fin m → ℤ) 
  (h : ∀ i j : Fin m, i ≠ j → x i % m ≠ x j % m) :
  ∀ k : Fin m, ∃ i : Fin m, x i % m = k.val :=
sorry

end NUMINAMATH_CALUDE_complete_residue_system_l3626_362659


namespace NUMINAMATH_CALUDE_race_distance_l3626_362603

/-- The race problem -/
theorem race_distance (speed_A speed_B : ℝ) (head_start win_margin total_distance : ℝ) :
  speed_A > 0 ∧ speed_B > 0 →
  speed_A / speed_B = 3 / 4 →
  head_start = 200 →
  win_margin = 100 →
  total_distance / speed_A = (total_distance - head_start - win_margin) / speed_B →
  total_distance = 900 := by
  sorry

#check race_distance

end NUMINAMATH_CALUDE_race_distance_l3626_362603


namespace NUMINAMATH_CALUDE_solution_set_inequality_l3626_362699

theorem solution_set_inequality (x : ℝ) : 
  (x - 3) * (x - 1) > 0 ↔ x < 1 ∨ x > 3 := by
sorry

end NUMINAMATH_CALUDE_solution_set_inequality_l3626_362699


namespace NUMINAMATH_CALUDE_wage_percentage_difference_l3626_362642

/-- Proves that the percentage difference between chef's and dishwasher's hourly wage is 20% -/
theorem wage_percentage_difference
  (manager_wage : ℝ)
  (chef_wage_difference : ℝ)
  (h_manager_wage : manager_wage = 6.50)
  (h_chef_wage_difference : chef_wage_difference = 2.60)
  (h_dishwasher_wage : dishwasher_wage = manager_wage / 2)
  (h_chef_wage : chef_wage = manager_wage - chef_wage_difference) :
  (chef_wage - dishwasher_wage) / dishwasher_wage * 100 = 20 :=
by sorry


end NUMINAMATH_CALUDE_wage_percentage_difference_l3626_362642


namespace NUMINAMATH_CALUDE_right_triangle_area_l3626_362681

/-- The area of a right triangle with legs 18 and 80 is 720 -/
theorem right_triangle_area : 
  ∀ (a b c : ℝ), 
  a = 18 → b = 80 → c = 82 →
  a^2 + b^2 = c^2 →
  (1/2) * a * b = 720 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_area_l3626_362681


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l3626_362622

theorem quadratic_equation_roots (m : ℝ) : m > 0 →
  (∃ (x : ℝ), x^2 + x - m = 0) ∧
  (∃ (x : ℝ), x^2 + x - m = 0) → m > 0 ∨
  m ≤ 0 → ¬(∃ (x : ℝ), x^2 + x - m = 0) ∨
  ¬(∃ (x : ℝ), x^2 + x - m = 0) → m ≤ 0 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l3626_362622


namespace NUMINAMATH_CALUDE_monotonic_increasing_interval_l3626_362625

open Real

noncomputable def f (ω x : ℝ) : ℝ := Real.sqrt 3 * sin (ω * x) + cos (ω * x)

theorem monotonic_increasing_interval
  (ω : ℝ)
  (h_ω_pos : ω > 0)
  (α β : ℝ)
  (h_f_α : f ω α = 2)
  (h_f_β : f ω β = 0)
  (h_min_diff : |α - β| = π / 2) :
  ∃ k : ℤ, StrictMonoOn f (Set.Icc (2 * k * π - 2 * π / 3) (2 * k * π + π / 3)) :=
by sorry

end NUMINAMATH_CALUDE_monotonic_increasing_interval_l3626_362625


namespace NUMINAMATH_CALUDE_game_properties_l3626_362695

/-- Represents the "What? Where? When?" game --/
structure Game where
  num_envelopes : Nat
  points_to_win : Nat
  num_games : Nat

/-- Calculates the expected number of points for one team in multiple games --/
def expectedPoints (g : Game) : ℝ :=
  sorry

/-- Calculates the probability of a specific envelope being chosen --/
def envelopeProbability (g : Game) : ℝ :=
  sorry

/-- Theorem stating the expected points and envelope probability for the given game --/
theorem game_properties :
  let g : Game := { num_envelopes := 13, points_to_win := 6, num_games := 100 }
  (expectedPoints g = 465) ∧ (envelopeProbability g = 12 / 13) := by
  sorry

end NUMINAMATH_CALUDE_game_properties_l3626_362695


namespace NUMINAMATH_CALUDE_number_of_subsets_complement_union_l3626_362656

universe u

def U : Finset ℕ := {1, 3, 5, 7, 9}
def A : Finset ℕ := {1, 5, 9}
def B : Finset ℕ := {3, 5, 9}

theorem number_of_subsets_complement_union : Finset.card (Finset.powerset (U \ (A ∪ B))) = 2 := by
  sorry

end NUMINAMATH_CALUDE_number_of_subsets_complement_union_l3626_362656


namespace NUMINAMATH_CALUDE_angle_measure_from_vector_sum_l3626_362646

/-- Given a triangle ABC with vectors m and n defined in terms of angle A, 
    prove that if the magnitude of their sum is √3, then A = π/3. -/
theorem angle_measure_from_vector_sum (A B C : ℝ) (a b c : ℝ) (m n : ℝ × ℝ) :
  0 < A ∧ A < π →
  m.1 = Real.cos (3 * A / 2) ∧ m.2 = Real.sin (3 * A / 2) →
  n.1 = Real.cos (A / 2) ∧ n.2 = Real.sin (A / 2) →
  Real.sqrt ((m.1 + n.1)^2 + (m.2 + n.2)^2) = Real.sqrt 3 →
  A = π / 3 := by
  sorry

end NUMINAMATH_CALUDE_angle_measure_from_vector_sum_l3626_362646


namespace NUMINAMATH_CALUDE_average_apples_per_day_l3626_362604

def boxes : ℕ := 12
def apples_per_box : ℕ := 25
def days : ℕ := 4

def total_apples : ℕ := boxes * apples_per_box

theorem average_apples_per_day : total_apples / days = 75 := by
  sorry

end NUMINAMATH_CALUDE_average_apples_per_day_l3626_362604


namespace NUMINAMATH_CALUDE_fraction_product_is_three_fifths_l3626_362628

theorem fraction_product_is_three_fifths :
  (7 / 4 : ℚ) * (8 / 14 : ℚ) * (20 / 12 : ℚ) * (15 / 25 : ℚ) *
  (21 / 14 : ℚ) * (12 / 18 : ℚ) * (28 / 14 : ℚ) * (30 / 50 : ℚ) = 3 / 5 := by
  sorry

end NUMINAMATH_CALUDE_fraction_product_is_three_fifths_l3626_362628


namespace NUMINAMATH_CALUDE_price_change_theorem_l3626_362682

theorem price_change_theorem (initial_price : ℝ) (h_pos : initial_price > 0) :
  let price_after_increase := initial_price * (1 + 0.34)
  let price_after_first_discount := price_after_increase * (1 - 0.10)
  let final_price := price_after_first_discount * (1 - 0.15)
  let percentage_change := (final_price - initial_price) / initial_price * 100
  percentage_change = 2.51 := by sorry

end NUMINAMATH_CALUDE_price_change_theorem_l3626_362682


namespace NUMINAMATH_CALUDE_triangle_cosA_value_l3626_362635

theorem triangle_cosA_value (A B C : ℝ) (a b c : ℝ) :
  0 < a ∧ 0 < b ∧ 0 < c →  -- Positive side lengths
  b = Real.sqrt 2 * c →  -- Given condition
  Real.sin A + Real.sqrt 2 * Real.sin C = 2 * Real.sin B →  -- Given condition
  -- Triangle inequality (to ensure it's a valid triangle)
  a + b > c ∧ b + c > a ∧ c + a > b →
  -- Law of sines (to connect side lengths and angles)
  a / Real.sin A = b / Real.sin B ∧ b / Real.sin B = c / Real.sin C →
  Real.cos A = Real.sqrt 2 / 4 :=
by
  sorry


end NUMINAMATH_CALUDE_triangle_cosA_value_l3626_362635


namespace NUMINAMATH_CALUDE_ac_values_l3626_362643

theorem ac_values (a c : ℝ) (h : ∀ x, 2 * Real.sin (3 * x) = a * Real.cos (3 * x + c)) :
  ∃ k : ℤ, a * c = (4 * k - 1) * Real.pi :=
sorry

end NUMINAMATH_CALUDE_ac_values_l3626_362643


namespace NUMINAMATH_CALUDE_simplify_expression_l3626_362658

theorem simplify_expression (y : ℝ) : 3 * y + 4 * y + 5 * y + 7 = 12 * y + 7 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3626_362658


namespace NUMINAMATH_CALUDE_min_value_theorem_l3626_362669

theorem min_value_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 2 * x + y = 3) :
  (2 / x + 1 / y) ≥ 3 ∧ ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ 2 * x₀ + y₀ = 3 ∧ 2 / x₀ + 1 / y₀ = 3 :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l3626_362669


namespace NUMINAMATH_CALUDE_max_value_product_sum_l3626_362608

theorem max_value_product_sum (X Y Z : ℕ) (sum_constraint : X + Y + Z = 15) :
  (∀ a b c : ℕ, a + b + c = 15 → X * Y * Z + X * Y + Y * Z + Z * X ≥ a * b * c + a * b + b * c + c * a) →
  X * Y * Z + X * Y + Y * Z + Z * X = 200 :=
by sorry

end NUMINAMATH_CALUDE_max_value_product_sum_l3626_362608


namespace NUMINAMATH_CALUDE_subtraction_proof_l3626_362688

theorem subtraction_proof :
  900000009000 - 123456789123 = 776543220777 := by
  sorry

end NUMINAMATH_CALUDE_subtraction_proof_l3626_362688


namespace NUMINAMATH_CALUDE_sin_beta_value_l3626_362668

theorem sin_beta_value (α β : Real)
  (h1 : 0 < α ∧ α < Real.pi / 2)
  (h2 : -Real.pi / 2 < β ∧ β < 0)
  (h3 : Real.cos (α - β) = -5/13)
  (h4 : Real.sin α = 4/5) :
  Real.sin β = -56/65 := by
  sorry

end NUMINAMATH_CALUDE_sin_beta_value_l3626_362668


namespace NUMINAMATH_CALUDE_a_8_value_l3626_362693

def sequence_property (a : ℕ+ → ℝ) : Prop :=
  ∀ m n : ℕ+, a (m * n) = a m * a n

theorem a_8_value (a : ℕ+ → ℝ) (h_prop : sequence_property a) (h_a2 : a 2 = 3) :
  a 8 = 27 := by
  sorry

end NUMINAMATH_CALUDE_a_8_value_l3626_362693


namespace NUMINAMATH_CALUDE_product_greater_than_sum_l3626_362629

theorem product_greater_than_sum (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a - b = a / b) : a * b > a + b := by
  sorry

end NUMINAMATH_CALUDE_product_greater_than_sum_l3626_362629


namespace NUMINAMATH_CALUDE_pyramid_hemisphere_theorem_l3626_362638

/-- A triangular pyramid with an equilateral triangular base -/
structure TriangularPyramid where
  /-- The height of the pyramid -/
  height : ℝ
  /-- The side length of the equilateral triangular base -/
  base_side : ℝ

/-- A hemisphere placed inside the pyramid -/
structure Hemisphere where
  /-- The radius of the hemisphere -/
  radius : ℝ

/-- Predicate to check if the hemisphere is properly placed in the pyramid -/
def is_properly_placed (p : TriangularPyramid) (h : Hemisphere) : Prop :=
  h.radius = 3 ∧ 
  p.height = 9 ∧
  -- The hemisphere is tangent to all three faces and rests on the base
  -- (This condition is assumed to be true when the predicate is true)
  True

/-- The main theorem -/
theorem pyramid_hemisphere_theorem (p : TriangularPyramid) (h : Hemisphere) :
  is_properly_placed p h → p.base_side = 6 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_pyramid_hemisphere_theorem_l3626_362638


namespace NUMINAMATH_CALUDE_smallest_value_between_one_and_two_l3626_362672

theorem smallest_value_between_one_and_two (x : ℝ) (h : 1 < x ∧ x < 2) :
  (1 / x^2) < min (x^3) (min (x^2) (min (2*x) (x^(1/2)))) :=
sorry

end NUMINAMATH_CALUDE_smallest_value_between_one_and_two_l3626_362672


namespace NUMINAMATH_CALUDE_factor_implies_b_value_l3626_362651

/-- The polynomial Q(x) = x^3 + 3x^2 + bx + 20 -/
def Q (b : ℝ) (x : ℝ) : ℝ := x^3 + 3*x^2 + b*x + 20

/-- Theorem: If x - 4 is a factor of Q(x), then b = -33 -/
theorem factor_implies_b_value (b : ℝ) :
  (∀ x, Q b x = 0 ↔ x = 4) → b = -33 := by
  sorry

end NUMINAMATH_CALUDE_factor_implies_b_value_l3626_362651


namespace NUMINAMATH_CALUDE_tan_alpha_value_l3626_362637

theorem tan_alpha_value (α : Real) (h : Real.tan (α + π/4) = 2) : Real.tan α = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_tan_alpha_value_l3626_362637


namespace NUMINAMATH_CALUDE_trig_expression_simplification_l3626_362674

theorem trig_expression_simplification (α : ℝ) : 
  (Real.tan (2 * π + α)) / (Real.tan (α + π) - Real.cos (-α) + Real.sin (π / 2 - α)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_trig_expression_simplification_l3626_362674


namespace NUMINAMATH_CALUDE_car_dealership_silver_percentage_l3626_362633

theorem car_dealership_silver_percentage
  (initial_cars : ℕ)
  (initial_silver_percentage : ℚ)
  (new_shipment : ℕ)
  (new_non_silver_percentage : ℚ)
  (h1 : initial_cars = 40)
  (h2 : initial_silver_percentage = 1/5)
  (h3 : new_shipment = 80)
  (h4 : new_non_silver_percentage = 7/20)
  : (initial_silver_percentage * initial_cars + (1 - new_non_silver_percentage) * new_shipment) / (initial_cars + new_shipment) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_car_dealership_silver_percentage_l3626_362633


namespace NUMINAMATH_CALUDE_solution_value_l3626_362666

theorem solution_value (x y m : ℝ) : x - 2*y = m → x = 2 → y = 1 → m = 0 := by
  sorry

end NUMINAMATH_CALUDE_solution_value_l3626_362666


namespace NUMINAMATH_CALUDE_draw_three_from_fifteen_l3626_362618

def box_numbers : List Nat := [1, 2, 3, 4, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610]

def total_combinations (n k : Nat) : Nat :=
  Nat.choose n k

theorem draw_three_from_fifteen :
  total_combinations (List.length box_numbers) 3 = 455 := by
  sorry

end NUMINAMATH_CALUDE_draw_three_from_fifteen_l3626_362618


namespace NUMINAMATH_CALUDE_purely_imaginary_condition_l3626_362648

-- Define the complex number i
def i : ℂ := Complex.I

-- Define the condition for a complex number to be purely imaginary
def is_purely_imaginary (z : ℂ) : Prop := z.re = 0 ∧ z.im ≠ 0

-- State the theorem
theorem purely_imaginary_condition (m : ℝ) :
  is_purely_imaginary ((m - i) * (1 + i)) → m = -1 := by
  sorry

end NUMINAMATH_CALUDE_purely_imaginary_condition_l3626_362648


namespace NUMINAMATH_CALUDE_train_bridge_problem_l3626_362611

/-- Represents the problem of determining the carriage position of a person walking through a train on a bridge. -/
theorem train_bridge_problem
  (bridge_length : ℝ)
  (train_speed : ℝ)
  (person_speed : ℝ)
  (carriage_length : ℝ)
  (h_bridge : bridge_length = 1400)
  (h_train : train_speed = 54 * (1000 / 3600))
  (h_person : person_speed = 3.6 * (1000 / 3600))
  (h_carriage : carriage_length = 23)
  : ∃ (n : ℕ), 5 ≤ n ∧ n ≤ 6 ∧
    (n : ℝ) * carriage_length ≥
      person_speed * (bridge_length / (train_speed + person_speed)) ∧
    ((n + 1) : ℝ) * carriage_length >
      person_speed * (bridge_length / (train_speed + person_speed)) :=
by sorry

end NUMINAMATH_CALUDE_train_bridge_problem_l3626_362611


namespace NUMINAMATH_CALUDE_decimal_to_fraction_l3626_362685

theorem decimal_to_fraction : (3.675 : ℚ) = 147 / 40 := by
  sorry

end NUMINAMATH_CALUDE_decimal_to_fraction_l3626_362685


namespace NUMINAMATH_CALUDE_a_gt_b_gt_c_l3626_362687

noncomputable def a : ℝ := Real.sqrt 2
noncomputable def b : ℝ := Real.log 3 / Real.log Real.pi
noncomputable def c : ℝ := Real.log 0.9 / Real.log 2

theorem a_gt_b_gt_c : a > b ∧ b > c := by sorry

end NUMINAMATH_CALUDE_a_gt_b_gt_c_l3626_362687


namespace NUMINAMATH_CALUDE_c_alone_time_l3626_362663

-- Define the rates of work for A, B, and C
variable (rA rB rC : ℝ)

-- Define the conditions
axiom ab_rate : rA + rB = 1/3
axiom bc_rate : rB + rC = 1/6
axiom ac_rate : rA + rC = 1/4

-- Define the theorem
theorem c_alone_time : 1 / rC = 24 := by
  sorry

end NUMINAMATH_CALUDE_c_alone_time_l3626_362663


namespace NUMINAMATH_CALUDE_investment_bankers_count_l3626_362697

/-- Proves that the number of investment bankers is 4 given the problem conditions -/
theorem investment_bankers_count : 
  ∀ (total_bill : ℝ) (avg_cost : ℝ) (num_clients : ℕ),
  total_bill = 756 →
  avg_cost = 70 →
  num_clients = 5 →
  ∃ (num_bankers : ℕ),
    num_bankers = 4 ∧
    total_bill = (avg_cost * (num_bankers + num_clients : ℝ)) * 1.2 :=
by sorry

end NUMINAMATH_CALUDE_investment_bankers_count_l3626_362697


namespace NUMINAMATH_CALUDE_inscribed_angles_sum_l3626_362612

theorem inscribed_angles_sum (circle : Real) (x y : Real) : 
  (circle > 0) →  -- circle has positive circumference
  (x = (2 / 12) * circle) →  -- x subtends 2/12 of the circle
  (y = (4 / 12) * circle) →  -- y subtends 4/12 of the circle
  (∃ (central_x central_y : Real), 
    central_x = 2 * x ∧ 
    central_y = 2 * y ∧ 
    central_x + central_y = circle) →  -- inscribed angle theorem
  x + y = 90 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_angles_sum_l3626_362612


namespace NUMINAMATH_CALUDE_odd_function_max_to_min_l3626_362606

/-- A function f is odd if f(-x) = -f(x) for all x in its domain -/
def IsOdd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

/-- A function f has a maximum value on [a, b] if there exists a point c in [a, b] 
    such that f(c) ≥ f(x) for all x in [a, b] -/
def HasMaxOn (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∃ c ∈ Set.Icc a b, ∀ x ∈ Set.Icc a b, f x ≤ f c

/-- A function f has a minimum value on [a, b] if there exists a point c in [a, b] 
    such that f(c) ≤ f(x) for all x in [a, b] -/
def HasMinOn (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∃ c ∈ Set.Icc a b, ∀ x ∈ Set.Icc a b, f c ≤ f x

theorem odd_function_max_to_min (f : ℝ → ℝ) (a b : ℝ) (h1 : a < b) 
  (h2 : IsOdd f) (h3 : HasMaxOn f a b) : HasMinOn f (-b) (-a) := by
  sorry

end NUMINAMATH_CALUDE_odd_function_max_to_min_l3626_362606


namespace NUMINAMATH_CALUDE_tank_plastering_cost_l3626_362609

/-- Calculates the total cost of plastering a rectangular tank -/
def plasteringCost (length width depth rate : ℝ) : ℝ :=
  let wallArea := 2 * (length * depth + width * depth)
  let bottomArea := length * width
  let totalArea := wallArea + bottomArea
  totalArea * rate

theorem tank_plastering_cost :
  let length : ℝ := 25
  let width : ℝ := 12
  let depth : ℝ := 6
  let rate : ℝ := 0.55  -- 55 paise converted to rupees
  plasteringCost length width depth rate = 409.2 := by
  sorry

end NUMINAMATH_CALUDE_tank_plastering_cost_l3626_362609


namespace NUMINAMATH_CALUDE_man_speed_in_still_water_l3626_362671

/-- The speed of the man in still water -/
def man_speed : ℝ := 10

/-- The speed of the stream -/
noncomputable def stream_speed : ℝ := sorry

/-- The downstream distance -/
def downstream_distance : ℝ := 28

/-- The upstream distance -/
def upstream_distance : ℝ := 12

/-- The time taken for both upstream and downstream journeys -/
def journey_time : ℝ := 2

theorem man_speed_in_still_water :
  (man_speed + stream_speed) * journey_time = downstream_distance ∧
  (man_speed - stream_speed) * journey_time = upstream_distance →
  man_speed = 10 := by
sorry

end NUMINAMATH_CALUDE_man_speed_in_still_water_l3626_362671


namespace NUMINAMATH_CALUDE_poem_word_count_l3626_362613

/-- Given a poem with the specified structure, prove that the total number of words is 1600. -/
theorem poem_word_count (stanzas : ℕ) (lines_per_stanza : ℕ) (words_per_line : ℕ)
  (h1 : stanzas = 20)
  (h2 : lines_per_stanza = 10)
  (h3 : words_per_line = 8) :
  stanzas * lines_per_stanza * words_per_line = 1600 := by
  sorry


end NUMINAMATH_CALUDE_poem_word_count_l3626_362613


namespace NUMINAMATH_CALUDE_black_marble_probability_l3626_362602

theorem black_marble_probability (yellow blue green black : ℕ) 
  (h1 : yellow = 12)
  (h2 : blue = 10)
  (h3 : green = 5)
  (h4 : black = 1) :
  (black * 14000) / (yellow + blue + green + black) = 500 := by
  sorry

end NUMINAMATH_CALUDE_black_marble_probability_l3626_362602


namespace NUMINAMATH_CALUDE_min_value_sum_fractions_l3626_362683

theorem min_value_sum_fractions (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a / (b + c) + b / (c + a) + c / (a + b)) ≥ (3 / 2) ∧
  (a / (b + c) + b / (c + a) + c / (a + b) = 3 / 2 ↔ a = b ∧ b = c) := by
  sorry

end NUMINAMATH_CALUDE_min_value_sum_fractions_l3626_362683


namespace NUMINAMATH_CALUDE_triangle_theorem_l3626_362640

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The main theorem about the triangle -/
theorem triangle_theorem (t : Triangle) 
  (h1 : t.a * Real.cos t.C + Real.sqrt 3 * t.a * Real.sin t.C - t.b - t.c = 0)
  (h2 : t.a = 2)
  (h3 : 1/2 * t.b * t.c * Real.sin t.A = Real.sqrt 3) : 
  t.A = π/3 ∧ t.b = 2 ∧ t.c = 2 := by
  sorry


end NUMINAMATH_CALUDE_triangle_theorem_l3626_362640


namespace NUMINAMATH_CALUDE_milk_revenue_calculation_l3626_362600

/-- Represents the revenue calculation for Mrs. Lim's milk sales --/
theorem milk_revenue_calculation :
  let yesterday_morning : ℕ := 68
  let yesterday_evening : ℕ := 82
  let this_morning : ℕ := yesterday_morning - 18
  let total_milk : ℕ := yesterday_morning + yesterday_evening + this_morning
  let milk_left : ℕ := 24
  let milk_sold : ℕ := total_milk - milk_left
  let price_per_gallon : ℚ := 7/2
  milk_sold * price_per_gallon = 616 := by
  sorry

end NUMINAMATH_CALUDE_milk_revenue_calculation_l3626_362600


namespace NUMINAMATH_CALUDE_increasing_function_inequality_l3626_362653

theorem increasing_function_inequality (f : ℝ → ℝ) (a : ℝ) 
  (h_increasing : ∀ x y, x < y → f x < f y) 
  (h_inequality : f (a^2 - a) > f (2*a^2 - 4*a)) : 
  0 < a ∧ a < 3 := by
sorry

end NUMINAMATH_CALUDE_increasing_function_inequality_l3626_362653


namespace NUMINAMATH_CALUDE_sum_congruence_mod_nine_l3626_362641

theorem sum_congruence_mod_nine :
  (2 + 33 + 444 + 5555 + 66666 + 777777 + 8888888 + 99999999) % 9 = 6 := by
  sorry

end NUMINAMATH_CALUDE_sum_congruence_mod_nine_l3626_362641


namespace NUMINAMATH_CALUDE_weight_of_second_new_student_l3626_362654

theorem weight_of_second_new_student
  (initial_students : Nat)
  (initial_avg_weight : ℝ)
  (new_students : Nat)
  (new_avg_weight : ℝ)
  (weight_of_first_new_student : ℝ)
  (h1 : initial_students = 29)
  (h2 : initial_avg_weight = 28)
  (h3 : new_students = initial_students + 2)
  (h4 : new_avg_weight = 27.5)
  (h5 : weight_of_first_new_student = 25)
  : ∃ (weight_of_second_new_student : ℝ),
    weight_of_second_new_student = 20.5 ∧
    (initial_students : ℝ) * initial_avg_weight + weight_of_first_new_student + weight_of_second_new_student =
    (new_students : ℝ) * new_avg_weight :=
by sorry

end NUMINAMATH_CALUDE_weight_of_second_new_student_l3626_362654


namespace NUMINAMATH_CALUDE_ratio_of_divisors_sums_l3626_362691

def M : ℕ := 36 * 36 * 98 * 210

def sum_odd_divisors (n : ℕ) : ℕ := sorry
def sum_even_divisors (n : ℕ) : ℕ := sorry

theorem ratio_of_divisors_sums :
  (sum_odd_divisors M : ℚ) / (sum_even_divisors M : ℚ) = 1 / 62 := by sorry

end NUMINAMATH_CALUDE_ratio_of_divisors_sums_l3626_362691


namespace NUMINAMATH_CALUDE_seating_theorem_l3626_362689

/-- Number of seats in a row -/
def num_seats : ℕ := 7

/-- Number of people to be seated -/
def num_people : ℕ := 3

/-- Function to calculate the number of seating arrangements -/
def seating_arrangements (seats : ℕ) (people : ℕ) : ℕ :=
  sorry

/-- Theorem stating the number of seating arrangements -/
theorem seating_theorem :
  seating_arrangements num_seats num_people = 100 :=
sorry

end NUMINAMATH_CALUDE_seating_theorem_l3626_362689


namespace NUMINAMATH_CALUDE_product_units_digit_l3626_362650

def is_composite (n : ℕ) : Prop := ∃ a b : ℕ, a > 1 ∧ b > 1 ∧ n = a * b

def units_digit (n : ℕ) : ℕ := n % 10

theorem product_units_digit :
  is_composite 4 ∧ is_composite 6 ∧ is_composite 9 →
  units_digit (4 * 6 * 9) = 6 := by sorry

end NUMINAMATH_CALUDE_product_units_digit_l3626_362650


namespace NUMINAMATH_CALUDE_evaluate_expression_l3626_362617

theorem evaluate_expression (x y : ℝ) (hx : x = 4) (hy : y = 5) :
  2 * y * (y - 2 * x) = -30 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l3626_362617


namespace NUMINAMATH_CALUDE_direct_proportion_problem_l3626_362667

theorem direct_proportion_problem (α β : ℝ) (k : ℝ) (h1 : α = k * β) (h2 : 6 = k * 18) (h3 : α = 15) : β = 45 := by
  sorry

end NUMINAMATH_CALUDE_direct_proportion_problem_l3626_362667
