import Mathlib

namespace income_ratio_proof_l485_48541

/-- Given two persons P1 and P2 with the following conditions:
    1. The ratio of their expenditures is 3:2
    2. Each saves Rs. 1800 at the end of the year
    3. The income of P1 is Rs. 4500
    Prove that the ratio of their incomes is 5:4 -/
theorem income_ratio_proof (expenditure_ratio : ℚ) (savings : ℕ) (income_p1 : ℕ) :
  expenditure_ratio = 3/2 →
  savings = 1800 →
  income_p1 = 4500 →
  ∃ (income_p2 : ℕ), (income_p1 : ℚ) / income_p2 = 5/4 :=
by sorry

end income_ratio_proof_l485_48541


namespace decreasing_geometric_sequence_characterization_l485_48540

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

def is_decreasing_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) < a n

theorem decreasing_geometric_sequence_characterization
  (a : ℕ → ℝ) (h_geometric : is_geometric_sequence a) :
  (a 1 > a 2 ∧ a 2 > a 3) ↔ is_decreasing_sequence a :=
by sorry

end decreasing_geometric_sequence_characterization_l485_48540


namespace tiffany_lives_lost_l485_48562

theorem tiffany_lives_lost (initial_lives gained_lives final_lives : ℕ) 
  (h1 : initial_lives = 43)
  (h2 : gained_lives = 27)
  (h3 : final_lives = 56) :
  initial_lives - (final_lives - gained_lives) = 14 :=
by sorry

end tiffany_lives_lost_l485_48562


namespace a_minus_b_value_l485_48568

theorem a_minus_b_value (a b : ℝ) (h1 : |a| = 2) (h2 : |b| = 5) (h3 : |a - b| = a - b) :
  a - b = 7 ∨ a - b = 3 := by
sorry

end a_minus_b_value_l485_48568


namespace decreasing_linear_function_l485_48575

def linear_function (k b x : ℝ) : ℝ := k * x + b

theorem decreasing_linear_function (k b : ℝ) :
  (∀ x₁ x₂ : ℝ, x₁ < x₂ → linear_function k b x₁ > linear_function k b x₂) ↔ k < 0 :=
sorry

end decreasing_linear_function_l485_48575


namespace quadratic_equation_implication_l485_48509

theorem quadratic_equation_implication (x : ℝ) : 
  x^2 + 3*x + 5 = 7 → 3*x^2 + 9*x - 11 = -5 := by
  sorry

end quadratic_equation_implication_l485_48509


namespace lunch_cost_proof_l485_48594

theorem lunch_cost_proof (total : ℝ) (difference : ℝ) (friend_cost : ℝ) : 
  total = 22 → difference = 5 → 
  (∃ (your_cost : ℝ), your_cost + (your_cost + difference) = total) →
  friend_cost = 13.5 := by
sorry

end lunch_cost_proof_l485_48594


namespace one_zero_in_interval_l485_48591

def f (x : ℝ) := -x^2 + 8*x - 14

theorem one_zero_in_interval :
  ∃! x, x ∈ Set.Icc 2 5 ∧ f x = 0 :=
by sorry

end one_zero_in_interval_l485_48591


namespace man_downstream_speed_l485_48549

/-- Given a man's upstream speed and the stream speed, calculates his downstream speed -/
def downstream_speed (upstream_speed stream_speed : ℝ) : ℝ :=
  upstream_speed + 2 * stream_speed

/-- Proves that given the specified conditions, the man's downstream speed is 14 kmph -/
theorem man_downstream_speed :
  let upstream_speed : ℝ := 8
  let stream_speed : ℝ := 3
  downstream_speed upstream_speed stream_speed = 14 := by
  sorry

end man_downstream_speed_l485_48549


namespace max_b_value_l485_48508

/-- The maximum value of b given the conditions -/
theorem max_b_value (a : ℝ) (f g : ℝ → ℝ) (h₁ : a > 0)
  (h₂ : ∀ x, f x = 6 * a^2 * Real.log x)
  (h₃ : ∀ x, g x = x^2 - 4*a*x - b)
  (h₄ : ∃ x₀, x₀ > 0 ∧ (deriv f x₀ = deriv g x₀) ∧ (f x₀ = g x₀)) :
  (∃ b : ℝ, ∀ b' : ℝ, b' ≤ b) ∧ (∀ b : ℝ, (∃ b' : ℝ, ∀ b'' : ℝ, b'' ≤ b') → b ≤ 1 / (3 * Real.exp 2)) :=
sorry

end max_b_value_l485_48508


namespace principal_is_720_l485_48524

/-- Calculates the principal amount given simple interest, time, and rate -/
def calculate_principal (simple_interest : ℚ) (time : ℚ) (rate : ℚ) : ℚ :=
  simple_interest * 100 / (rate * time)

/-- Theorem stating that the principal amount is 720 given the problem conditions -/
theorem principal_is_720 :
  let simple_interest : ℚ := 180
  let time : ℚ := 4
  let rate : ℚ := 6.25
  calculate_principal simple_interest time rate = 720 := by
  sorry

end principal_is_720_l485_48524


namespace always_two_real_roots_one_nonnegative_root_iff_l485_48528

-- Define the quadratic equation
def quadratic (m : ℝ) (x : ℝ) : ℝ := x^2 + (4-m)*x + (3-m)

-- Theorem 1: The equation always has two real roots
theorem always_two_real_roots (m : ℝ) :
  ∃ (x1 x2 : ℝ), quadratic m x1 = 0 ∧ quadratic m x2 = 0 :=
sorry

-- Theorem 2: The equation has exactly one non-negative real root iff m ≥ 3
theorem one_nonnegative_root_iff (m : ℝ) :
  (∃! (x : ℝ), x ≥ 0 ∧ quadratic m x = 0) ↔ m ≥ 3 :=
sorry

end always_two_real_roots_one_nonnegative_root_iff_l485_48528


namespace polygon_sides_l485_48512

/-- A convex polygon with the sum of all angles except one equal to 2790° has 18 sides -/
theorem polygon_sides (n : ℕ) (angle_sum : ℝ) : 
  n > 2 →
  angle_sum = 2790 →
  (n - 2) * 180 > angle_sum →
  (n - 1) * 180 ≥ angle_sum →
  n = 18 := by
sorry

end polygon_sides_l485_48512


namespace nancy_football_tickets_l485_48582

/-- The total amount Nancy spends on football tickets for three months -/
def total_spent (this_month_games : ℕ) (this_month_price : ℕ) 
                (last_month_games : ℕ) (last_month_price : ℕ) 
                (next_month_games : ℕ) (next_month_price : ℕ) : ℕ :=
  this_month_games * this_month_price + 
  last_month_games * last_month_price + 
  next_month_games * next_month_price

theorem nancy_football_tickets : 
  total_spent 9 5 8 4 7 6 = 119 := by
  sorry

end nancy_football_tickets_l485_48582


namespace product_abcd_zero_l485_48577

theorem product_abcd_zero 
  (a b c d : ℝ) 
  (eq1 : 3*a + 2*b + 4*c + 6*d = 60)
  (eq2 : 4*(d+c) = b^2)
  (eq3 : 4*b + 2*c = a)
  (eq4 : c - 2 = d) :
  a * b * c * d = 0 := by
sorry

end product_abcd_zero_l485_48577


namespace arithmetic_sequence_property_a_7_value_l485_48578

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_property (a : ℕ → ℝ) (h : arithmetic_sequence a) :
  a 1 + a 7 = a 3 + a 5 := by sorry

theorem a_7_value (a : ℕ → ℝ) 
  (h1 : arithmetic_sequence a) 
  (h2 : a 1 = 2) 
  (h3 : a 3 + a 5 = 10) : 
  a 7 = 8 := by sorry

end arithmetic_sequence_property_a_7_value_l485_48578


namespace add_million_minutes_to_start_date_l485_48510

/-- Represents a date and time -/
structure DateTime where
  year : Nat
  month : Nat
  day : Nat
  hour : Nat
  minute : Nat

/-- Adds minutes to a DateTime -/
def addMinutes (dt : DateTime) (minutes : Nat) : DateTime :=
  sorry

/-- The start date and time -/
def startDateTime : DateTime :=
  { year := 2007, month := 4, day := 15, hour := 12, minute := 0 }

/-- The number of minutes to add -/
def minutesToAdd : Nat := 1000000

/-- The expected end date and time -/
def expectedEndDateTime : DateTime :=
  { year := 2009, month := 3, day := 10, hour := 10, minute := 40 }

theorem add_million_minutes_to_start_date :
  addMinutes startDateTime minutesToAdd = expectedEndDateTime :=
sorry

end add_million_minutes_to_start_date_l485_48510


namespace equal_pairs_l485_48571

theorem equal_pairs (x y z : ℝ) (h : xy + z = yz + x ∧ yz + x = zx + y) :
  x = y ∨ y = z ∨ z = x := by
  sorry

end equal_pairs_l485_48571


namespace unripe_orange_harvest_l485_48592

/-- The number of sacks of unripe oranges harvested per day -/
def daily_unripe_harvest : ℕ := 65

/-- The number of days of harvest -/
def harvest_days : ℕ := 6

/-- The total number of sacks of unripe oranges harvested over the harvest period -/
def total_unripe_harvest : ℕ := daily_unripe_harvest * harvest_days

theorem unripe_orange_harvest : total_unripe_harvest = 390 := by
  sorry

end unripe_orange_harvest_l485_48592


namespace polynomial_value_l485_48583

/-- Given a polynomial function f(x) = ax^5 + bx^3 - cx + 2 where f(-3) = 9, 
    prove that f(3) = -5 -/
theorem polynomial_value (a b c : ℝ) :
  let f : ℝ → ℝ := λ x => a * x^5 + b * x^3 - c * x + 2
  f (-3) = 9 → f 3 = -5 := by sorry

end polynomial_value_l485_48583


namespace dodecagon_diagonals_l485_48553

/-- The number of diagonals in a polygon with n sides -/
def diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- A dodecagon has 12 sides -/
def dodecagon_sides : ℕ := 12

theorem dodecagon_diagonals : diagonals dodecagon_sides = 54 := by
  sorry

end dodecagon_diagonals_l485_48553


namespace gcd_lcm_sum_75_7350_l485_48599

theorem gcd_lcm_sum_75_7350 : Nat.gcd 75 7350 + Nat.lcm 75 7350 = 3225 := by
  sorry

end gcd_lcm_sum_75_7350_l485_48599


namespace length_of_PQ_l485_48502

/-- The problem setup -/
structure ProblemSetup where
  /-- Point R with coordinates (10, 15) -/
  R : ℝ × ℝ
  hR : R = (10, 15)
  
  /-- Line 1 with equation 7y = 24x -/
  line1 : ℝ → ℝ
  hline1 : ∀ x y, line1 y = 24 * x ∧ 7 * y = 24 * x
  
  /-- Line 2 with equation 15y = 4x -/
  line2 : ℝ → ℝ
  hline2 : ∀ x y, line2 y = 4/15 * x ∧ 15 * y = 4 * x
  
  /-- Point P on Line 1 -/
  P : ℝ × ℝ
  hP : line1 P.2 = 24 * P.1 ∧ 7 * P.2 = 24 * P.1
  
  /-- Point Q on Line 2 -/
  Q : ℝ × ℝ
  hQ : line2 Q.2 = 4/15 * Q.1 ∧ 15 * Q.2 = 4 * Q.1
  
  /-- R is the midpoint of PQ -/
  hMidpoint : R = ((P.1 + Q.1) / 2, (P.2 + Q.2) / 2)

/-- The main theorem -/
theorem length_of_PQ (setup : ProblemSetup) : 
  Real.sqrt ((setup.P.1 - setup.Q.1)^2 + (setup.P.2 - setup.Q.2)^2) = 3460 / 83 := by
  sorry

end length_of_PQ_l485_48502


namespace second_alignment_l485_48527

/-- Represents the number of Heavenly Stems -/
def heavenly_stems : ℕ := 10

/-- Represents the number of Earthly Branches -/
def earthly_branches : ℕ := 12

/-- Represents the cycle length of the combined Heavenly Stems and Earthly Branches -/
def cycle_length : ℕ := lcm heavenly_stems earthly_branches

/-- 
Theorem: The second occurrence of the first Heavenly Stem aligning with 
the first Earthly Branch happens at column 61.
-/
theorem second_alignment : 
  cycle_length + 1 = 61 := by sorry

end second_alignment_l485_48527


namespace perpendicular_vectors_l485_48539

def a : Fin 2 → ℝ := ![(-1), 2]
def b (m : ℝ) : Fin 2 → ℝ := ![m, 1]

theorem perpendicular_vectors (m : ℝ) : 
  (∀ i : Fin 2, (a + b m) i * a i = 0) → m = 7 := by
  sorry

end perpendicular_vectors_l485_48539


namespace system_solution_negative_implies_m_range_l485_48547

theorem system_solution_negative_implies_m_range (m : ℝ) : 
  (∃ x y : ℝ, x - y = 2*m + 7 ∧ x + y = 4*m - 3 ∧ x < 0 ∧ y < 0) → m < -2/3 := by
  sorry

end system_solution_negative_implies_m_range_l485_48547


namespace power_fraction_simplification_l485_48543

theorem power_fraction_simplification :
  (1 : ℝ) / ((-5^4)^2) * (-5)^9 = -5 := by
  sorry

end power_fraction_simplification_l485_48543


namespace annas_ebook_readers_l485_48573

theorem annas_ebook_readers (anna_readers john_initial_readers john_final_readers total_readers : ℕ) 
  (h1 : john_initial_readers = anna_readers - 15)
  (h2 : john_final_readers = john_initial_readers - 3)
  (h3 : anna_readers + john_final_readers = total_readers)
  (h4 : total_readers = 82) : anna_readers = 50 := by
  sorry

end annas_ebook_readers_l485_48573


namespace A_intersect_B_eq_zero_one_two_l485_48561

def A : Set ℕ := {x : ℕ | 5 + 4 * x - x^2 > 0}

def B : Set ℕ := {x : ℕ | x < 3}

theorem A_intersect_B_eq_zero_one_two : A ∩ B = {0, 1, 2} := by
  sorry

end A_intersect_B_eq_zero_one_two_l485_48561


namespace box_volume_perimeter_triples_l485_48535

def is_valid_triple (a b c : ℕ) : Prop :=
  1 ≤ a ∧ a ≤ b ∧ b ≤ c ∧ a * b * c = 4 * (a + b + c)

theorem box_volume_perimeter_triples :
  ∃! (n : ℕ), ∃ (S : Finset (ℕ × ℕ × ℕ)),
    S.card = n ∧
    (∀ (t : ℕ × ℕ × ℕ), t ∈ S ↔ is_valid_triple t.1 t.2.1 t.2.2) ∧
    n = 5 :=
sorry

end box_volume_perimeter_triples_l485_48535


namespace min_value_quadratic_sum_l485_48533

theorem min_value_quadratic_sum (a b c : ℝ) (h : 2*a + 2*b + c = 8) :
  (a - 1)^2 + (b + 2)^2 + (c - 3)^2 ≥ 49/9 := by
  sorry

end min_value_quadratic_sum_l485_48533


namespace inequality_range_l485_48584

theorem inequality_range (a : ℝ) : 
  (∀ x : ℝ, |x + 6| - |x - 4| ≤ a^2 - 3*a) ↔ 
  (a ≤ -2 ∨ a ≥ 5) :=
sorry

end inequality_range_l485_48584


namespace det_B_is_one_l485_48563

def B (a d : ℝ) : Matrix (Fin 2) (Fin 2) ℝ := !![a, 2; -3, d]

theorem det_B_is_one (a d : ℝ) (h : B a d + (B a d)⁻¹ = 0) : 
  Matrix.det (B a d) = 1 := by
  sorry

end det_B_is_one_l485_48563


namespace polynomial_decomposition_l485_48554

/-- The set of valid n values for which the polynomial decomposition is possible -/
def valid_n : Set ℕ :=
  {0, 1, 3, 7, 15, 12, 18, 25, 37, 51, 75, 151, 246, 493, 987, 1975}

/-- Predicate to check if a list of coefficients is valid for a given n -/
def valid_coefficients (n : ℕ) (coeffs : List ℕ) : Prop :=
  coeffs.length = n ∧
  coeffs.Nodup ∧
  ∀ a ∈ coeffs, 0 < a ∧ a ≤ n

/-- The main theorem stating the condition for valid polynomial decomposition -/
theorem polynomial_decomposition (n : ℕ) :
  (∃ coeffs : List ℕ, valid_coefficients n coeffs) ↔ n ∈ valid_n := by
  sorry

#check polynomial_decomposition

end polynomial_decomposition_l485_48554


namespace rice_container_problem_l485_48507

theorem rice_container_problem (total_weight : ℚ) (container_weight : ℕ) 
  (h1 : total_weight = 33 / 4)
  (h2 : container_weight = 33)
  (h3 : (1 : ℚ) = 16 / 16) : 
  (total_weight * 16) / container_weight = 4 := by
  sorry

end rice_container_problem_l485_48507


namespace polynomial_division_remainder_l485_48565

theorem polynomial_division_remainder :
  ∃ q : Polynomial ℝ, x^4 + 2*x^2 - 3 = (x^2 + 3*x + 2) * q + (-21*x - 21) :=
by sorry

end polynomial_division_remainder_l485_48565


namespace integral_x_plus_sqrt_4_minus_x_squared_l485_48564

open Set
open MeasureTheory
open Interval

/-- The definite integral of x + √(4 - x^2) from -2 to 2 equals 2π -/
theorem integral_x_plus_sqrt_4_minus_x_squared : 
  ∫ x in (-2)..2, (x + Real.sqrt (4 - x^2)) = 2 * Real.pi := by
  sorry

end integral_x_plus_sqrt_4_minus_x_squared_l485_48564


namespace range_when_p_and_q_range_when_p_or_q_and_not_p_and_q_l485_48551

-- Define propositions p and q
def p (m : ℝ) : Prop := 2^m > Real.sqrt 2

def q (m : ℝ) : Prop := ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁^2 - 2*x₁ + m^2 = 0 ∧ x₂^2 - 2*x₂ + m^2 = 0

-- Theorem for the first part
theorem range_when_p_and_q (m : ℝ) :
  p m ∧ q m → m > 1/2 ∧ m < 1 :=
by sorry

-- Theorem for the second part
theorem range_when_p_or_q_and_not_p_and_q (m : ℝ) :
  (p m ∨ q m) ∧ ¬(p m ∧ q m) → (m > -1 ∧ m ≤ 1/2) ∨ m ≥ 1 :=
by sorry

end range_when_p_and_q_range_when_p_or_q_and_not_p_and_q_l485_48551


namespace animals_per_aquarium_l485_48579

/-- Given that Tyler has 8 aquariums and 512 saltwater animals in total,
    prove that there are 64 animals in each aquarium. -/
theorem animals_per_aquarium (num_aquariums : ℕ) (total_animals : ℕ) 
  (h1 : num_aquariums = 8) (h2 : total_animals = 512) :
  total_animals / num_aquariums = 64 := by
  sorry

end animals_per_aquarium_l485_48579


namespace student_ticket_price_l485_48531

/-- The price of a senior citizen ticket -/
def senior_price : ℝ := sorry

/-- The price of a student ticket -/
def student_price : ℝ := sorry

/-- First day sales equation -/
axiom first_day_sales : 4 * senior_price + 3 * student_price = 79

/-- Second day sales equation -/
axiom second_day_sales : 12 * senior_price + 10 * student_price = 246

/-- Theorem stating that the student ticket price is 9 dollars -/
theorem student_ticket_price : student_price = 9 := by sorry

end student_ticket_price_l485_48531


namespace special_product_equality_l485_48559

theorem special_product_equality (x y : ℝ) : 
  (2 * x^3 - 5 * y^2) * (4 * x^6 + 10 * x^3 * y^2 + 25 * y^4) = 8 * x^9 - 125 * y^6 := by
  sorry

end special_product_equality_l485_48559


namespace contrapositive_equivalence_l485_48504

theorem contrapositive_equivalence (a b : ℝ) :
  (¬(a^2 < b → -Real.sqrt b < a ∧ a < Real.sqrt b)) ↔
  ((a ≥ Real.sqrt b ∨ a ≤ -Real.sqrt b) → a^2 ≥ b) :=
by sorry

end contrapositive_equivalence_l485_48504


namespace fish_tanks_theorem_l485_48556

/-- The total number of fish in three tanks, where one tank has a given number of fish
    and the other two have twice as many fish each as the first. -/
def total_fish (first_tank_fish : ℕ) : ℕ :=
  first_tank_fish + 2 * (2 * first_tank_fish)

/-- Theorem stating that with 3 fish tanks, where one tank has 20 fish and the other two
    have twice as many fish each as the first, the total number of fish is 100. -/
theorem fish_tanks_theorem : total_fish 20 = 100 := by
  sorry

end fish_tanks_theorem_l485_48556


namespace f_of_g_10_l485_48589

def g (x : ℝ) : ℝ := 4 * x + 6

def f (x : ℝ) : ℝ := 6 * x - 10

theorem f_of_g_10 : f (g 10) = 266 := by
  sorry

end f_of_g_10_l485_48589


namespace davids_pushups_l485_48526

/-- Given that Zachary did 19 push-ups and David did 39 more push-ups than Zachary,
    prove that David did 58 push-ups. -/
theorem davids_pushups (zachary_pushups : ℕ) (david_extra_pushups : ℕ) 
    (h1 : zachary_pushups = 19)
    (h2 : david_extra_pushups = 39) : 
    zachary_pushups + david_extra_pushups = 58 := by
  sorry

end davids_pushups_l485_48526


namespace consecutive_integers_sum_of_squares_l485_48585

theorem consecutive_integers_sum_of_squares : 
  ∀ x : ℕ, 
    x > 0 → 
    x * (x + 1) * (x + 2) = 12 * (x + (x + 1) + (x + 2)) → 
    x^2 + (x + 1)^2 + (x + 2)^2 = 77 := by
  sorry

end consecutive_integers_sum_of_squares_l485_48585


namespace price_change_l485_48597

theorem price_change (r s : ℝ) (h : r ≠ -100) (h2 : s ≠ 100) : 
  let initial_price := (10000 : ℝ) / (10000 + 100 * (r - s) - r * s)
  let price_after_increase := initial_price * (1 + r / 100)
  let final_price := price_after_increase * (1 - s / 100)
  final_price = 1 :=
by sorry

end price_change_l485_48597


namespace angle_problem_l485_48560

theorem angle_problem (x : ℝ) : 
  x + (3 * x - 10) = 180 → x = 47.5 := by
  sorry

end angle_problem_l485_48560


namespace prize_distribution_l485_48513

theorem prize_distribution (total_winners : ℕ) (min_award : ℚ) (max_award : ℚ) :
  total_winners = 20 →
  min_award = 20 →
  max_award = 160 →
  ∃ (total_prize : ℚ),
    total_prize > 0 ∧
    (2 / 5 : ℚ) * total_prize = max_award ∧
    (∀ (winner : ℕ), winner ≤ total_winners → ∃ (award : ℚ), min_award ≤ award ∧ award ≤ max_award) ∧
    total_prize = 1000 :=
by sorry

end prize_distribution_l485_48513


namespace systematic_sampling_problem_l485_48511

/-- Represents the systematic sampling problem --/
theorem systematic_sampling_problem 
  (total_students : ℕ) 
  (sample_size : ℕ) 
  (num_groups : ℕ) 
  (group_size : ℕ) 
  (sixteenth_group_num : ℕ) :
  total_students = 160 →
  sample_size = 20 →
  num_groups = 20 →
  group_size = total_students / num_groups →
  sixteenth_group_num = 126 →
  ∃ (first_group_num : ℕ), first_group_num = 6 :=
by
  sorry


end systematic_sampling_problem_l485_48511


namespace circle_equation_l485_48572

/-- Theorem: Equation of a Circle
    For any point (x, y) on a circle with radius R and center (a, b),
    the equation (x-a)^2 + (y-b)^2 = R^2 holds. -/
theorem circle_equation (R a b x y : ℝ) (h : (x - a)^2 + (y - b)^2 = R^2) :
  (x - a)^2 + (y - b)^2 = R^2 := by
  sorry

#check circle_equation

end circle_equation_l485_48572


namespace problem_solution_l485_48552

theorem problem_solution (t : ℝ) (x y : ℝ) 
  (h1 : x = 3 - 2*t) 
  (h2 : y = 3*t + 5) 
  (h3 : x = 9) : 
  y = -4 := by
  sorry

end problem_solution_l485_48552


namespace esha_lag_behind_anusha_l485_48545

/-- Represents the runners in the race -/
inductive Runner
| Anusha
| Banu
| Esha

/-- The race parameters and conditions -/
structure RaceConditions where
  race_length : ℝ
  speeds : Runner → ℝ
  anusha_fastest : speeds Runner.Anusha > speeds Runner.Banu ∧ speeds Runner.Banu > speeds Runner.Esha
  banu_lag : speeds Runner.Banu / speeds Runner.Anusha = 9 / 10
  esha_lag : speeds Runner.Esha / speeds Runner.Banu = 9 / 10

/-- The theorem to be proved -/
theorem esha_lag_behind_anusha (rc : RaceConditions) (h : rc.race_length = 100) :
  rc.race_length - (rc.speeds Runner.Esha / rc.speeds Runner.Anusha) * rc.race_length = 19 := by
  sorry

end esha_lag_behind_anusha_l485_48545


namespace problem_solution_l485_48536

theorem problem_solution (a b : ℝ) (h1 : a + b = 8) (h2 : a^2 * b^2 = 4) :
  (a^2 + b^2)/2 - a*b = 28 ∨ (a^2 + b^2)/2 - a*b = 36 := by
sorry

end problem_solution_l485_48536


namespace sarah_speeding_tickets_l485_48542

theorem sarah_speeding_tickets (total_tickets : ℕ) (mark_parking : ℕ) :
  total_tickets = 24 →
  mark_parking = 8 →
  ∃ (sarah_speeding : ℕ),
    sarah_speeding = 6 ∧
    sarah_speeding + sarah_speeding + mark_parking + mark_parking / 2 = total_tickets :=
by sorry

end sarah_speeding_tickets_l485_48542


namespace arithmetic_mean_difference_l485_48544

theorem arithmetic_mean_difference (p q r : ℝ) : 
  (p + q) / 2 = 10 → (q + r) / 2 = 24 → r - p = 28 := by
  sorry

end arithmetic_mean_difference_l485_48544


namespace train_speed_l485_48521

/-- Calculate the speed of a train given its length and time to cross a point -/
theorem train_speed (length : ℝ) (time : ℝ) (h1 : length = 500) (h2 : time = 20) :
  length / time = 25 := by
  sorry

end train_speed_l485_48521


namespace min_balls_for_three_same_color_60_6_l485_48581

/-- Given a bag with colored balls, returns the minimum number of balls
    that must be picked to ensure at least three balls of the same color are picked. -/
def min_balls_for_three_same_color (total_balls : ℕ) (balls_per_color : ℕ) : ℕ :=
  2 * (total_balls / balls_per_color) + 1

/-- Proves that for a bag with 60 balls and 6 balls of each color,
    the minimum number of balls to pick to ensure at least three of the same color is 21. -/
theorem min_balls_for_three_same_color_60_6 :
  min_balls_for_three_same_color 60 6 = 21 := by
  sorry

#eval min_balls_for_three_same_color 60 6

end min_balls_for_three_same_color_60_6_l485_48581


namespace sqrt_equation_solution_l485_48570

theorem sqrt_equation_solution (x : ℝ) : 
  x ≥ 2 → 
  (Real.sqrt (x + 5 - 6 * Real.sqrt (x - 2)) + Real.sqrt (x + 12 - 8 * Real.sqrt (x - 2)) = 2) ↔
  (11 ≤ x ∧ x ≤ 27) := by
sorry

end sqrt_equation_solution_l485_48570


namespace school_club_members_l485_48587

theorem school_club_members :
  ∃! n : ℕ,
    200 ≤ n ∧ n ≤ 300 ∧
    n % 6 = 3 ∧
    n % 8 = 5 ∧
    n % 9 = 7 ∧
    n = 269 := by sorry

end school_club_members_l485_48587


namespace triangle_inequality_l485_48576

theorem triangle_inequality (A B C : Real) (h_acute : 0 < A ∧ A < π/2 ∧ 0 < B ∧ B < π/2 ∧ 0 < C ∧ C < π/2) 
  (h_sum : A + B + C = π) : 
  (Real.sin (2*A) + Real.sin (2*B))^2 / (Real.sin A * Real.sin B) + 
  (Real.sin (2*B) + Real.sin (2*C))^2 / (Real.sin B * Real.sin C) + 
  (Real.sin (2*C) + Real.sin (2*A))^2 / (Real.sin C * Real.sin A) ≤ 12 := by
sorry

end triangle_inequality_l485_48576


namespace triangle_inequalities_l485_48569

theorem triangle_inequalities (a b c : ℝ) 
  (h1 : 0 ≤ a ∧ a ≤ 1) 
  (h2 : 0 ≤ b ∧ b ≤ 1) 
  (h3 : 0 ≤ c ∧ c ≤ 1) 
  (h4 : a + b + c = 2) : 
  (a * b * c + 28 / 27 ≥ a * b + b * c + c * a) ∧ 
  (a * b + b * c + c * a ≥ a * b * c + 1) := by
  sorry

end triangle_inequalities_l485_48569


namespace diamond_equation_solution_l485_48517

/-- Definition of the diamond operation -/
def diamond (a b : ℝ) : ℝ := 3 * a - b^2

/-- Theorem stating that if a ◇ 6 = 15, then a = 17 -/
theorem diamond_equation_solution (a : ℝ) : diamond a 6 = 15 → a = 17 := by
  sorry

end diamond_equation_solution_l485_48517


namespace min_value_theorem_l485_48500

theorem min_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 4) :
  ∃ (m : ℝ), m = 3 ∧ ∀ x y, x > 0 → y > 0 → x + y = 4 → (y / x + 4 / y) ≥ m :=
by sorry

end min_value_theorem_l485_48500


namespace trajectory_of_symmetric_point_l485_48514

/-- The equation of the trajectory of point N, which is symmetric to a point M on the circle x^2 + y^2 = 4 with respect to the point A(1,1) -/
theorem trajectory_of_symmetric_point (x y : ℝ) :
  (∃ (mx my : ℝ), mx^2 + my^2 = 4 ∧ x = 2 - mx ∧ y = 2 - my) →
  (x - 2)^2 + (y - 2)^2 = 4 := by
sorry

end trajectory_of_symmetric_point_l485_48514


namespace square_field_area_l485_48501

/-- The area of a square field with a diagonal of 20 meters is 200 square meters. -/
theorem square_field_area (diagonal : Real) (area : Real) :
  diagonal = 20 →
  area = diagonal^2 / 2 →
  area = 200 := by sorry

end square_field_area_l485_48501


namespace income_calculation_l485_48580

/-- Proves that given a person's income and expenditure are in the ratio 4:3, 
    and their savings are Rs. 5,000, their income is Rs. 20,000. -/
theorem income_calculation (income expenditure savings : ℕ) : 
  income * 3 = expenditure * 4 →  -- Income and expenditure ratio is 4:3
  income - expenditure = savings → -- Savings definition
  savings = 5000 →                -- Given savings amount
  income = 20000 :=               -- Conclusion to prove
by
  sorry

end income_calculation_l485_48580


namespace final_savings_calculation_correct_l485_48557

/-- Calculates the final savings given initial savings, monthly income, monthly expenses, and number of months. -/
def calculate_final_savings (initial_savings monthly_income monthly_expenses : ℕ) (num_months : ℕ) : ℕ :=
  initial_savings + num_months * monthly_income - num_months * monthly_expenses

/-- Theorem stating that the final savings calculation is correct for the given problem. -/
theorem final_savings_calculation_correct :
  let initial_savings : ℕ := 849400
  let monthly_income : ℕ := 45000 + 35000 + 7000 + 10000 + 13000
  let monthly_expenses : ℕ := 30000 + 10000 + 5000 + 4500 + 9000
  let num_months : ℕ := 5
  calculate_final_savings initial_savings monthly_income monthly_expenses num_months = 1106900 := by
  sorry

#eval calculate_final_savings 849400 110000 58500 5

end final_savings_calculation_correct_l485_48557


namespace vacation_duration_l485_48519

/-- The number of emails received on the first day -/
def first_day_emails : ℕ := 16

/-- The ratio of emails received on each subsequent day compared to the previous day -/
def email_ratio : ℚ := 1/2

/-- The total number of emails received during the vacation -/
def total_emails : ℕ := 30

/-- Calculate the sum of a geometric series -/
def geometric_sum (a : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

/-- The number of days in the vacation -/
def vacation_days : ℕ := 4

theorem vacation_duration :
  geometric_sum first_day_emails email_ratio vacation_days = total_emails := by
  sorry

end vacation_duration_l485_48519


namespace squared_one_necessary_not_sufficient_l485_48538

theorem squared_one_necessary_not_sufficient (x : ℝ) :
  (x = 1 → x^2 = 1) ∧ ¬(x^2 = 1 → x = 1) := by sorry

end squared_one_necessary_not_sufficient_l485_48538


namespace expression_evaluation_l485_48596

theorem expression_evaluation (a : ℝ) (h : a = -6) :
  (1 - a / (a - 3)) / ((a^2 + 3*a) / (a^2 - 9)) = 1/2 := by
  sorry

end expression_evaluation_l485_48596


namespace derivative_f_at_zero_l485_48546

def f (x : ℝ) : ℝ := x^2 + 2*x

theorem derivative_f_at_zero : 
  deriv f 0 = 2 := by sorry

end derivative_f_at_zero_l485_48546


namespace min_product_of_three_l485_48534

def S : Finset Int := {-9, -7, -5, 0, 4, 6, 8}

theorem min_product_of_three (a b c : Int) (ha : a ∈ S) (hb : b ∈ S) (hc : c ∈ S) 
  (hab : a ≠ b) (hbc : b ≠ c) (hac : a ≠ c) :
  (∀ x y z : Int, x ∈ S → y ∈ S → z ∈ S → x ≠ y → y ≠ z → x ≠ z → 
    a * b * c ≤ x * y * z) → 
  a * b * c = -336 :=
sorry

end min_product_of_three_l485_48534


namespace mirror_area_l485_48548

/-- The area of a rectangular mirror inside a frame -/
theorem mirror_area (frame_width : ℕ) (frame_height : ℕ) (frame_thickness : ℕ) : 
  frame_width = 90 ∧ frame_height = 70 ∧ frame_thickness = 15 →
  (frame_width - 2 * frame_thickness) * (frame_height - 2 * frame_thickness) = 2400 := by
sorry

end mirror_area_l485_48548


namespace division_result_l485_48515

theorem division_result : (3486 : ℚ) / 189 = 18.444444444444443 := by
  sorry

end division_result_l485_48515


namespace equation_solution_l485_48503

theorem equation_solution : ∃ x : ℚ, x + 2/5 = 7/10 + 1/2 ∧ x = 4/5 := by
  sorry

end equation_solution_l485_48503


namespace angle_in_second_quadrant_l485_48588

theorem angle_in_second_quadrant (α : Real) (x : Real) :
  -- α is in the second quadrant
  π / 2 < α ∧ α < π →
  -- P(x,6) is on the terminal side of α
  x < 0 →
  -- sin α = 3/5
  Real.sin α = 3 / 5 →
  -- x = -8
  x = -8 :=
by
  sorry

end angle_in_second_quadrant_l485_48588


namespace james_annual_training_hours_l485_48595

/-- Calculates the total training hours per year for an athlete with a specific schedule. -/
def training_hours_per_year (sessions_per_day : ℕ) (hours_per_session : ℕ) (training_days_per_week : ℕ) (weeks_per_year : ℕ) : ℕ :=
  sessions_per_day * hours_per_session * training_days_per_week * weeks_per_year

/-- Proves that James' training schedule results in 2080 hours per year. -/
theorem james_annual_training_hours :
  training_hours_per_year 2 4 5 52 = 2080 := by
  sorry

#eval training_hours_per_year 2 4 5 52

end james_annual_training_hours_l485_48595


namespace velocity_at_5_seconds_l485_48555

-- Define the position function
def s (t : ℝ) : ℝ := 3 * t^2 - 2 * t + 4

-- Define the velocity function as the derivative of the position function
def v (t : ℝ) : ℝ := 6 * t - 2

-- Theorem statement
theorem velocity_at_5_seconds :
  v 5 = 28 := by
  sorry

end velocity_at_5_seconds_l485_48555


namespace interest_rate_difference_l485_48529

theorem interest_rate_difference 
  (principal : ℝ) 
  (time : ℝ) 
  (interest_diff : ℝ) 
  (h1 : principal = 2100) 
  (h2 : time = 3) 
  (h3 : interest_diff = 63) : 
  ∃ (rate1 rate2 : ℝ), rate2 - rate1 = 0.01 ∧ 
    principal * rate2 * time - principal * rate1 * time = interest_diff :=
by sorry

end interest_rate_difference_l485_48529


namespace min_value_a_plus_b_plus_c_l485_48598

theorem min_value_a_plus_b_plus_c (a b c : ℝ) 
  (h1 : a^2 + b^2 ≤ c) (h2 : c ≤ 1) : 
  ∀ x y z : ℝ, x^2 + y^2 ≤ z ∧ z ≤ 1 → a + b + c ≤ x + y + z ∧ 
  ∃ a₀ b₀ c₀ : ℝ, a₀^2 + b₀^2 ≤ c₀ ∧ c₀ ≤ 1 ∧ a₀ + b₀ + c₀ = -1/2 :=
by sorry

end min_value_a_plus_b_plus_c_l485_48598


namespace right_triangle_tangent_circles_area_sum_l485_48506

theorem right_triangle_tangent_circles_area_sum :
  ∀ (r s t : ℝ),
  r > 0 → s > 0 → t > 0 →
  r + s = 6 →
  r + t = 8 →
  s + t = 10 →
  (6 : ℝ)^2 + 8^2 = 10^2 →
  π * (r^2 + s^2 + t^2) = 36 * π := by
sorry

end right_triangle_tangent_circles_area_sum_l485_48506


namespace min_value_of_sequence_l485_48505

/-- Given a sequence {a_n} where a₂ = 102 and aₙ₊₁ - aₙ = 4n for n ∈ ℕ*, 
    the minimum value of {aₙ/n} is 26. -/
theorem min_value_of_sequence (a : ℕ → ℝ) : 
  (a 2 = 102) → 
  (∀ n : ℕ, n ≥ 1 → a (n + 1) - a n = 4 * n) → 
  (∃ n₀ : ℕ, n₀ ≥ 1 ∧ a n₀ / n₀ = 26) ∧ 
  (∀ n : ℕ, n ≥ 1 → a n / n ≥ 26) :=
by sorry

end min_value_of_sequence_l485_48505


namespace sqrt_seven_to_sixth_l485_48518

theorem sqrt_seven_to_sixth : (Real.sqrt 7) ^ 6 = 343 := by
  sorry

end sqrt_seven_to_sixth_l485_48518


namespace polygon_triangle_division_l485_48566

/-- 
A polygon where the sum of interior angles is twice the sum of exterior angles
can be divided into at most 4 triangles by connecting one vertex to all others.
-/
theorem polygon_triangle_division :
  ∀ (n : ℕ), 
  (n - 2) * 180 = 2 * 360 →
  n - 2 = 4 :=
by sorry

end polygon_triangle_division_l485_48566


namespace visitor_growth_rate_l485_48532

theorem visitor_growth_rate (initial_visitors : ℝ) (final_visitors : ℝ) (x : ℝ) : 
  initial_visitors = 42 → 
  final_visitors = 133.91 → 
  initial_visitors * (1 + x)^2 = final_visitors :=
by sorry

end visitor_growth_rate_l485_48532


namespace work_completion_theorem_l485_48530

def work_completion_time (rate_A rate_B rate_C : ℚ) (initial_days : ℕ) : ℚ :=
  let combined_rate_AB := rate_A + rate_B
  let work_done_AB := combined_rate_AB * initial_days
  let remaining_work := 1 - work_done_AB
  let combined_rate_AC := rate_A + rate_C
  initial_days + remaining_work / combined_rate_AC

theorem work_completion_theorem :
  let rate_A : ℚ := 1 / 30
  let rate_B : ℚ := 1 / 15
  let rate_C : ℚ := 1 / 20
  let initial_days : ℕ := 5
  work_completion_time rate_A rate_B rate_C initial_days = 11 := by
  sorry

end work_completion_theorem_l485_48530


namespace min_value_of_expression_min_value_sqrt2_minus_half_achievable_min_value_l485_48567

theorem min_value_of_expression (a d b c : ℝ) 
  (ha : a ≥ 0) (hd : d ≥ 0) (hb : b > 0) (hc : c > 0) (h_sum : b + c ≥ a + d) :
  ∀ x y z w, x ≥ 0 ∧ y ≥ 0 ∧ z > 0 ∧ w > 0 ∧ z + w ≥ x + y →
  (b / (c + d)) + (c / (a + b)) ≤ (z / (w + y)) + (w / (x + z)) :=
by sorry

theorem min_value_sqrt2_minus_half (a d b c : ℝ) 
  (ha : a ≥ 0) (hd : d ≥ 0) (hb : b > 0) (hc : c > 0) (h_sum : b + c ≥ a + d) :
  (b / (c + d)) + (c / (a + b)) ≥ Real.sqrt 2 - 1/2 :=
by sorry

theorem achievable_min_value (a d b c : ℝ) :
  ∃ a d b c, a ≥ 0 ∧ d ≥ 0 ∧ b > 0 ∧ c > 0 ∧ b + c ≥ a + d ∧
  (b / (c + d)) + (c / (a + b)) = Real.sqrt 2 - 1/2 :=
by sorry

end min_value_of_expression_min_value_sqrt2_minus_half_achievable_min_value_l485_48567


namespace more_polygons_with_specific_point_l485_48593

theorem more_polygons_with_specific_point (n : ℕ) (h : n = 16) :
  let total_polygons := 2^n - (Nat.choose n 0 + Nat.choose n 1 + Nat.choose n 2)
  let polygons_with_point := 2^(n-1) - (Nat.choose (n-1) 0 + Nat.choose (n-1) 1)
  let polygons_without_point := total_polygons - polygons_with_point
  polygons_with_point > polygons_without_point := by
sorry

end more_polygons_with_specific_point_l485_48593


namespace digits_used_128_l485_48550

/-- The number of digits used to number pages from 1 to n -/
def digits_used (n : ℕ) : ℕ :=
  (min n 9) +
  (if n ≥ 10 then 2 * (min (n - 9) 90) else 0) +
  (if n ≥ 100 then 3 * (n - 99) else 0)

/-- The theorem stating that the number of digits used to number pages from 1 to 128 is 276 -/
theorem digits_used_128 : digits_used 128 = 276 := by
  sorry

end digits_used_128_l485_48550


namespace batsman_new_average_l485_48523

/-- Represents a batsman's score statistics -/
structure BatsmanStats where
  initialAverage : ℝ
  inningsPlayed : ℕ
  newInningScore : ℝ
  averageIncrease : ℝ

/-- Theorem: Given a batsman's stats, prove that his new average is 40 runs -/
theorem batsman_new_average (stats : BatsmanStats)
  (h1 : stats.inningsPlayed = 10)
  (h2 : stats.newInningScore = 90)
  (h3 : stats.averageIncrease = 5)
  : stats.initialAverage + stats.averageIncrease = 40 := by
  sorry

#check batsman_new_average

end batsman_new_average_l485_48523


namespace no_solution_condition_l485_48520

theorem no_solution_condition (a : ℝ) : 
  (∀ x : ℝ, 8 * |x - 4*a| + |x - a^2| + 7*x - 2*a ≠ 0) ↔ (a < -22 ∨ a > 0) :=
sorry

end no_solution_condition_l485_48520


namespace profit_margin_ratio_l485_48590

/-- Prove that for an article with selling price S, cost C, and profit margin M = (1/n)S, 
    the ratio of M to C is equal to 1/(n-1) -/
theorem profit_margin_ratio (n : ℝ) (S : ℝ) (C : ℝ) (M : ℝ) 
    (h1 : n ≠ 0) 
    (h2 : n ≠ 1)
    (h3 : M = (1/n) * S) 
    (h4 : C = S - M) : 
  M / C = 1 / (n - 1) := by
  sorry

end profit_margin_ratio_l485_48590


namespace parallel_line_y_intercept_l485_48537

/-- A line in the 2D plane represented by its slope and a point it passes through. -/
structure Line where
  slope : ℝ
  point : ℝ × ℝ

/-- The y-intercept of a line. -/
def yIntercept (l : Line) : ℝ :=
  l.point.2 - l.slope * l.point.1

theorem parallel_line_y_intercept :
  ∀ (b : Line),
    b.slope = 3 →                -- b is parallel to y = 3x - 6
    b.point = (3, 4) →           -- b passes through (3, 4)
    yIntercept b = -5            -- the y-intercept of b is -5
  := by sorry

end parallel_line_y_intercept_l485_48537


namespace staircase_steps_l485_48525

/-- Represents a staircase with a given number of steps. -/
structure Staircase :=
  (steps : ℕ)

/-- Calculates the total number of toothpicks used in a staircase. -/
def toothpicks (s : Staircase) : ℕ :=
  3 * (s.steps * (s.steps + 1)) / 2

/-- Theorem stating that a staircase with 270 toothpicks has 12 steps. -/
theorem staircase_steps : ∃ s : Staircase, toothpicks s = 270 ∧ s.steps = 12 := by
  sorry


end staircase_steps_l485_48525


namespace magnitude_of_4_minus_15i_l485_48586

-- Define the complex number
def z : ℂ := 4 - 15 * Complex.I

-- State the theorem
theorem magnitude_of_4_minus_15i : Complex.abs z = Real.sqrt 241 := by
  sorry

end magnitude_of_4_minus_15i_l485_48586


namespace airplane_speed_proof_l485_48522

/-- Proves that the speed of one airplane is 400 mph given the conditions of the problem -/
theorem airplane_speed_proof (v : ℝ) : 
  v > 0 →  -- Assuming positive speed for the first airplane
  (2.5 * v + 2.5 * 250 = 1625) →  -- Condition from the problem
  v = 400 := by
sorry

end airplane_speed_proof_l485_48522


namespace product_sum_8670_l485_48516

theorem product_sum_8670 : ∃ (a b : ℕ), 
  10 ≤ a ∧ a < 100 ∧ 
  10 ≤ b ∧ b < 100 ∧ 
  a * b = 8670 ∧ 
  a + b = 187 := by
  sorry

end product_sum_8670_l485_48516


namespace hat_markup_price_l485_48558

theorem hat_markup_price (P : ℝ) (h : 2 * P - 1.6 * P = 6) : 1.6 * P = 24 := by
  sorry

end hat_markup_price_l485_48558


namespace expression_evaluation_l485_48574

theorem expression_evaluation (x y : ℕ) (h1 : x = 3) (h2 : y = 4) :
  5 * x^y + 6 * y^x + x * y = 801 := by
  sorry

end expression_evaluation_l485_48574
