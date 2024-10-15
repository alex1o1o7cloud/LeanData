import Mathlib

namespace NUMINAMATH_CALUDE_zoo_animals_count_l913_91360

theorem zoo_animals_count (female_count : ℕ) (male_excess : ℕ) : 
  female_count = 35 → male_excess = 7 → female_count + (female_count + male_excess) = 77 := by
  sorry

end NUMINAMATH_CALUDE_zoo_animals_count_l913_91360


namespace NUMINAMATH_CALUDE_population_difference_after_increase_l913_91327

/-- Represents the population of birds in a wildlife reserve -/
structure BirdPopulation where
  eagles : ℕ
  falcons : ℕ
  hawks : ℕ
  owls : ℕ

/-- Calculates the difference between the most and least populous bird types -/
def populationDifference (pop : BirdPopulation) : ℕ :=
  max pop.eagles (max pop.falcons (max pop.hawks pop.owls)) -
  min pop.eagles (min pop.falcons (min pop.hawks pop.owls))

/-- Calculates the new population after increasing the least populous by 10% -/
def increaseLeastPopulous (pop : BirdPopulation) : BirdPopulation :=
  let minPop := min pop.eagles (min pop.falcons (min pop.hawks pop.owls))
  let increase := minPop * 10 / 100
  { eagles := if pop.eagles = minPop then pop.eagles + increase else pop.eagles,
    falcons := if pop.falcons = minPop then pop.falcons + increase else pop.falcons,
    hawks := if pop.hawks = minPop then pop.hawks + increase else pop.hawks,
    owls := if pop.owls = minPop then pop.owls + increase else pop.owls }

theorem population_difference_after_increase (initialPop : BirdPopulation) :
  initialPop.eagles = 150 →
  initialPop.falcons = 200 →
  initialPop.hawks = 320 →
  initialPop.owls = 270 →
  populationDifference (increaseLeastPopulous initialPop) = 155 := by
  sorry

end NUMINAMATH_CALUDE_population_difference_after_increase_l913_91327


namespace NUMINAMATH_CALUDE_speed_of_light_scientific_notation_l913_91389

def speed_of_light : ℝ := 300000000

theorem speed_of_light_scientific_notation : 
  speed_of_light = 3 * (10 : ℝ) ^ 8 := by sorry

end NUMINAMATH_CALUDE_speed_of_light_scientific_notation_l913_91389


namespace NUMINAMATH_CALUDE_coin_stack_order_l913_91324

-- Define the type for coins
inductive Coin | A | B | C | D | E

-- Define the covering relation
def covers (x y : Coin) : Prop := sorry

-- Define the partial covering relation
def partially_covers (x y : Coin) : Prop := sorry

-- Define the order relation
def above (x y : Coin) : Prop := sorry

-- State the theorem
theorem coin_stack_order :
  (partially_covers Coin.A Coin.B) →
  (covers Coin.C Coin.A) →
  (covers Coin.C Coin.D) →
  (covers Coin.D Coin.B) →
  (¬ covers Coin.D Coin.E) →
  (covers Coin.C Coin.E) →
  (∀ x, ¬ covers Coin.E x) →
  (above Coin.C Coin.E) ∧
  (above Coin.E Coin.A) ∧
  (above Coin.E Coin.D) ∧
  (above Coin.A Coin.B) ∧
  (above Coin.D Coin.B) :=
by sorry

end NUMINAMATH_CALUDE_coin_stack_order_l913_91324


namespace NUMINAMATH_CALUDE_sqrt_123400_l913_91328

theorem sqrt_123400 (h1 : Real.sqrt 12.34 = 3.512) (h2 : Real.sqrt 123.4 = 11.108) :
  Real.sqrt 123400 = 351.2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_123400_l913_91328


namespace NUMINAMATH_CALUDE_bruce_payment_l913_91321

/-- The total amount Bruce paid to the shopkeeper -/
def total_amount (grape_quantity mangoe_quantity grape_rate mangoe_rate : ℕ) : ℕ :=
  grape_quantity * grape_rate + mangoe_quantity * mangoe_rate

/-- Theorem stating that Bruce paid 1000 to the shopkeeper -/
theorem bruce_payment :
  total_amount 8 8 70 55 = 1000 := by
  sorry

end NUMINAMATH_CALUDE_bruce_payment_l913_91321


namespace NUMINAMATH_CALUDE_rio_persimmon_picking_l913_91361

/-- Given the conditions of Rio's persimmon picking, calculate the average number of persimmons
    she must pick from each of the last 5 trees to achieve her desired overall average. -/
theorem rio_persimmon_picking (first_pick : ℕ) (first_trees : ℕ) (remaining_trees : ℕ) (desired_avg : ℚ) :
  first_pick = 12 →
  first_trees = 5 →
  remaining_trees = 5 →
  desired_avg = 4 →
  (desired_avg * (first_trees + remaining_trees) - first_pick) / remaining_trees = 28/5 := by
  sorry

end NUMINAMATH_CALUDE_rio_persimmon_picking_l913_91361


namespace NUMINAMATH_CALUDE_circle_equation_proof_l913_91325

theorem circle_equation_proof (x y : ℝ) :
  (∃ (h k r : ℝ), r > 0 ∧ ∀ x y, x^2 + y^2 + 1 = 2*x + 4*y ↔ (x - h)^2 + (y - k)^2 = r^2) ∧
  (∃ (h k : ℝ), ∀ x y, x^2 + y^2 + 1 = 2*x + 4*y ↔ (x - h)^2 + (y - k)^2 = 4) :=
by sorry

#check circle_equation_proof

end NUMINAMATH_CALUDE_circle_equation_proof_l913_91325


namespace NUMINAMATH_CALUDE_inequality_count_l913_91388

theorem inequality_count (x y a b : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (ha : a ≠ 0) (hb : b ≠ 0)
  (hxa : x^2 < a^2) (hyb : y^2 < b^2) : 
  ∃! n : ℕ, n = 2 ∧ 
  (n = (ite (x + y < a + b) 1 0 : ℕ) + 
       (ite (x + y^2 < a + b^2) 1 0 : ℕ) + 
       (ite (x * y < a * b) 1 0 : ℕ) + 
       (ite (|x / y| < |a / b|) 1 0 : ℕ)) :=
by sorry

end NUMINAMATH_CALUDE_inequality_count_l913_91388


namespace NUMINAMATH_CALUDE_smallest_valid_n_l913_91364

def is_valid (n : ℕ) : Prop :=
  ∃ k : ℕ, 17 * n - 1 = 11 * k

theorem smallest_valid_n :
  ∃ n : ℕ, n > 0 ∧ is_valid n ∧ ∀ m : ℕ, 0 < m ∧ m < n → ¬is_valid m :=
by sorry

end NUMINAMATH_CALUDE_smallest_valid_n_l913_91364


namespace NUMINAMATH_CALUDE_greatest_common_multiple_under_150_l913_91326

theorem greatest_common_multiple_under_150 (n : ℕ) :
  (n % 10 = 0 ∧ n % 15 = 0 ∧ n < 150) →
  n ≤ 120 :=
by sorry

end NUMINAMATH_CALUDE_greatest_common_multiple_under_150_l913_91326


namespace NUMINAMATH_CALUDE_total_points_is_201_l913_91377

/- Define the scoring for Mark's team -/
def marks_team_two_pointers : ℕ := 25
def marks_team_three_pointers : ℕ := 8
def marks_team_free_throws : ℕ := 10

/- Define the scoring for the opponents relative to Mark's team -/
def opponents_two_pointers : ℕ := 2 * marks_team_two_pointers
def opponents_three_pointers : ℕ := marks_team_three_pointers / 2
def opponents_free_throws : ℕ := marks_team_free_throws / 2

/- Calculate the total points for both teams -/
def total_points : ℕ := 
  (marks_team_two_pointers * 2 + marks_team_three_pointers * 3 + marks_team_free_throws) +
  (opponents_two_pointers * 2 + opponents_three_pointers * 3 + opponents_free_throws)

/- Theorem stating that the total points scored by both teams is 201 -/
theorem total_points_is_201 : total_points = 201 := by
  sorry

end NUMINAMATH_CALUDE_total_points_is_201_l913_91377


namespace NUMINAMATH_CALUDE_no_simultaneous_overtake_l913_91332

/-- Proves that there is no time when Teena is simultaneously 25 miles ahead of Yoe and 10 miles ahead of Lona -/
theorem no_simultaneous_overtake :
  ¬ ∃ t : ℝ, t > 0 ∧ 
  (85 * t - 60 * t = 25 + 17.5) ∧ 
  (85 * t - 70 * t = 10 + 20) :=
sorry

end NUMINAMATH_CALUDE_no_simultaneous_overtake_l913_91332


namespace NUMINAMATH_CALUDE_quadratic_roots_identity_l913_91349

theorem quadratic_roots_identity (p q r s k : ℝ) (α β : ℝ) : 
  (α^2 + p*α + q = 0) →
  (β^2 + r*β + s = 0) →
  (α / β = k) →
  (q - k^2 * s)^2 + k * (p - k * r) * (k * p * s - q * r) = 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_identity_l913_91349


namespace NUMINAMATH_CALUDE_qiannan_establishment_year_l913_91338

/-- Represents the Heavenly Stems -/
inductive HeavenlyStem
| Jia | Yi | Bing | Ding | Wu | Ji | Geng | Xin | Ren | Gui

/-- Represents the Earthly Branches -/
inductive EarthlyBranch
| Zi | Chou | Yin | Mao | Chen | Si | Wu | Wei | Shen | You | Xu | Hai

/-- Represents a year in the Heavenly Stems and Earthly Branches system -/
structure StemBranchYear :=
  (stem : HeavenlyStem)
  (branch : EarthlyBranch)

/-- Function to get the previous stem -/
def prevStem (s : HeavenlyStem) : HeavenlyStem :=
  match s with
  | HeavenlyStem.Jia => HeavenlyStem.Gui
  | HeavenlyStem.Yi => HeavenlyStem.Jia
  | HeavenlyStem.Bing => HeavenlyStem.Yi
  | HeavenlyStem.Ding => HeavenlyStem.Bing
  | HeavenlyStem.Wu => HeavenlyStem.Ding
  | HeavenlyStem.Ji => HeavenlyStem.Wu
  | HeavenlyStem.Geng => HeavenlyStem.Ji
  | HeavenlyStem.Xin => HeavenlyStem.Geng
  | HeavenlyStem.Ren => HeavenlyStem.Xin
  | HeavenlyStem.Gui => HeavenlyStem.Ren

/-- Function to get the previous branch -/
def prevBranch (b : EarthlyBranch) : EarthlyBranch :=
  match b with
  | EarthlyBranch.Zi => EarthlyBranch.Hai
  | EarthlyBranch.Chou => EarthlyBranch.Zi
  | EarthlyBranch.Yin => EarthlyBranch.Chou
  | EarthlyBranch.Mao => EarthlyBranch.Yin
  | EarthlyBranch.Chen => EarthlyBranch.Mao
  | EarthlyBranch.Si => EarthlyBranch.Chen
  | EarthlyBranch.Wu => EarthlyBranch.Si
  | EarthlyBranch.Wei => EarthlyBranch.Wu
  | EarthlyBranch.Shen => EarthlyBranch.Wei
  | EarthlyBranch.You => EarthlyBranch.Shen
  | EarthlyBranch.Xu => EarthlyBranch.You
  | EarthlyBranch.Hai => EarthlyBranch.Xu

/-- Function to get the year n years before a given year -/
def yearsBefore (n : Nat) (year : StemBranchYear) : StemBranchYear :=
  if n = 0 then year
  else yearsBefore (n - 1) { stem := prevStem year.stem, branch := prevBranch year.branch }

theorem qiannan_establishment_year :
  let year2023 : StemBranchYear := { stem := HeavenlyStem.Gui, branch := EarthlyBranch.Mao }
  let establishmentYear := yearsBefore 67 year2023
  establishmentYear.stem = HeavenlyStem.Bing ∧ establishmentYear.branch = EarthlyBranch.Shen :=
by sorry

end NUMINAMATH_CALUDE_qiannan_establishment_year_l913_91338


namespace NUMINAMATH_CALUDE_fixed_point_on_line_l913_91348

/-- The line (a-1)x + ay + 3 = 0 passes through the point (3, -3) for any real a -/
theorem fixed_point_on_line (a : ℝ) : (a - 1) * 3 + a * (-3) + 3 = 0 := by
  sorry

end NUMINAMATH_CALUDE_fixed_point_on_line_l913_91348


namespace NUMINAMATH_CALUDE_complement_of_A_l913_91353

-- Define the universal set U as the set of real numbers
def U : Set ℝ := Set.univ

-- Define set A
def A : Set ℝ := {x : ℝ | x ≥ 1}

-- State the theorem
theorem complement_of_A : Set.compl A = {x : ℝ | x < 1} := by sorry

end NUMINAMATH_CALUDE_complement_of_A_l913_91353


namespace NUMINAMATH_CALUDE_mile_to_rod_l913_91365

-- Define the units
def mile : ℕ := 1
def furlong : ℕ := 1
def rod : ℕ := 1

-- Define the conversion rates
axiom mile_to_furlong : mile = 6 * furlong
axiom furlong_to_rod : furlong = 60 * rod

-- Theorem to prove
theorem mile_to_rod : mile = 360 * rod := by
  sorry

end NUMINAMATH_CALUDE_mile_to_rod_l913_91365


namespace NUMINAMATH_CALUDE_f_iterated_property_l913_91376

-- Define the function f
def f (p q : ℝ) (x : ℝ) : ℝ := x^2 + p*x + q

-- Define the iteration of f
def iterate_f (p q : ℝ) : ℕ → ℝ → ℝ
  | 0, x => x
  | n+1, x => iterate_f p q n (f p q x)

theorem f_iterated_property (p q : ℝ) 
  (h : ∀ x ∈ Set.Icc 1 3, |f p q x| ≤ 1/2) :
  iterate_f p q 2017 ((3 + Real.sqrt 7) / 2) = (3 - Real.sqrt 7) / 2 := by
  sorry

end NUMINAMATH_CALUDE_f_iterated_property_l913_91376


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l913_91357

-- Define sets A and B
def A : Set (ℝ × ℝ) := {p | 3 * p.1 + p.2 = 0}
def B : Set (ℝ × ℝ) := {p | 2 * p.1 - p.2 = 3}

-- Theorem statement
theorem intersection_of_A_and_B : A ∩ B = {(3/5, -9/5)} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l913_91357


namespace NUMINAMATH_CALUDE_polynomial_sum_of_terms_l913_91301

def polynomial (x : ℝ) : ℝ := 4 * x^2 - 3 * x - 2

def term1 (x : ℝ) : ℝ := 4 * x^2
def term2 (x : ℝ) : ℝ := -3 * x
def term3 : ℝ := -2

theorem polynomial_sum_of_terms :
  ∀ x : ℝ, polynomial x = term1 x + term2 x + term3 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_sum_of_terms_l913_91301


namespace NUMINAMATH_CALUDE_possible_ordering_l913_91380

theorem possible_ordering (a b c : ℝ) 
  (distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c)
  (positive : a > 0 ∧ b > 0 ∧ c > 0)
  (eq : a^2 + c^2 = 2*b*c) :
  b > a ∧ a > c :=
sorry

end NUMINAMATH_CALUDE_possible_ordering_l913_91380


namespace NUMINAMATH_CALUDE_system_solution_l913_91341

theorem system_solution : ∃ (x y : ℚ), 
  (3 * x - 4 * y = -7) ∧ 
  (7 * x - 3 * y = 5) ∧ 
  (x = 41 / 19) ∧ 
  (y = 64 / 19) := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l913_91341


namespace NUMINAMATH_CALUDE_red_cell_remains_l913_91308

theorem red_cell_remains (n : ℕ) :
  ∀ (black_rows black_cols : Finset (Fin (2*n))),
  black_rows.card = n ∧ black_cols.card = n →
  ∃ (red_cells : Finset (Fin (2*n) × Fin (2*n))),
  red_cells.card = 2*n^2 + 1 ∧
  ∃ (cell : Fin (2*n) × Fin (2*n)),
  cell ∈ red_cells ∧ cell.1 ∉ black_rows ∧ cell.2 ∉ black_cols :=
sorry

end NUMINAMATH_CALUDE_red_cell_remains_l913_91308


namespace NUMINAMATH_CALUDE_ceiling_of_negative_real_l913_91307

theorem ceiling_of_negative_real : ⌈(-3.67 : ℝ)⌉ = -3 := by sorry

end NUMINAMATH_CALUDE_ceiling_of_negative_real_l913_91307


namespace NUMINAMATH_CALUDE_sum_of_digits_of_seven_to_fifteen_l913_91315

def last_two_digits (n : ℕ) : ℕ := n % 100

def tens_digit (n : ℕ) : ℕ := (n / 10) % 10

def ones_digit (n : ℕ) : ℕ := n % 10

theorem sum_of_digits_of_seven_to_fifteen (n : ℕ) (h : n = (3 + 4)^15) :
  tens_digit (last_two_digits n) + ones_digit (last_two_digits n) = 7 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_digits_of_seven_to_fifteen_l913_91315


namespace NUMINAMATH_CALUDE_output_for_15_l913_91367

def function_machine (input : ℕ) : ℕ :=
  let step1 := input * 3
  if step1 > 20 then step1 - 7 else step1 + 10

theorem output_for_15 : function_machine 15 = 38 := by
  sorry

end NUMINAMATH_CALUDE_output_for_15_l913_91367


namespace NUMINAMATH_CALUDE_mans_walking_speed_l913_91331

/-- Proves that given a man who walks a certain distance in 5 hours and runs the same distance at 15 kmph in 36 minutes, his walking speed is 1.8 kmph. -/
theorem mans_walking_speed 
  (walking_time : ℝ) 
  (running_speed : ℝ) 
  (running_time_minutes : ℝ) :
  walking_time = 5 →
  running_speed = 15 →
  running_time_minutes = 36 →
  (walking_time * (running_speed * (running_time_minutes / 60))) / walking_time = 1.8 :=
by sorry

end NUMINAMATH_CALUDE_mans_walking_speed_l913_91331


namespace NUMINAMATH_CALUDE_eleven_items_division_l913_91311

theorem eleven_items_division (n : ℕ) (h : n = 11) : 
  (Finset.sum (Finset.range 3) (λ k => Nat.choose n (k + 3))) = 957 := by
  sorry

end NUMINAMATH_CALUDE_eleven_items_division_l913_91311


namespace NUMINAMATH_CALUDE_power_division_rule_l913_91381

theorem power_division_rule (a : ℝ) : a^4 / a^3 = a := by
  sorry

end NUMINAMATH_CALUDE_power_division_rule_l913_91381


namespace NUMINAMATH_CALUDE_square_perimeter_side_ratio_l913_91319

theorem square_perimeter_side_ratio (s : ℝ) (hs : s > 0) :
  let new_side := s + 1
  let new_perimeter := 4 * new_side
  new_perimeter / new_side = 4 := by
sorry

end NUMINAMATH_CALUDE_square_perimeter_side_ratio_l913_91319


namespace NUMINAMATH_CALUDE_parabola_equation_l913_91374

/-- A parabola is a set of points in a plane that are equidistant from a fixed point (focus) and a fixed line (directrix). -/
structure Parabola where
  focus : ℝ × ℝ
  opens_left : Bool

/-- The standard form equation of a parabola. -/
def standard_equation (p : Parabola) : ℝ → ℝ → Prop :=
  fun x y => y^2 = -4 * p.focus.1 * x

theorem parabola_equation (p : Parabola) 
  (h1 : p.focus = (-3, 0)) 
  (h2 : p.opens_left = true) : 
  standard_equation p = fun x y => y^2 = -12 * x := by
sorry

end NUMINAMATH_CALUDE_parabola_equation_l913_91374


namespace NUMINAMATH_CALUDE_rent_increase_problem_l913_91317

/-- Given a group of 4 friends paying rent, where:
  - The initial average rent is $800
  - After one person's rent increases by 20%, the new average is $870
  This theorem proves that the original rent of the person whose rent increased was $1400. -/
theorem rent_increase_problem (initial_average : ℝ) (new_average : ℝ) (num_friends : ℕ) 
  (increase_percentage : ℝ) (h1 : initial_average = 800)
  (h2 : new_average = 870) (h3 : num_friends = 4) (h4 : increase_percentage = 0.2) :
  ∃ (original_rent : ℝ), 
    original_rent * (1 + increase_percentage) = 
      num_friends * new_average - (num_friends - 1) * initial_average ∧
    original_rent = 1400 :=
by sorry

end NUMINAMATH_CALUDE_rent_increase_problem_l913_91317


namespace NUMINAMATH_CALUDE_cricket_team_age_theorem_l913_91336

def cricket_team_age_problem (team_size : ℕ) (captain_age : ℕ) (wicket_keeper_age_diff : ℕ) (remaining_players_age_diff : ℕ) : Prop :=
  let total_age := team_size * average_age
  let captain_and_keeper_age := captain_age + (captain_age + wicket_keeper_age_diff)
  let remaining_players := team_size - 2
  total_age - captain_and_keeper_age = remaining_players * (average_age - remaining_players_age_diff)
  where
    average_age : ℕ := 23

theorem cricket_team_age_theorem :
  cricket_team_age_problem 11 26 3 1 := by
  sorry

end NUMINAMATH_CALUDE_cricket_team_age_theorem_l913_91336


namespace NUMINAMATH_CALUDE_function_inequality_l913_91333

theorem function_inequality (f : ℝ → ℝ) (h_diff : Differentiable ℝ f) 
  (h_ineq : ∀ x, deriv f x < f x) : 
  f 1 < Real.exp 1 * f 0 ∧ f 2013 < Real.exp 2013 * f 0 := by
  sorry

end NUMINAMATH_CALUDE_function_inequality_l913_91333


namespace NUMINAMATH_CALUDE_sum_of_fractions_l913_91345

theorem sum_of_fractions : 
  (1 / (2 * 3 : ℚ)) + (1 / (3 * 4 : ℚ)) + (1 / (4 * 5 : ℚ)) + 
  (1 / (5 * 6 : ℚ)) + (1 / (6 * 7 : ℚ)) + (1 / (7 * 8 : ℚ)) + 
  (1 / (8 * 9 : ℚ)) = 7 / 18 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_fractions_l913_91345


namespace NUMINAMATH_CALUDE_zeros_after_one_in_power_l913_91302

theorem zeros_after_one_in_power (n : ℕ) (h : 10000 = 10^4) :
  10000^50 = 10^200 := by
  sorry

end NUMINAMATH_CALUDE_zeros_after_one_in_power_l913_91302


namespace NUMINAMATH_CALUDE_geometric_sequence_product_l913_91346

/-- Given a geometric sequence {a_n} where a₄ = 4, prove that a₃a₅ = 16 -/
theorem geometric_sequence_product (a : ℕ → ℝ) :
  (∀ n m : ℕ, a (n + 1) / a n = a (m + 1) / a m) →  -- geometric sequence condition
  a 4 = 4 →                                        -- given condition
  a 3 * a 5 = 16 :=                                -- conclusion to prove
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_product_l913_91346


namespace NUMINAMATH_CALUDE_intersection_equals_interval_l913_91373

open Set

def S : Set ℝ := {x | x > -2}
def T : Set ℝ := {x | -4 ≤ x ∧ x ≤ 1}

theorem intersection_equals_interval : S ∩ T = Ioc (-2) 1 := by sorry

end NUMINAMATH_CALUDE_intersection_equals_interval_l913_91373


namespace NUMINAMATH_CALUDE_cos_two_theta_value_l913_91351

theorem cos_two_theta_value (θ : Real) 
  (h : Real.exp (Real.log 2 * (1 - 3/2 + 3 * Real.cos θ)) + 3 = Real.exp (Real.log 2 * (2 + Real.cos θ))) :
  Real.cos (2 * θ) = -1/2 := by
sorry

end NUMINAMATH_CALUDE_cos_two_theta_value_l913_91351


namespace NUMINAMATH_CALUDE_fifth_root_of_3125_l913_91330

theorem fifth_root_of_3125 (x : ℝ) (h1 : x > 0) (h2 : x^5 = 3125) : x = 5 := by
  sorry

end NUMINAMATH_CALUDE_fifth_root_of_3125_l913_91330


namespace NUMINAMATH_CALUDE_geometric_sequence_eighth_term_l913_91339

/-- Given a geometric sequence of positive numbers where the fifth term is 16 and the eleventh term is 2, 
    prove that the eighth term is 4√2. -/
theorem geometric_sequence_eighth_term 
  (a : ℝ) 
  (r : ℝ) 
  (h1 : a > 0) 
  (h2 : r > 0) 
  (h3 : a * r^4 = 16) 
  (h4 : a * r^10 = 2) : 
  a * r^7 = 4 * Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_eighth_term_l913_91339


namespace NUMINAMATH_CALUDE_men_in_club_l913_91382

theorem men_in_club (total : ℕ) (attendees : ℕ) (h_total : total = 30) (h_attendees : attendees = 18) :
  ∃ (men women : ℕ),
    men + women = total ∧
    men + (women / 3) = attendees ∧
    men = 12 := by
  sorry

end NUMINAMATH_CALUDE_men_in_club_l913_91382


namespace NUMINAMATH_CALUDE_jelly_bean_probability_l913_91342

theorem jelly_bean_probability (p_r p_o p_y p_g : ℝ) :
  p_r = 0.1 →
  p_o = 0.4 →
  p_r + p_o + p_y + p_g = 1 →
  p_y + p_g = 0.5 := by
sorry

end NUMINAMATH_CALUDE_jelly_bean_probability_l913_91342


namespace NUMINAMATH_CALUDE_min_games_prediction_l913_91312

/-- Represents a chess tournament between two schools -/
structure ChessTournament where
  white_rook : ℕ  -- Number of students from "White Rook" school
  black_elephant : ℕ  -- Number of students from "Black Elephant" school
  total_games : ℕ  -- Total number of games to be played

/-- Predicate to check if a tournament setup is valid -/
def valid_tournament (t : ChessTournament) : Prop :=
  t.white_rook * t.black_elephant = t.total_games

/-- The minimum number of games after which one can definitely name a participant -/
def min_games_to_predict (t : ChessTournament) : ℕ :=
  t.total_games - t.black_elephant

/-- Theorem stating the minimum number of games for prediction in the given tournament -/
theorem min_games_prediction (t : ChessTournament) 
  (h_valid : valid_tournament t) 
  (h_white : t.white_rook = 15) 
  (h_black : t.black_elephant = 20) 
  (h_total : t.total_games = 300) : 
  min_games_to_predict t = 280 := by
  sorry

#eval min_games_to_predict { white_rook := 15, black_elephant := 20, total_games := 300 }

end NUMINAMATH_CALUDE_min_games_prediction_l913_91312


namespace NUMINAMATH_CALUDE_complement_intersection_theorem_l913_91390

-- Define the universal set U
def U : Set Nat := {1, 2, 3, 4, 5}

-- Define set M
def M : Set Nat := {1, 2, 3}

-- Define set N
def N : Set Nat := {2, 3, 5}

-- Theorem statement
theorem complement_intersection_theorem :
  (U \ M) ∩ N = {5} := by
  sorry

end NUMINAMATH_CALUDE_complement_intersection_theorem_l913_91390


namespace NUMINAMATH_CALUDE_three_distinct_values_l913_91352

/-- The number of distinct values possible when evaluating 3^(3^(3^3)) with different parenthesizations -/
def num_distinct_values : ℕ := 3

/-- The original expression 3^(3^(3^3)) -/
def original_expr : ℕ := 3^(3^(3^3))

theorem three_distinct_values :
  ∃ (a b : ℕ), a ≠ b ∧ a ≠ original_expr ∧ b ≠ original_expr ∧
  (∀ (x : ℕ), x ≠ a ∧ x ≠ b ∧ x ≠ original_expr →
    ¬∃ (e₁ e₂ e₃ : ℕ → ℕ → ℕ), x = e₁ 3 (e₂ 3 (e₃ 3 3))) ∧
  num_distinct_values = 3 :=
sorry

end NUMINAMATH_CALUDE_three_distinct_values_l913_91352


namespace NUMINAMATH_CALUDE_no_integer_square_root_l913_91323

/-- The polynomial p(x) = x^4 + 6x^3 + 11x^2 + 13x + 37 -/
def p (x : ℤ) : ℤ := x^4 + 6*x^3 + 11*x^2 + 13*x + 37

/-- Theorem stating that there are no integer values of x such that p(x) is a perfect square -/
theorem no_integer_square_root : ∀ x : ℤ, ¬∃ y : ℤ, p x = y^2 := by
  sorry

end NUMINAMATH_CALUDE_no_integer_square_root_l913_91323


namespace NUMINAMATH_CALUDE_problem_solution_l913_91391

theorem problem_solution (x y : ℝ) (hx : x = Real.sqrt 3 + 1) (hy : y = Real.sqrt 3 - 1) :
  (x^2 + 2*x*y + y^2 = 12) ∧ (x^2 - y^2 = 4 * Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l913_91391


namespace NUMINAMATH_CALUDE_square_not_sum_of_periodic_l913_91305

-- Define a periodic function
def Periodic (f : ℝ → ℝ) : Prop :=
  ∃ p : ℝ, p > 0 ∧ ∀ x : ℝ, f (x + p) = f x

-- State the theorem
theorem square_not_sum_of_periodic :
  ¬ ∃ (g h : ℝ → ℝ), (Periodic g ∧ Periodic h) ∧ (∀ x : ℝ, x^2 = g x + h x) := by
  sorry

end NUMINAMATH_CALUDE_square_not_sum_of_periodic_l913_91305


namespace NUMINAMATH_CALUDE_petya_vasya_meeting_l913_91363

/-- The number of lampposts along the alley -/
def num_lampposts : ℕ := 100

/-- The lamppost where Petya is observed -/
def petya_observed : ℕ := 22

/-- The lamppost where Vasya is observed -/
def vasya_observed : ℕ := 88

/-- The meeting point of Petya and Vasya -/
def meeting_point : ℕ := 64

theorem petya_vasya_meeting :
  ∀ (petya_speed vasya_speed : ℚ),
    petya_speed > 0 →
    vasya_speed > 0 →
    (petya_observed - 1 : ℚ) / petya_speed = (num_lampposts - vasya_observed : ℚ) / vasya_speed →
    (meeting_point - 1 : ℚ) / petya_speed = (num_lampposts - meeting_point : ℚ) / vasya_speed :=
by sorry

end NUMINAMATH_CALUDE_petya_vasya_meeting_l913_91363


namespace NUMINAMATH_CALUDE_monotonicity_and_extrema_l913_91370

noncomputable def f (x : ℝ) : ℝ := x^3 - 3*x

theorem monotonicity_and_extrema :
  (∀ x y, x < y ∧ ((x < -1 ∧ y < -1) ∨ (x > 1 ∧ y > 1)) → f x < f y) ∧
  (∀ x y, -1 < x ∧ x < y ∧ y < 1 → f x > f y) ∧
  (∀ x ∈ Set.Icc (-3) 2, f x ≤ 2) ∧
  (∃ x ∈ Set.Icc (-3) 2, f x = 2) ∧
  (∀ x ∈ Set.Icc (-3) 2, f x ≥ -18) ∧
  (∃ x ∈ Set.Icc (-3) 2, f x = -18) :=
by sorry

end NUMINAMATH_CALUDE_monotonicity_and_extrema_l913_91370


namespace NUMINAMATH_CALUDE_parallel_lines_a_value_l913_91383

/-- Two lines are parallel if and only if they have the same slope -/
axiom parallel_lines_same_slope (m₁ m₂ : ℝ) : 
  (∃ b₁ b₂ : ℝ, ∀ x y : ℝ, y = m₁ * x + b₁ ↔ y = m₂ * x + b₂) ↔ m₁ = m₂

/-- The value of a when two given lines are parallel -/
theorem parallel_lines_a_value : 
  (∃ b₁ b₂ : ℝ, ∀ x y : ℝ, y = 3 * x + a / 3 ↔ y = (a - 3) * x + 2) → a = 6 :=
by sorry

end NUMINAMATH_CALUDE_parallel_lines_a_value_l913_91383


namespace NUMINAMATH_CALUDE_find_M_l913_91356

theorem find_M (p q r s M : ℚ) 
  (sum_eq : p + q + r + s = 100)
  (p_eq : p + 10 = M)
  (q_eq : q - 5 = M)
  (r_eq : 10 * r = M)
  (s_eq : s / 2 = M) :
  M = 1050 / 41 := by
  sorry

end NUMINAMATH_CALUDE_find_M_l913_91356


namespace NUMINAMATH_CALUDE_triangle_projection_similarity_l913_91399

/-- For any triangle, there exist perpendicular distances that make the projected triangle similar to the original -/
theorem triangle_projection_similarity (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  ∃ (x y : ℝ), x ≥ 0 ∧ y ≥ 0 ∧ 
    x^2 + b^2 = y^2 + a^2 ∧
    (x - y)^2 + c^2 = y^2 + a^2 :=
by sorry

end NUMINAMATH_CALUDE_triangle_projection_similarity_l913_91399


namespace NUMINAMATH_CALUDE_last_two_digits_sum_l913_91359

theorem last_two_digits_sum (n : ℕ) : (9^25 + 13^25) % 100 = 42 := by
  sorry

end NUMINAMATH_CALUDE_last_two_digits_sum_l913_91359


namespace NUMINAMATH_CALUDE_coin_stack_arrangements_l913_91393

/-- The number of indistinguishable gold coins -/
def num_gold_coins : ℕ := 5

/-- The number of indistinguishable silver coins -/
def num_silver_coins : ℕ := 3

/-- The total number of coins -/
def total_coins : ℕ := num_gold_coins + num_silver_coins

/-- The number of ways to arrange gold and silver coins -/
def gold_silver_arrangements : ℕ := Nat.choose total_coins num_gold_coins

/-- The number of valid head-tail sequences -/
def valid_head_tail_sequences : ℕ := total_coins + 1

/-- The total number of distinguishable arrangements -/
def total_arrangements : ℕ := gold_silver_arrangements * valid_head_tail_sequences

theorem coin_stack_arrangements :
  total_arrangements = 504 :=
sorry

end NUMINAMATH_CALUDE_coin_stack_arrangements_l913_91393


namespace NUMINAMATH_CALUDE_f_five_values_l913_91371

def FunctionProperty (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (f x + y^2) = f (x^2 - y) + 4 * (f x) * y^2

theorem f_five_values (f : ℝ → ℝ) (h : FunctionProperty f) : 
  f 5 = 0 ∨ f 5 = 25 := by sorry

end NUMINAMATH_CALUDE_f_five_values_l913_91371


namespace NUMINAMATH_CALUDE_min_words_for_90_percent_l913_91355

/-- The minimum number of words needed to achieve at least 90% on a vocabulary exam -/
theorem min_words_for_90_percent (total_words : ℕ) (min_percentage : ℚ) : 
  total_words = 600 → min_percentage = 90 / 100 → 
  ∃ (min_words : ℕ), min_words = 540 ∧ 
    (min_words : ℚ) / total_words ≥ min_percentage ∧
    ∀ (n : ℕ), n < min_words → (n : ℚ) / total_words < min_percentage :=
by sorry

end NUMINAMATH_CALUDE_min_words_for_90_percent_l913_91355


namespace NUMINAMATH_CALUDE_rectangle_property_l913_91375

-- Define the rectangle's properties
def rectangle_length (x : ℝ) : ℝ := 4 * x
def rectangle_width (x : ℝ) : ℝ := x + 3

-- Define the area and perimeter functions
def area (x : ℝ) : ℝ := rectangle_length x * rectangle_width x
def perimeter (x : ℝ) : ℝ := 2 * (rectangle_length x + rectangle_width x)

-- State the theorem
theorem rectangle_property :
  ∃ x : ℝ, x > 0 ∧ area x = 3 * perimeter x ∧ Real.sqrt ((9 + Real.sqrt 153) / 4 - x) < 0.001 :=
by sorry

end NUMINAMATH_CALUDE_rectangle_property_l913_91375


namespace NUMINAMATH_CALUDE_trapezoid_area_l913_91369

/-- A trapezoid with given side lengths -/
structure Trapezoid :=
  (BC : ℝ)
  (AD : ℝ)
  (AB : ℝ)
  (CD : ℝ)

/-- The area of a trapezoid -/
def area (t : Trapezoid) : ℝ := sorry

/-- Theorem: The area of the given trapezoid is 59 -/
theorem trapezoid_area :
  let t : Trapezoid := { BC := 9.5, AD := 20, AB := 5, CD := 8.5 }
  area t = 59 := by sorry

end NUMINAMATH_CALUDE_trapezoid_area_l913_91369


namespace NUMINAMATH_CALUDE_calculation_proof_l913_91392

theorem calculation_proof : (π - 2019)^0 + |Real.sqrt 3 - 1| + (-1/2)⁻¹ - 2 * Real.tan (30 * π / 180) = -2 + Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l913_91392


namespace NUMINAMATH_CALUDE_fraction_equation_solutions_l913_91366

theorem fraction_equation_solutions (x : ℝ) : 
  1 / (x^2 + 17*x - 8) + 1 / (x^2 + 4*x - 8) + 1 / (x^2 - 9*x - 8) = 0 ↔ 
  x = 1 ∨ x = -8 ∨ x = 2 ∨ x = -4 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equation_solutions_l913_91366


namespace NUMINAMATH_CALUDE_fraction_equality_l913_91310

theorem fraction_equality (x y : ℝ) 
  (h : (1/3)^2 + (1/4)^2 / ((1/5)^2 + (1/6)^2) = 37*x / (73*y)) : 
  Real.sqrt x / Real.sqrt y = 75 * Real.sqrt 73 / (6 * Real.sqrt 61 * Real.sqrt 37) := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l913_91310


namespace NUMINAMATH_CALUDE_smallest_cube_factor_l913_91343

theorem smallest_cube_factor (n : ℕ) (h : n = 1512) :
  (∃ (y : ℕ), y > 0 ∧ n * 49 = y^3) ∧
  (∀ (x : ℕ), x > 0 ∧ x < 49 → ¬∃ (y : ℕ), y > 0 ∧ n * x = y^3) :=
by sorry

end NUMINAMATH_CALUDE_smallest_cube_factor_l913_91343


namespace NUMINAMATH_CALUDE_perpendicular_lines_m_value_l913_91395

/-- Given two lines in the general form ax + by + c = 0, this function returns true if they are perpendicular --/
def are_perpendicular (a1 b1 c1 a2 b2 c2 : ℝ) : Prop :=
  a1 * a2 + b1 * b2 = 0

/-- The problem statement --/
theorem perpendicular_lines_m_value :
  ∀ m : ℝ, are_perpendicular m 4 (-2) 2 (-5) 1 → m = 10 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_lines_m_value_l913_91395


namespace NUMINAMATH_CALUDE_train_length_calculation_l913_91316

/-- Calculates the length of a train given its speed, time to pass a bridge, and the bridge length. -/
theorem train_length_calculation (train_speed : ℝ) (time_to_pass : ℝ) (bridge_length : ℝ) :
  train_speed = 30 →
  time_to_pass = 60 →
  bridge_length = 140 →
  ∃ (train_length : ℝ), abs (train_length - 359.8) < 0.1 :=
by
  sorry

end NUMINAMATH_CALUDE_train_length_calculation_l913_91316


namespace NUMINAMATH_CALUDE_empty_set_implies_a_range_l913_91396

theorem empty_set_implies_a_range (a : ℝ) : 
  (∀ x : ℝ, a * x^2 - |x + 1| + 2 * a ≥ 0) → 
  a > (Real.sqrt 3 + 1) / 4 := by
sorry

end NUMINAMATH_CALUDE_empty_set_implies_a_range_l913_91396


namespace NUMINAMATH_CALUDE_equation_equality_l913_91344

theorem equation_equality (a b : ℝ) : (a - b)^3 * (b - a)^4 = (a - b)^7 := by
  sorry

end NUMINAMATH_CALUDE_equation_equality_l913_91344


namespace NUMINAMATH_CALUDE_square_area_multiple_l913_91322

theorem square_area_multiple (a p m : ℝ) : 
  a > 0 → 
  p > 0 → 
  p = 36 → 
  a = (p / 4)^2 → 
  m * a = 10 * p + 45 → 
  m = 5 := by
sorry

end NUMINAMATH_CALUDE_square_area_multiple_l913_91322


namespace NUMINAMATH_CALUDE_garden_fence_posts_l913_91350

/-- Calculates the number of fence posts required for a rectangular garden -/
def fencePostsRequired (length width postSpacing : ℕ) : ℕ :=
  let perimeter := 2 * (length + width)
  let postsOnPerimeter := perimeter / postSpacing
  postsOnPerimeter + 1

/-- Theorem stating the number of fence posts required for the specific garden -/
theorem garden_fence_posts :
  fencePostsRequired 72 32 8 = 26 := by
  sorry

end NUMINAMATH_CALUDE_garden_fence_posts_l913_91350


namespace NUMINAMATH_CALUDE_school_capacity_l913_91300

/-- Given a school with the following properties:
  * It has 15 classrooms
  * One-third of the classrooms have 30 desks each
  * The rest of the classrooms have 25 desks each
  * Only one student can sit at one desk
  This theorem proves that the school can accommodate 400 students. -/
theorem school_capacity :
  let total_classrooms : ℕ := 15
  let desks_per_large_classroom : ℕ := 30
  let desks_per_small_classroom : ℕ := 25
  let large_classrooms : ℕ := total_classrooms / 3
  let small_classrooms : ℕ := total_classrooms - large_classrooms
  let total_capacity : ℕ := large_classrooms * desks_per_large_classroom +
                            small_classrooms * desks_per_small_classroom
  total_capacity = 400 := by
  sorry

end NUMINAMATH_CALUDE_school_capacity_l913_91300


namespace NUMINAMATH_CALUDE_probability_theorem_l913_91372

/-- A rectangle with dimensions 3 × 2 units -/
structure Rectangle where
  length : ℝ
  width : ℝ

/-- 10 points evenly spaced along the perimeter of the rectangle -/
def num_points : ℕ := 10

/-- The probability of selecting two points one unit apart -/
def probability_one_unit_apart (rect : Rectangle) : ℚ :=
  2 / 9

/-- Theorem stating the probability of selecting two points one unit apart -/
theorem probability_theorem (rect : Rectangle) 
  (h1 : rect.length = 3) 
  (h2 : rect.width = 2) : 
  probability_one_unit_apart rect = 2 / 9 := by
  sorry

end NUMINAMATH_CALUDE_probability_theorem_l913_91372


namespace NUMINAMATH_CALUDE_corner_cut_pentagon_area_l913_91314

/-- Pentagon formed by cutting a triangular corner from a rectangle --/
structure CornerCutPentagon where
  sides : Finset ℕ
  is_valid : sides = {14, 21, 22, 28, 37}

/-- The area of the CornerCutPentagon --/
def pentagon_area (p : CornerCutPentagon) : ℕ := sorry

/-- Theorem stating that the area of the CornerCutPentagon is 826 --/
theorem corner_cut_pentagon_area (p : CornerCutPentagon) : pentagon_area p = 826 := by sorry

end NUMINAMATH_CALUDE_corner_cut_pentagon_area_l913_91314


namespace NUMINAMATH_CALUDE_base_representation_five_digits_l913_91318

theorem base_representation_five_digits (b' : ℕ+) : 
  (∃ (a b c d e : ℕ), a ≠ 0 ∧ 216 = a*(b'^4) + b*(b'^3) + c*(b'^2) + d*(b'^1) + e ∧ 
   a < b' ∧ b < b' ∧ c < b' ∧ d < b' ∧ e < b') ↔ b' = 3 :=
sorry

end NUMINAMATH_CALUDE_base_representation_five_digits_l913_91318


namespace NUMINAMATH_CALUDE_two_red_two_blue_probability_l913_91358

def total_marbles : ℕ := 20
def red_marbles : ℕ := 12
def blue_marbles : ℕ := 8
def selected_marbles : ℕ := 4

def probability_two_red_two_blue : ℚ :=
  6 * (red_marbles * (red_marbles - 1) * blue_marbles * (blue_marbles - 1)) /
  (total_marbles * (total_marbles - 1) * (total_marbles - 2) * (total_marbles - 3))

theorem two_red_two_blue_probability :
  probability_two_red_two_blue = 1232 / 4845 := by
  sorry

end NUMINAMATH_CALUDE_two_red_two_blue_probability_l913_91358


namespace NUMINAMATH_CALUDE_circle_and_line_intersection_l913_91384

-- Define the parabola
def parabola (x y : ℝ) : Prop := y = x^2 - 4*x + 3

-- Define the circle C
def circle_C (x y : ℝ) : Prop := (x - 2)^2 + (y - 2)^2 = 5

-- Define the line
def line (x y : ℝ) : Prop := 2*x - y + 2 = 0

-- Theorem statement
theorem circle_and_line_intersection :
  -- Circle C passes through intersection points of parabola and coordinate axes
  (∃ x₁ x₂ x₃ y₃ : ℝ, 
    parabola x₁ 0 ∧ parabola x₂ 0 ∧ parabola 0 y₃ ∧
    circle_C x₁ 0 ∧ circle_C x₂ 0 ∧ circle_C 0 y₃) →
  -- Line intersects circle C at two points
  (∃ A B : ℝ × ℝ, 
    line A.1 A.2 ∧ line B.1 B.2 ∧
    circle_C A.1 A.2 ∧ circle_C B.1 B.2 ∧ A ≠ B) →
  -- Distance between intersection points is 6√5/5
  ∃ A B : ℝ × ℝ, line A.1 A.2 ∧ line B.1 B.2 ∧
    circle_C A.1 A.2 ∧ circle_C B.1 B.2 ∧
    Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = (6 * Real.sqrt 5) / 5 :=
by
  sorry


end NUMINAMATH_CALUDE_circle_and_line_intersection_l913_91384


namespace NUMINAMATH_CALUDE_jessy_jokes_count_l913_91379

theorem jessy_jokes_count (jessy_jokes alan_jokes : ℕ) : 
  alan_jokes = 7 →
  2 * (jessy_jokes + alan_jokes) = 54 →
  jessy_jokes = 20 := by
sorry

end NUMINAMATH_CALUDE_jessy_jokes_count_l913_91379


namespace NUMINAMATH_CALUDE_brenda_banana_pudding_trays_l913_91362

/-- Proof that Brenda can make 3 trays of banana pudding given the conditions --/
theorem brenda_banana_pudding_trays :
  ∀ (cookies_per_tray : ℕ) 
    (cookies_per_box : ℕ) 
    (cost_per_box : ℚ) 
    (total_spent : ℚ),
  cookies_per_tray = 80 →
  cookies_per_box = 60 →
  cost_per_box = 7/2 →
  total_spent = 14 →
  (total_spent / cost_per_box * cookies_per_box) / cookies_per_tray = 3 :=
by
  sorry

#check brenda_banana_pudding_trays

end NUMINAMATH_CALUDE_brenda_banana_pudding_trays_l913_91362


namespace NUMINAMATH_CALUDE_train_crossing_length_train_B_length_l913_91386

/-- The length of a train crossing another train in opposite direction --/
theorem train_crossing_length (length_A : ℝ) (speed_A speed_B : ℝ) (time : ℝ) : ℝ :=
  let relative_speed := (speed_A + speed_B) * (1000 / 3600)
  let total_distance := relative_speed * time
  total_distance - length_A

/-- Proof of the length of Train B --/
theorem train_B_length : 
  train_crossing_length 360 120 150 6 = 90 := by
  sorry

end NUMINAMATH_CALUDE_train_crossing_length_train_B_length_l913_91386


namespace NUMINAMATH_CALUDE_cos_20_cos_25_minus_sin_20_sin_25_l913_91354

theorem cos_20_cos_25_minus_sin_20_sin_25 :
  Real.cos (20 * Real.pi / 180) * Real.cos (25 * Real.pi / 180) -
  Real.sin (20 * Real.pi / 180) * Real.sin (25 * Real.pi / 180) =
  Real.sqrt 2 / 2 := by
sorry

end NUMINAMATH_CALUDE_cos_20_cos_25_minus_sin_20_sin_25_l913_91354


namespace NUMINAMATH_CALUDE_smallest_n_with_four_pairs_l913_91335

/-- Returns the number of distinct ordered pairs of positive integers (a, b) such that a^2 + b^2 = n -/
def f (n : ℕ) : ℕ :=
  (Finset.filter (fun p : ℕ × ℕ => p.1^2 + p.2^2 = n ∧ p.1 > 0 ∧ p.2 > 0) (Finset.product (Finset.range n) (Finset.range n))).card

/-- 26 is the smallest positive integer n for which f(n) = 4 -/
theorem smallest_n_with_four_pairs : ∀ m : ℕ, m > 0 → m < 26 → f m ≠ 4 ∧ f 26 = 4 := by
  sorry

end NUMINAMATH_CALUDE_smallest_n_with_four_pairs_l913_91335


namespace NUMINAMATH_CALUDE_pencil_and_pen_cost_l913_91387

/-- The cost of a pencil -/
def pencil_cost : ℝ := sorry

/-- The cost of a pen -/
def pen_cost : ℝ := sorry

/-- The first condition: four pencils and three pens cost $3.70 -/
axiom condition1 : 4 * pencil_cost + 3 * pen_cost = 3.70

/-- The second condition: three pencils and four pens cost $4.20 -/
axiom condition2 : 3 * pencil_cost + 4 * pen_cost = 4.20

/-- Theorem: The cost of one pencil and one pen is $1.1286 -/
theorem pencil_and_pen_cost : pencil_cost + pen_cost = 1.1286 := by
  sorry

end NUMINAMATH_CALUDE_pencil_and_pen_cost_l913_91387


namespace NUMINAMATH_CALUDE_min_xy_value_l913_91378

theorem min_xy_value (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 3*x*y - x - y - 1 = 0) :
  ∀ z, z = x*y → z ≥ 1 :=
by sorry

end NUMINAMATH_CALUDE_min_xy_value_l913_91378


namespace NUMINAMATH_CALUDE_power_expression_l913_91329

theorem power_expression (m n : ℕ+) (a b : ℝ) 
  (h1 : 9^(m : ℕ) = a) 
  (h2 : 3^(n : ℕ) = b) : 
  3^((2*m + 4*n) : ℕ) = a * b^4 := by
  sorry

end NUMINAMATH_CALUDE_power_expression_l913_91329


namespace NUMINAMATH_CALUDE_area_ratio_is_three_fiftieths_l913_91303

/-- A large square subdivided into 25 equal smaller squares -/
structure LargeSquare :=
  (side_length : ℝ)
  (num_subdivisions : ℕ)
  (h_subdivisions : num_subdivisions = 25)

/-- A shaded region formed by connecting midpoints of sides of five smaller squares -/
structure ShadedRegion :=
  (large_square : LargeSquare)
  (num_squares : ℕ)
  (h_num_squares : num_squares = 5)

/-- The ratio of the area of the shaded region to the area of the large square -/
def area_ratio (sr : ShadedRegion) : ℚ :=
  3 / 50

/-- Theorem stating that the area ratio is 3/50 -/
theorem area_ratio_is_three_fiftieths (sr : ShadedRegion) :
  area_ratio sr = 3 / 50 := by
  sorry

end NUMINAMATH_CALUDE_area_ratio_is_three_fiftieths_l913_91303


namespace NUMINAMATH_CALUDE_range_of_shifted_and_translated_function_l913_91304

/-- Given a function f: ℝ → ℝ with range [1,2], 
    prove that the range of g(x) = f(x+1)-2 is [-1,0] -/
theorem range_of_shifted_and_translated_function 
  (f : ℝ → ℝ) (h : Set.range f = Set.Icc 1 2) :
  Set.range (fun x ↦ f (x + 1) - 2) = Set.Icc (-1) 0 := by
  sorry

end NUMINAMATH_CALUDE_range_of_shifted_and_translated_function_l913_91304


namespace NUMINAMATH_CALUDE_samantha_birth_year_l913_91347

def mathLeagueYear (n : ℕ) : ℕ := 1995 + 2 * (n - 1)

theorem samantha_birth_year :
  (∀ n : ℕ, mathLeagueYear n = 1995 + 2 * (n - 1)) →
  mathLeagueYear 5 - 13 = 1990 :=
by sorry

end NUMINAMATH_CALUDE_samantha_birth_year_l913_91347


namespace NUMINAMATH_CALUDE_target_equals_fraction_l913_91309

/-- The decimal representation of a rational number -/
def decimal_rep (q : ℚ) : ℕ → ℕ := sorry

/-- A function that checks if a decimal representation is repeating -/
def is_repeating (d : ℕ → ℕ) : Prop := sorry

/-- The rational number represented by 0.2̄34 -/
def target : ℚ := sorry

theorem target_equals_fraction : 
  (is_repeating (decimal_rep target)) → 
  (∀ a b : ℤ, (a / b : ℚ) = target → ∃ k : ℤ, k * 116 = a ∧ k * 495 = b) →
  target = 116 / 495 := by sorry

end NUMINAMATH_CALUDE_target_equals_fraction_l913_91309


namespace NUMINAMATH_CALUDE_quirkyville_reading_paradox_l913_91397

/-- Represents the student population at Quirkyville College -/
structure StudentPopulation where
  total : ℕ
  enjoy_reading : ℕ
  claim_enjoy : ℕ
  claim_not_enjoy : ℕ

/-- The fraction of students who say they don't enjoy reading but actually do -/
def fraction_false_negative (pop : StudentPopulation) : ℚ :=
  (pop.enjoy_reading - pop.claim_enjoy) / pop.claim_not_enjoy

/-- Theorem stating the fraction of students who say they don't enjoy reading but actually do -/
theorem quirkyville_reading_paradox (pop : StudentPopulation) : 
  pop.total > 0 ∧ 
  pop.enjoy_reading = (70 * pop.total) / 100 ∧
  pop.claim_enjoy = (75 * pop.enjoy_reading) / 100 ∧
  pop.claim_not_enjoy = pop.total - pop.claim_enjoy →
  fraction_false_negative pop = 35 / 83 := by
  sorry

#eval (35 : ℚ) / 83

end NUMINAMATH_CALUDE_quirkyville_reading_paradox_l913_91397


namespace NUMINAMATH_CALUDE_unique_five_digit_number_exists_l913_91313

/-- Represents a 5-digit number with different non-zero digits -/
structure FiveDigitNumber where
  a : Nat
  b : Nat
  c : Nat
  d : Nat
  e : Nat
  a_nonzero : a ≠ 0
  b_nonzero : b ≠ 0
  c_nonzero : c ≠ 0
  d_nonzero : d ≠ 0
  e_nonzero : e ≠ 0
  all_different : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ c ≠ d ∧ c ≠ e ∧ d ≠ e

/-- Checks if the sum of the shifted additions equals a 7-digit number with all digits A -/
def isValidSum (n : FiveDigitNumber) : Prop :=
  let sum := n.a * 1000000 + n.b * 100000 + n.c * 10000 + n.d * 1000 + n.e * 100 + n.d * 10 + n.b +
             n.b * 100000 + n.c * 10000 + n.d * 1000 + n.e * 100 + n.d * 10 + n.b +
             n.c * 10000 + n.d * 1000 + n.e * 100 + n.d * 10 + n.b +
             n.d * 1000 + n.e * 100 + n.d * 10 + n.b +
             n.e * 100 + n.d * 10 + n.b +
             n.d * 10 + n.b +
             n.b
  sum = n.a * 1111111

theorem unique_five_digit_number_exists : ∃! n : FiveDigitNumber, isValidSum n ∧ n.a = 8 ∧ n.b = 4 ∧ n.c = 2 ∧ n.d = 6 ∧ n.e = 9 := by
  sorry

end NUMINAMATH_CALUDE_unique_five_digit_number_exists_l913_91313


namespace NUMINAMATH_CALUDE_digit_B_value_l913_91398

theorem digit_B_value (A B : ℕ) (h : 100 * A + 10 * B + 2 - 41 = 591) : B = 3 := by
  sorry

end NUMINAMATH_CALUDE_digit_B_value_l913_91398


namespace NUMINAMATH_CALUDE_sum_of_factors_24_l913_91337

def factors (n : ℕ) : Finset ℕ :=
  Finset.filter (λ x => n % x = 0) (Finset.range (n + 1))

theorem sum_of_factors_24 :
  (factors 24).sum id = 60 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_factors_24_l913_91337


namespace NUMINAMATH_CALUDE_lamp_marked_price_l913_91385

/-- The marked price of a lamp given initial price, purchase discount, desired gain, and sales discount -/
def marked_price (initial_price : ℚ) (purchase_discount : ℚ) (desired_gain : ℚ) (sales_discount : ℚ) : ℚ :=
  let cost_price := initial_price * (1 - purchase_discount)
  let selling_price := cost_price * (1 + desired_gain)
  selling_price / (1 - sales_discount)

theorem lamp_marked_price :
  marked_price 40 (1/5) (1/4) (3/20) = 800/17 := by
  sorry

end NUMINAMATH_CALUDE_lamp_marked_price_l913_91385


namespace NUMINAMATH_CALUDE_intersection_M_N_l913_91320

-- Define set M
def M : Set ℝ := {y | ∃ x : ℝ, y = x^2 - 1}

-- Define set N
def N : Set ℝ := {x | ∃ y : ℝ, y = Real.sqrt (4 - x^2)}

-- Theorem statement
theorem intersection_M_N : M ∩ N = Set.Icc (-1) 2 := by
  sorry

end NUMINAMATH_CALUDE_intersection_M_N_l913_91320


namespace NUMINAMATH_CALUDE_prime_triplet_with_perfect_square_sum_l913_91394

theorem prime_triplet_with_perfect_square_sum (p₁ p₂ p₃ : ℕ) : 
  Prime p₁ → Prime p₂ → Prime p₃ → 
  p₂ ≠ p₃ → 
  ∃ x y : ℕ, x^2 = 4 + p₁ * p₂ ∧ y^2 = 4 + p₁ * p₃ → 
  ((p₁ = 7 ∧ p₂ = 11 ∧ p₃ = 3) ∨ (p₁ = 7 ∧ p₂ = 3 ∧ p₃ = 11)) := by
sorry

end NUMINAMATH_CALUDE_prime_triplet_with_perfect_square_sum_l913_91394


namespace NUMINAMATH_CALUDE_jan_height_is_42_l913_91306

def cary_height : ℕ := 72

def bill_height : ℕ := cary_height / 2

def jan_height : ℕ := bill_height + 6

theorem jan_height_is_42 : jan_height = 42 := by
  sorry

end NUMINAMATH_CALUDE_jan_height_is_42_l913_91306


namespace NUMINAMATH_CALUDE_ceiling_sum_of_roots_l913_91368

theorem ceiling_sum_of_roots : ⌈Real.sqrt 3⌉ + ⌈Real.sqrt 27⌉ + ⌈Real.sqrt 243⌉ = 24 := by
  sorry

end NUMINAMATH_CALUDE_ceiling_sum_of_roots_l913_91368


namespace NUMINAMATH_CALUDE_cookies_for_guests_l913_91340

/-- Given a total number of cookies and cookies per guest, calculate the number of guests --/
def calculate_guests (total_cookies : ℕ) (cookies_per_guest : ℕ) : ℕ :=
  total_cookies / cookies_per_guest

/-- Theorem: Given 10 total cookies and 2 cookies per guest, the number of guests is 5 --/
theorem cookies_for_guests : calculate_guests 10 2 = 5 := by
  sorry

end NUMINAMATH_CALUDE_cookies_for_guests_l913_91340


namespace NUMINAMATH_CALUDE_wall_bricks_proof_l913_91334

/-- Represents the number of bricks in the wall -/
def wall_bricks : ℕ := 127

/-- Bea's time to build the wall alone in hours -/
def bea_time : ℚ := 8

/-- Ben's time to build the wall alone in hours -/
def ben_time : ℚ := 12

/-- Bea's break time in minutes per hour -/
def bea_break : ℚ := 10

/-- Ben's break time in minutes per hour -/
def ben_break : ℚ := 15

/-- Decrease in output when working together in bricks per hour -/
def output_decrease : ℕ := 12

/-- Time taken to complete the wall when working together in hours -/
def combined_time : ℚ := 6

/-- Bea's effective working time per hour in minutes -/
def bea_effective_time : ℚ := 60 - bea_break

/-- Ben's effective working time per hour in minutes -/
def ben_effective_time : ℚ := 60 - ben_break

theorem wall_bricks_proof :
  let bea_rate : ℚ := wall_bricks / (bea_time * bea_effective_time / 60)
  let ben_rate : ℚ := wall_bricks / (ben_time * ben_effective_time / 60)
  let combined_rate : ℚ := bea_rate + ben_rate - output_decrease
  combined_rate * combined_time = wall_bricks :=
by sorry

end NUMINAMATH_CALUDE_wall_bricks_proof_l913_91334
