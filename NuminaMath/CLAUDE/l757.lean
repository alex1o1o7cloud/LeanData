import Mathlib

namespace sanchez_problem_l757_75767

theorem sanchez_problem (x y : ℕ+) 
  (h1 : x.val - y.val = 2)
  (h2 : x.val * y.val = 120) :
  x.val + y.val = 22 := by
  sorry

end sanchez_problem_l757_75767


namespace sum_reciprocals_l757_75711

theorem sum_reciprocals (a b c d : ℝ) (ω : ℂ) 
  (h1 : ω^4 = 1)
  (h2 : ω ≠ 1)
  (h3 : 1 / (a + ω) + 1 / (b + ω) + 1 / (c + ω) + 1 / (d + ω) = 4 / ω^2) :
  1 / (a + 1) + 1 / (b + 1) + 1 / (c + 1) + 1 / (d + 1) = 2.4 := by
  sorry

end sum_reciprocals_l757_75711


namespace smallest_n_for_unique_k_l757_75776

theorem smallest_n_for_unique_k : ∃ (k : ℤ), (9 : ℚ)/17 < (3 : ℚ)/(3 + k) ∧ (3 : ℚ)/(3 + k) < 8/15 ∧
  ∀ (n : ℕ), n < 3 → ¬(∃! (k : ℤ), (9 : ℚ)/17 < (n : ℚ)/(n + k) ∧ (n : ℚ)/(n + k) < 8/15) :=
by sorry

end smallest_n_for_unique_k_l757_75776


namespace sqrt_221_range_l757_75709

theorem sqrt_221_range : 14 < Real.sqrt 221 ∧ Real.sqrt 221 < 15 := by
  sorry

end sqrt_221_range_l757_75709


namespace cuboid_from_rectangular_projections_l757_75714

/-- Represents a solid object in 3D space -/
structure Solid :=
  (shape : Type)

/-- Represents an orthographic projection (view) of a solid -/
inductive Projection
  | Rectangle
  | Other

/-- Defines the front view of a solid -/
def front_view (s : Solid) : Projection := sorry

/-- Defines the top view of a solid -/
def top_view (s : Solid) : Projection := sorry

/-- Defines the side view of a solid -/
def side_view (s : Solid) : Projection := sorry

/-- Defines a cuboid -/
def is_cuboid (s : Solid) : Prop := sorry

/-- Theorem: If all three orthographic projections of a solid are rectangles, then the solid is a cuboid -/
theorem cuboid_from_rectangular_projections (s : Solid) :
  front_view s = Projection.Rectangle →
  top_view s = Projection.Rectangle →
  side_view s = Projection.Rectangle →
  is_cuboid s :=
sorry

end cuboid_from_rectangular_projections_l757_75714


namespace y_derivative_at_zero_l757_75797

noncomputable def y (x : ℝ) : ℝ := Real.exp (Real.sin x) * Real.cos (Real.sin x)

theorem y_derivative_at_zero : 
  deriv y 0 = 1 := by sorry

end y_derivative_at_zero_l757_75797


namespace foma_cannot_guarantee_win_l757_75781

/-- Represents a player in the coin game -/
inductive Player : Type
| Foma : Player
| Yerema : Player

/-- Represents the state of the game -/
structure GameState :=
(coins : List Nat)  -- List of remaining coin values
(foma_coins : Nat)  -- Total value of Foma's coins
(yerema_coins : Nat)  -- Total value of Yerema's coins
(last_selector : Player)  -- Player who made the last selection

/-- Function to determine the next selector based on current game state -/
def next_selector (state : GameState) : Player :=
  if state.foma_coins > state.yerema_coins then Player.Foma
  else if state.yerema_coins > state.foma_coins then Player.Yerema
  else state.last_selector

/-- Theorem stating that Foma cannot guarantee winning -/
theorem foma_cannot_guarantee_win :
  ∀ (initial_state : GameState),
    initial_state.coins = List.range 25
    → initial_state.foma_coins = 0
    → initial_state.yerema_coins = 0
    → initial_state.last_selector = Player.Foma
    → ¬ (∀ (strategy : GameState → Nat),
         ∃ (final_state : GameState),
           final_state.coins = []
           ∧ final_state.foma_coins > final_state.yerema_coins) :=
sorry

end foma_cannot_guarantee_win_l757_75781


namespace find_a_l757_75774

def f (a : ℝ) (x : ℝ) : ℝ := a * x^3 + 3 * x^2 + 2

theorem find_a : ∃ a : ℝ, (∀ x : ℝ, (deriv (f a)) x = 3 * a * x^2 + 6 * x) ∧ (deriv (f a)) (-1) = 3 → a = 3 := by
  sorry

end find_a_l757_75774


namespace factorial_ratio_2017_2016_l757_75713

-- Define factorial operation
def factorial : Nat → Nat
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

-- Theorem statement
theorem factorial_ratio_2017_2016 :
  factorial 2017 / factorial 2016 = 2017 := by
  sorry

end factorial_ratio_2017_2016_l757_75713


namespace sliding_ladder_inequality_l757_75735

/-- Represents a sliding ladder against a wall -/
structure SlidingLadder where
  length : ℝ
  topSlideDistance : ℝ
  bottomSlipDistance : ℝ

/-- The bottom slip distance is always greater than the top slide distance for a sliding ladder -/
theorem sliding_ladder_inequality (ladder : SlidingLadder) :
  ladder.bottomSlipDistance > ladder.topSlideDistance :=
sorry

end sliding_ladder_inequality_l757_75735


namespace rocket_fuel_ratio_l757_75780

theorem rocket_fuel_ratio (m M : ℝ) (h : m > 0) :
  2000 * Real.log (1 + M / m) = 12000 → M / m = Real.exp 6 - 1 := by
  sorry

end rocket_fuel_ratio_l757_75780


namespace root_existence_condition_l757_75791

def f (m : ℝ) (x : ℝ) : ℝ := m * x + 6

theorem root_existence_condition (m : ℝ) :
  (∃ x ∈ Set.Icc (-2 : ℝ) 3, f m x = 0) ↔ m ≤ -2 ∨ m ≥ 3 := by
  sorry

end root_existence_condition_l757_75791


namespace sum_of_squares_quadratic_roots_l757_75702

theorem sum_of_squares_quadratic_roots : 
  let a : ℝ := 1
  let b : ℝ := -15
  let c : ℝ := 6
  let r₁ : ℝ := (-b + Real.sqrt (b^2 - 4*a*c)) / (2*a)
  let r₂ : ℝ := (-b - Real.sqrt (b^2 - 4*a*c)) / (2*a)
  r₁^2 + r₂^2 = 213 :=
by sorry

end sum_of_squares_quadratic_roots_l757_75702


namespace geometric_series_equality_l757_75777

theorem geometric_series_equality (n : ℕ) : n ≥ 1 → (
  let C : ℕ → ℝ := λ k => 512 * (1 - (1 / 2^k))
  let D : ℕ → ℝ := λ k => (2048 / 3) * (1 - (-1 / 2)^k)
  (∀ k < n, C k ≠ D k) ∧ C n = D n → n = 4
) := by sorry

end geometric_series_equality_l757_75777


namespace solution_set_when_a_eq_two_range_of_a_for_inequality_l757_75716

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x - 3| - |x + a|

-- Theorem for part I
theorem solution_set_when_a_eq_two :
  {x : ℝ | f 2 x < 1} = {x : ℝ | x > 0} := by sorry

-- Theorem for part II
theorem range_of_a_for_inequality :
  {a : ℝ | ∀ x, f a x ≤ 2*a} = {a : ℝ | a ≥ 3} := by sorry

end solution_set_when_a_eq_two_range_of_a_for_inequality_l757_75716


namespace complete_square_l757_75764

theorem complete_square (x : ℝ) : (x^2 - 5*x = 31) → ((x - 5/2)^2 = 149/4) := by
  intro h
  sorry

end complete_square_l757_75764


namespace savings_after_expense_increase_l757_75786

def monthly_savings (salary : ℝ) (initial_savings_rate : ℝ) (expense_increase_rate : ℝ) : ℝ :=
  let initial_savings := salary * initial_savings_rate
  let initial_expenses := salary - initial_savings
  let new_expenses := initial_expenses * (1 + expense_increase_rate)
  salary - new_expenses

theorem savings_after_expense_increase :
  monthly_savings 1000 0.25 0.1 = 175 := by sorry

end savings_after_expense_increase_l757_75786


namespace jack_marbles_shared_l757_75739

/-- Calculates the number of marbles shared given initial and remaining marbles -/
def marblesShared (initial remaining : ℕ) : ℕ := initial - remaining

/-- Proves that the number of marbles shared is correct for Jack's scenario -/
theorem jack_marbles_shared :
  marblesShared 62 29 = 33 := by
  sorry

end jack_marbles_shared_l757_75739


namespace polynomial_on_unit_circle_l757_75778

/-- A polynomial p(z) with complex coefficients a and b -/
def p (a b z : ℂ) : ℂ := z^2 + a*z + b

/-- The property that |p(z)| = 1 on the unit circle -/
def unit_circle_property (a b : ℂ) : Prop :=
  ∀ z : ℂ, Complex.abs z = 1 → Complex.abs (p a b z) = 1

/-- Theorem: If |p(z)| = 1 on the unit circle, then a = 0 and b = 0 -/
theorem polynomial_on_unit_circle (a b : ℂ) 
  (h : unit_circle_property a b) : a = 0 ∧ b = 0 := by
  sorry

end polynomial_on_unit_circle_l757_75778


namespace hardware_store_earnings_l757_75792

/-- Calculates the total earnings of a hardware store for a week given the sales and prices of various items. -/
theorem hardware_store_earnings 
  (graphics_cards_sold : ℕ) (graphics_card_price : ℕ)
  (hard_drives_sold : ℕ) (hard_drive_price : ℕ)
  (cpus_sold : ℕ) (cpu_price : ℕ)
  (ram_pairs_sold : ℕ) (ram_pair_price : ℕ)
  (h1 : graphics_cards_sold = 10)
  (h2 : graphics_card_price = 600)
  (h3 : hard_drives_sold = 14)
  (h4 : hard_drive_price = 80)
  (h5 : cpus_sold = 8)
  (h6 : cpu_price = 200)
  (h7 : ram_pairs_sold = 4)
  (h8 : ram_pair_price = 60) :
  graphics_cards_sold * graphics_card_price +
  hard_drives_sold * hard_drive_price +
  cpus_sold * cpu_price +
  ram_pairs_sold * ram_pair_price = 8960 := by
  sorry

end hardware_store_earnings_l757_75792


namespace gcd_lcm_sum_63_2898_l757_75798

theorem gcd_lcm_sum_63_2898 : Nat.gcd 63 2898 + Nat.lcm 63 2898 = 182575 := by
  sorry

end gcd_lcm_sum_63_2898_l757_75798


namespace pqr_sum_bounds_l757_75734

theorem pqr_sum_bounds (p q r : ℝ) (h : 5 * (p + q + r) = p^2 + q^2 + r^2) :
  let R := p*q + p*r + q*r
  ∃ (N n : ℝ),
    (∀ x y z : ℝ, 5 * (x + y + z) = x^2 + y^2 + z^2 → x*y + x*z + y*z ≤ N) ∧
    (∀ x y z : ℝ, 5 * (x + y + z) = x^2 + y^2 + z^2 → n ≤ x*y + x*z + y*z) ∧
    N = 150 ∧
    n = -12.5 ∧
    N + 15*n = -37.5 :=
by sorry

end pqr_sum_bounds_l757_75734


namespace proper_subset_condition_l757_75784

def A (a : ℝ) : Set ℝ := {1, 4, a}
def B (a : ℝ) : Set ℝ := {1, a^2}

def valid_a : Set ℝ := {-2, -1, 0, 1, 2}

theorem proper_subset_condition (a : ℝ) : 
  (B a ⊂ A a) ↔ a ∈ valid_a := by sorry

end proper_subset_condition_l757_75784


namespace quadratic_equation_properties_l757_75727

/-- The quadratic equation x^2 - 4mx + 3m^2 = 0 -/
def quadratic_equation (x m : ℝ) : Prop :=
  x^2 - 4*m*x + 3*m^2 = 0

theorem quadratic_equation_properties :
  ∀ m : ℝ,
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ quadratic_equation x1 m ∧ quadratic_equation x2 m) ∧
  (∀ x1 x2 : ℝ, x1 > x2 → quadratic_equation x1 m → quadratic_equation x2 m → x1 - x2 = 2 → m = 1) :=
by sorry

end quadratic_equation_properties_l757_75727


namespace ian_says_smallest_unclaimed_number_l757_75772

/-- Represents a student in the counting game -/
structure Student where
  name : String
  index : Nat

/-- The list of students in alphabetical order -/
def students : List Student := [
  ⟨"Alice", 0⟩, ⟨"Barbara", 1⟩, ⟨"Candice", 2⟩, ⟨"Debbie", 3⟩,
  ⟨"Eliza", 4⟩, ⟨"Fatima", 5⟩, ⟨"Greg", 6⟩, ⟨"Helen", 7⟩
]

/-- The maximum number in the counting sequence -/
def maxNumber : Nat := 1200

/-- Determines if a student says a given number -/
def saysNumber (s : Student) (n : Nat) : Prop :=
  n ≤ maxNumber ∧ n % (4 * 4^s.index) ≠ 0

/-- The number that Ian says -/
def iansNumber : Nat := 1021

/-- Theorem stating that Ian's number is the smallest not said by any other student -/
theorem ian_says_smallest_unclaimed_number :
  (∀ n < iansNumber, ∃ s ∈ students, saysNumber s n) ∧
  (∀ s ∈ students, ¬saysNumber s iansNumber) :=
sorry

end ian_says_smallest_unclaimed_number_l757_75772


namespace max_product_f_value_l757_75738

-- Define the function f
def f (a b x : ℝ) : ℝ := 2 * a * x + b

-- State the theorem
theorem max_product_f_value :
  ∀ a b : ℝ,
    a > 0 →
    b > 0 →
    (∀ x : ℝ, x ∈ Set.Icc (-1/2) (1/2) → |f a b x| ≤ 2) →
    (∀ a' b' : ℝ, a' > 0 → b' > 0 → 
      (∀ x : ℝ, x ∈ Set.Icc (-1/2) (1/2) → |f a' b' x| ≤ 2) → 
      a * b ≥ a' * b') →
    f a b 2017 = 4035 :=
by
  sorry


end max_product_f_value_l757_75738


namespace complement_intersection_equals_three_l757_75760

universe u

def U : Finset (Fin 5) := {0, 1, 2, 3, 4}
def M : Finset (Fin 5) := {0, 1, 2}
def N : Finset (Fin 5) := {2, 3}

theorem complement_intersection_equals_three :
  (U \ M) ∩ N = {3} := by sorry

end complement_intersection_equals_three_l757_75760


namespace product_xyz_l757_75730

theorem product_xyz (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (eq1 : x + 1/y = 5)
  (eq2 : y + 1/z = 2)
  (eq3 : z + 1/x = 8/3) :
  x * y * z = (11 + Real.sqrt 117) / 2 := by
  sorry

end product_xyz_l757_75730


namespace winnie_balloon_distribution_l757_75770

/-- Calculates the number of balloons left after equal distribution --/
def balloons_left (red blue green purple friends : ℕ) : ℕ :=
  (red + blue + green + purple) % friends

/-- Proves that Winnie has 0 balloons left after distribution --/
theorem winnie_balloon_distribution :
  balloons_left 22 44 66 88 10 = 0 := by
  sorry

end winnie_balloon_distribution_l757_75770


namespace policeman_speed_l757_75718

/-- Given a chase scenario between a policeman and a thief, this theorem proves
    the speed of the policeman required to catch the thief. -/
theorem policeman_speed (initial_distance : ℝ) (thief_speed : ℝ) (thief_distance : ℝ) :
  initial_distance = 0.15 →
  thief_speed = 8 →
  thief_distance = 0.6 →
  ∃ (policeman_speed : ℝ),
    policeman_speed * (thief_distance / thief_speed) = initial_distance + thief_distance ∧
    policeman_speed = 10 := by
  sorry

end policeman_speed_l757_75718


namespace intersection_complement_equality_l757_75769

-- Define the universe U
def U : Set ℝ := { x | x > -3 }

-- Define set A
def A : Set ℝ := { x | x < -2 ∨ x > 3 }

-- Define set B
def B : Set ℝ := { x | -1 ≤ x ∧ x ≤ 4 }

-- Theorem statement
theorem intersection_complement_equality :
  A ∩ (U \ B) = { x | -3 < x ∧ x < -2 ∨ x > 4 } := by sorry

end intersection_complement_equality_l757_75769


namespace ratio_problem_l757_75736

theorem ratio_problem (A B C : ℝ) (h1 : A + B + C = 98) (h2 : A / B = 2 / 3) (h3 : B = 30) :
  B / C = 5 / 8 := by
  sorry

end ratio_problem_l757_75736


namespace joan_total_cents_l757_75799

/-- The value of a quarter in cents -/
def quarter_value : ℕ := 25

/-- The number of quarters Joan has -/
def joan_quarters : ℕ := 6

/-- Theorem: Joan's total cents -/
theorem joan_total_cents : joan_quarters * quarter_value = 150 := by
  sorry

end joan_total_cents_l757_75799


namespace exists_sum_of_digits_div_11_l757_75743

/-- Sum of digits function in base 10 -/
def sumOfDigits (n : ℕ) : ℕ := sorry

/-- Theorem: Among 39 consecutive natural numbers, there is always one whose sum of digits (in base 10) is divisible by 11 -/
theorem exists_sum_of_digits_div_11 (n : ℕ) : 
  ∃ k ∈ Finset.range 39, (sumOfDigits (n + k)) % 11 = 0 := by sorry

end exists_sum_of_digits_div_11_l757_75743


namespace min_value_of_fraction_l757_75742

theorem min_value_of_fraction (x : ℝ) (h : x ≥ 3/2) :
  (2*x^2 - 2*x + 1) / (x - 1) ≥ 2*Real.sqrt 2 + 2 := by
  sorry

end min_value_of_fraction_l757_75742


namespace three_digit_cube_divisible_by_8_and_9_l757_75752

theorem three_digit_cube_divisible_by_8_and_9 :
  ∃! n : ℕ, 100 ≤ n^3 ∧ n^3 ≤ 999 ∧ 6 ∣ n ∧ 8 ∣ n^3 ∧ 9 ∣ n^3 :=
by sorry

end three_digit_cube_divisible_by_8_and_9_l757_75752


namespace jessica_initial_money_l757_75710

/-- The amount of money Jessica spent on a cat toy -/
def spent : ℚ := 10.22

/-- The amount of money Jessica has left -/
def left : ℚ := 1.51

/-- Jessica's initial amount of money -/
def initial : ℚ := spent + left

/-- Theorem stating that Jessica's initial amount of money was $11.73 -/
theorem jessica_initial_money : initial = 11.73 := by
  sorry

end jessica_initial_money_l757_75710


namespace projection_is_regular_polygon_l757_75785

-- Define the types of polyhedra
inductive Polyhedron
  | Dodecahedron
  | Icosahedron

-- Define a regular polygon
structure RegularPolygon where
  sides : ℕ
  is_regular : Bool

-- Define a projection function
def project (p : Polyhedron) : RegularPolygon :=
  match p with
  | Polyhedron.Dodecahedron => { sides := 10, is_regular := true }
  | Polyhedron.Icosahedron => { sides := 6, is_regular := true }

-- Theorem statement
theorem projection_is_regular_polygon (p : Polyhedron) :
  (project p).is_regular = true :=
by sorry

end projection_is_regular_polygon_l757_75785


namespace house_pets_problem_l757_75779

theorem house_pets_problem (total : Nat) (dogs cats turtles : Nat)
  (h_total : total = 2017)
  (h_dogs : dogs = 1820)
  (h_cats : cats = 1651)
  (h_turtles : turtles = 1182)
  (h_dogs_le : dogs ≤ total)
  (h_cats_le : cats ≤ total)
  (h_turtles_le : turtles ≤ total) :
  ∃ (max min : Nat),
    (max ≤ turtles) ∧
    (min ≥ dogs + cats + turtles - 2 * total) ∧
    (max - min = 563) := by
  sorry

end house_pets_problem_l757_75779


namespace exists_functions_with_even_product_l757_75726

-- Define the type for real-valued functions
def RealFunction := ℝ → ℝ

-- Define what it means for a function to be even
def IsEven (f : RealFunction) : Prop :=
  ∀ x : ℝ, f (-x) = f x

-- Define what it means for a function to be odd
def IsOdd (f : RealFunction) : Prop :=
  ∀ x : ℝ, f (-x) = -f x

-- Define what it means for a function to be neither odd nor even
def NeitherOddNorEven (f : RealFunction) : Prop :=
  ¬(IsEven f) ∧ ¬(IsOdd f)

-- State the theorem
theorem exists_functions_with_even_product :
  ∃ (f g : RealFunction),
    NeitherOddNorEven f ∧
    NeitherOddNorEven g ∧
    IsEven (fun x ↦ f x * g x) :=
by sorry

end exists_functions_with_even_product_l757_75726


namespace problem_statement_l757_75722

theorem problem_statement (a b : ℝ) (ha : a > 0) (heq : Real.exp a * (1 - Real.log b) = 1) :
  (1 < b ∧ b < Real.exp 1) ∧ (a > Real.log b) ∧ (b - a < 1) := by
  sorry

end problem_statement_l757_75722


namespace largest_divided_by_smallest_l757_75731

def numbers : List ℝ := [10, 11, 12]

theorem largest_divided_by_smallest : 
  (List.maximum numbers).get! / (List.minimum numbers).get! = 1.2 := by
  sorry

end largest_divided_by_smallest_l757_75731


namespace tangent_line_intercept_l757_75744

open Real

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * exp x - 3 * x + 1

theorem tangent_line_intercept (a : ℝ) :
  let f' := fun x => a * exp x - 3
  (f' 0 = 1) →
  (f a 0 = 5) →
  (∃ b, ∀ x, f a 0 + f' 0 * x = x + b) →
  ∃ b, f a 0 + f' 0 * 0 = 0 + b ∧ b = 5 := by
sorry

end tangent_line_intercept_l757_75744


namespace complex_equation_sum_l757_75756

theorem complex_equation_sum (a b : ℝ) :
  (2 : ℂ) - 2 * Complex.I^3 = a + b * Complex.I → a + b = 4 := by
  sorry

end complex_equation_sum_l757_75756


namespace quadratic_value_relation_l757_75732

theorem quadratic_value_relation (x : ℝ) (h : x^2 + x + 1 = 8) : 4*x^2 + 4*x + 9 = 37 := by
  sorry

end quadratic_value_relation_l757_75732


namespace number_of_houses_street_houses_l757_75748

/-- Given a street with clotheslines, prove the number of houses -/
theorem number_of_houses (children : ℕ) (adults : ℕ) (child_items : ℕ) (adult_items : ℕ) 
  (items_per_line : ℕ) (lines_per_house : ℕ) : ℕ :=
  let total_items := children * child_items + adults * adult_items
  let total_lines := total_items / items_per_line
  total_lines / lines_per_house

/-- Prove that there are 26 houses on the street -/
theorem street_houses : 
  number_of_houses 11 20 4 3 2 2 = 26 := by
  sorry

end number_of_houses_street_houses_l757_75748


namespace quadratic_solution_difference_l757_75788

theorem quadratic_solution_difference : ∃ (x₁ x₂ : ℝ),
  (x₁^2 - 5*x₁ + 15 = x₁ + 55) ∧
  (x₂^2 - 5*x₂ + 15 = x₂ + 55) ∧
  x₁ ≠ x₂ ∧
  |x₁ - x₂| = 14 := by
sorry

end quadratic_solution_difference_l757_75788


namespace car_banker_speed_ratio_l757_75796

/-- The ratio of car speed to banker speed given specific timing conditions -/
theorem car_banker_speed_ratio :
  ∀ (T : ℝ) (Vb Vc : ℝ) (d : ℝ),
    Vb > 0 →
    Vc > 0 →
    d > 0 →
    (Vb * 60 = Vc * 5) →
    (Vc / Vb = 12) := by
  sorry

end car_banker_speed_ratio_l757_75796


namespace joao_chocolate_bars_l757_75704

theorem joao_chocolate_bars (x y z : ℕ) : 
  x + y + z = 30 →
  2 * x + 3 * y + 4 * z = 100 →
  z > x :=
by sorry

end joao_chocolate_bars_l757_75704


namespace machine_production_time_l757_75759

/-- Proves that a machine producing 360 items in 2 hours takes 1/3 minute to produce one item. -/
theorem machine_production_time 
  (items_produced : ℕ) 
  (production_hours : ℕ) 
  (minutes_per_hour : ℕ) 
  (h1 : items_produced = 360)
  (h2 : production_hours = 2)
  (h3 : minutes_per_hour = 60) :
  (production_hours * minutes_per_hour) / items_produced = 1 / 3 := by
  sorry

#check machine_production_time

end machine_production_time_l757_75759


namespace lawrence_county_kids_count_l757_75762

/-- The number of kids in Lawrence county who stayed home -/
def kids_stayed_home : ℕ := 644997

/-- The number of kids in Lawrence county who went to camp -/
def kids_went_to_camp : ℕ := 893835

/-- The number of kids from outside the county who attended the camp -/
def outside_kids_in_camp : ℕ := 78

/-- The total number of kids in Lawrence county -/
def total_kids_in_county : ℕ := kids_stayed_home + kids_went_to_camp

theorem lawrence_county_kids_count : total_kids_in_county = 1538832 := by
  sorry

end lawrence_county_kids_count_l757_75762


namespace max_value_problem_l757_75729

theorem max_value_problem (x y z : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) (hz : z ≥ 0) (hsum : x + y + z = 3) :
  (x^2 - x*y + y^2) * (x^2 - x*z + z^2) * (y^2 - y*z + z^2) * (z^2 - z*x + x^2) ≤ 1 := by
  sorry

end max_value_problem_l757_75729


namespace quadrilateral_sides_diagonals_inequality_l757_75717

/-- Theorem: For any quadrilateral, the sum of the squares of its sides is not less than
    the sum of the squares of its diagonals. -/
theorem quadrilateral_sides_diagonals_inequality 
  (x₁ y₁ x₂ y₂ x₃ y₃ x₄ y₄ : ℝ) : 
  (x₂ - x₁)^2 + (y₂ - y₁)^2 + (x₃ - x₂)^2 + (y₃ - y₂)^2 + 
  (x₄ - x₃)^2 + (y₄ - y₃)^2 + (x₄ - x₁)^2 + (y₄ - y₁)^2 ≥ 
  (x₃ - x₁)^2 + (y₃ - y₁)^2 + (x₄ - x₂)^2 + (y₄ - y₂)^2 :=
by sorry

end quadrilateral_sides_diagonals_inequality_l757_75717


namespace sphere_volume_equals_surface_area_l757_75787

theorem sphere_volume_equals_surface_area (r : ℝ) (h : r > 0) :
  (4 / 3 * Real.pi * r^3) = (4 * Real.pi * r^2) → r = 3 := by
  sorry

end sphere_volume_equals_surface_area_l757_75787


namespace complex_fraction_power_simplification_l757_75746

theorem complex_fraction_power_simplification :
  (((3 : ℂ) + 4*I) / ((3 : ℂ) - 4*I))^8 = 1 := by
  sorry

end complex_fraction_power_simplification_l757_75746


namespace exponential_monotonicity_l757_75753

theorem exponential_monotonicity (a b c : ℝ) 
  (h1 : a > b) (h2 : b > 0) (h3 : c > 1) : c^a > c^b := by
  sorry

end exponential_monotonicity_l757_75753


namespace solve_for_c_l757_75733

theorem solve_for_c (c d : ℚ) 
  (eq1 : (c - 34) / 2 = (2 * d - 8) / 7)
  (eq2 : d = c + 9) : 
  c = 86 := by
sorry

end solve_for_c_l757_75733


namespace jose_investment_is_45000_l757_75705

/-- Represents the investment and profit scenario of Tom and Jose --/
structure InvestmentScenario where
  tom_investment : ℕ
  jose_join_delay : ℕ
  total_profit : ℕ
  jose_profit : ℕ

/-- Calculates Jose's investment amount based on the given scenario --/
def calculate_jose_investment (scenario : InvestmentScenario) : ℕ :=
  -- Implementation details are omitted
  sorry

/-- Theorem stating that Jose's investment is 45000 given the specific scenario --/
theorem jose_investment_is_45000 :
  let scenario : InvestmentScenario := {
    tom_investment := 30000,
    jose_join_delay := 2,
    total_profit := 72000,
    jose_profit := 40000
  }
  calculate_jose_investment scenario = 45000 := by
  sorry

end jose_investment_is_45000_l757_75705


namespace sqrt_equation_solution_l757_75790

theorem sqrt_equation_solution (x : ℝ) : 
  Real.sqrt (3 + Real.sqrt (2 * x - 1)) = 4 → x = 85 := by
sorry

end sqrt_equation_solution_l757_75790


namespace square_area_from_adjacent_points_l757_75758

/-- The area of a square with adjacent points (1,2) and (4,6) on a Cartesian coordinate plane is 25. -/
theorem square_area_from_adjacent_points : 
  let p1 : ℝ × ℝ := (1, 2)
  let p2 : ℝ × ℝ := (4, 6)
  let side_length := Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)
  side_length^2 = 25 := by sorry

end square_area_from_adjacent_points_l757_75758


namespace paperboy_delivery_ways_l757_75793

/-- Represents the number of ways to deliver newspapers to n houses without missing four consecutive houses. -/
def E : ℕ → ℕ
  | 0 => 0  -- Define E(0) as 0 for completeness
  | 1 => 2
  | 2 => 4
  | 3 => 8
  | 4 => 15
  | n + 5 => E (n + 4) + E (n + 3) + E (n + 2) + E (n + 1)

/-- Theorem stating that there are 2872 ways for a paperboy to deliver newspapers to 12 houses without missing four consecutive houses. -/
theorem paperboy_delivery_ways : E 12 = 2872 := by
  sorry

end paperboy_delivery_ways_l757_75793


namespace computer_program_output_l757_75737

theorem computer_program_output (x : ℝ) (y : ℝ) : 
  x = Real.sqrt 3 - 2 → y = Real.sqrt ((x^2).sqrt - 2) → y = -Real.sqrt 3 := by sorry

end computer_program_output_l757_75737


namespace porter_buns_problem_l757_75712

/-- The maximum number of buns that can be transported given the conditions -/
def max_buns_transported (total_buns : ℕ) (capacity : ℕ) (eaten_per_trip : ℕ) : ℕ :=
  total_buns - (2 * (total_buns / capacity) - 1) * eaten_per_trip

/-- Theorem stating that given the specific conditions, the maximum number of buns transported is 191 -/
theorem porter_buns_problem :
  max_buns_transported 200 40 1 = 191 := by
  sorry

end porter_buns_problem_l757_75712


namespace range_of_a_l757_75789

-- Define propositions p and q as functions of a
def p (a : ℝ) : Prop := ∀ x : ℝ, a * x^2 + a * x + 1 > 0

def q (a : ℝ) : Prop := ∃ x : ℝ, x^2 - x + a = 0

-- Define the theorem
theorem range_of_a :
  (∀ a : ℝ, (p a ∨ q a) ∧ ¬(p a ∧ q a)) →
  (∀ a : ℝ, (a < 0 ∨ (1/4 < a ∧ a < 4)) ↔ (p a ∨ q a) ∧ ¬(p a ∧ q a)) :=
sorry

end range_of_a_l757_75789


namespace simplify_fourth_root_l757_75751

theorem simplify_fourth_root (a b : ℕ+) : 
  (2^6 * 5^5 : ℝ)^(1/4) = a * (b : ℝ)^(1/4) ∧ a + b = 30 := by
  sorry

end simplify_fourth_root_l757_75751


namespace expression_evaluation_l757_75708

theorem expression_evaluation (x : ℝ) : x = 2 → 2 * x^2 - 3 * x + 4 = 6 := by
  sorry

end expression_evaluation_l757_75708


namespace amys_garden_space_l757_75721

/-- Calculates the total square footage of growing space for Amy's garden beds -/
theorem amys_garden_space (small_bed_length small_bed_width : ℝ)
                           (large_bed_length large_bed_width : ℝ)
                           (num_small_beds num_large_beds : ℕ) :
  small_bed_length = 3 →
  small_bed_width = 3 →
  large_bed_length = 4 →
  large_bed_width = 3 →
  num_small_beds = 2 →
  num_large_beds = 2 →
  (num_small_beds : ℝ) * (small_bed_length * small_bed_width) +
  (num_large_beds : ℝ) * (large_bed_length * large_bed_width) = 42 := by
  sorry

#check amys_garden_space

end amys_garden_space_l757_75721


namespace same_gender_officers_l757_75720

theorem same_gender_officers (total_members : Nat) (boys : Nat) (girls : Nat) :
  total_members = 24 →
  boys = 12 →
  girls = 12 →
  boys + girls = total_members →
  (boys * (boys - 1) + girls * (girls - 1) : Nat) = 264 := by
  sorry

end same_gender_officers_l757_75720


namespace geometric_sequence_property_l757_75755

/-- A geometric sequence -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

/-- The product of the 1st, 7th, and 13th terms equals 8 -/
def product_condition (a : ℕ → ℝ) : Prop :=
  a 1 * a 7 * a 13 = 8

theorem geometric_sequence_property (a : ℕ → ℝ) 
  (h1 : geometric_sequence a) 
  (h2 : product_condition a) : 
  a 3 * a 11 = 4 := by
  sorry

end geometric_sequence_property_l757_75755


namespace investment_after_three_years_l757_75773

def compound_interest (initial_investment : ℝ) (interest_rate : ℝ) (additional_investment : ℝ) (years : ℕ) : ℝ :=
  let rec helper (n : ℕ) (current_amount : ℝ) : ℝ :=
    if n = 0 then
      current_amount
    else
      helper (n - 1) ((current_amount * (1 + interest_rate)) + additional_investment)
  helper years initial_investment

theorem investment_after_three_years :
  let initial_investment : ℝ := 500
  let interest_rate : ℝ := 0.02
  let additional_investment : ℝ := 500
  let years : ℕ := 3
  compound_interest initial_investment interest_rate additional_investment years = 2060.80 := by
  sorry

end investment_after_three_years_l757_75773


namespace solution_set_inequality_l757_75725

theorem solution_set_inequality (x : ℝ) : 
  (0 < x ∧ x < 2) ↔ (4 / x > |x| ∧ x ≠ 0) :=
sorry

end solution_set_inequality_l757_75725


namespace lottery_distribution_l757_75771

/-- The total amount received by 100 students, each getting one-thousandth of $155250 -/
def total_amount (lottery_win : ℚ) (num_students : ℕ) : ℚ :=
  (lottery_win / 1000) * num_students

theorem lottery_distribution :
  total_amount 155250 100 = 15525 := by
  sorry

end lottery_distribution_l757_75771


namespace F_3_f_4_equals_7_l757_75749

def f (a : ℝ) : ℝ := a - 2

def F (a b : ℝ) : ℝ := b^2 + a

theorem F_3_f_4_equals_7 : F 3 (f 4) = 7 := by
  sorry

end F_3_f_4_equals_7_l757_75749


namespace sum_reciprocals_lower_bound_l757_75707

theorem sum_reciprocals_lower_bound (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 2) :
  1/x + 1/y ≥ 2 ∧ ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ a + b = 2 ∧ 1/a + 1/b = 2 := by
  sorry

end sum_reciprocals_lower_bound_l757_75707


namespace prob_at_least_one_one_value_l757_75768

/-- The probability of rolling a specific number on a fair six-sided die -/
def prob_single : ℚ := 1 / 6

/-- The probability of not rolling a specific number on a fair six-sided die -/
def prob_not_single : ℚ := 1 - prob_single

/-- The probability of at least one die showing 1 when two fair six-sided dice are rolled once -/
def prob_at_least_one_one : ℚ := 
  prob_single * prob_not_single + 
  prob_not_single * prob_single + 
  prob_single * prob_single

theorem prob_at_least_one_one_value : prob_at_least_one_one = 11 / 36 := by
  sorry

end prob_at_least_one_one_value_l757_75768


namespace equation_solution_l757_75703

theorem equation_solution : ∃ x : ℝ, 300 * x + (12 + 4) * (1 / 8) = 602 ∧ x = 2 := by
  sorry

end equation_solution_l757_75703


namespace rhombus_longer_diagonal_l757_75795

/-- A rhombus with side length 35 units and shorter diagonal 42 units has a longer diagonal of 56 units. -/
theorem rhombus_longer_diagonal (s d_short : ℝ) (h1 : s = 35) (h2 : d_short = 42) :
  let d_long := Real.sqrt (4 * s^2 - d_short^2)
  d_long = 56 := by sorry

end rhombus_longer_diagonal_l757_75795


namespace photo_difference_l757_75741

theorem photo_difference (claire_photos : ℕ) (lisa_photos : ℕ) (robert_photos : ℕ) : 
  claire_photos = 12 →
  lisa_photos = 3 * claire_photos →
  robert_photos = lisa_photos →
  robert_photos - claire_photos = 24 := by sorry

end photo_difference_l757_75741


namespace zero_location_l757_75715

def has_unique_zero (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∃! x, a < x ∧ x < b ∧ f x = 0

theorem zero_location (f : ℝ → ℝ) :
  has_unique_zero f 0 16 ∧
  has_unique_zero f 0 8 ∧
  has_unique_zero f 0 4 ∧
  has_unique_zero f 0 2 →
  ∀ x, 2 ≤ x ∧ x < 16 → f x ≠ 0 :=
by sorry

end zero_location_l757_75715


namespace amoeba_count_after_week_l757_75740

/-- The number of amoebas after n days, given an initial population of 1 and each amoeba splitting into two every day. -/
def amoeba_count (n : ℕ) : ℕ := 2^n

/-- Theorem stating that the number of amoebas after 7 days is 128. -/
theorem amoeba_count_after_week : amoeba_count 7 = 128 := by
  sorry

end amoeba_count_after_week_l757_75740


namespace pirate_treasure_year_l757_75775

/-- Converts a base-8 number to base-10 --/
def base8_to_base10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (8 ^ i)) 0

/-- The base-8 representation of the year --/
def year_base8 : List Nat := [7, 6, 3]

/-- The claimed base-10 equivalent of the year --/
def year_base10 : Nat := 247

/-- Theorem stating that the base-8 year converts to the claimed base-10 year --/
theorem pirate_treasure_year : base8_to_base10 year_base8 = year_base10 := by
  sorry

end pirate_treasure_year_l757_75775


namespace quadratic_no_real_roots_l757_75761

theorem quadratic_no_real_roots :
  ∀ x : ℝ, x^2 + 2*x + 3 ≠ 0 :=
by
  sorry

end quadratic_no_real_roots_l757_75761


namespace exists_winning_strategy_l757_75766

/-- The set of numbers from which the hidden numbers are chosen -/
def S : Set ℕ := Finset.range 250

/-- A strategy is a function that takes the player's number and the history of announcements,
    and returns the next announcement -/
def Strategy := ℕ → List ℕ → ℕ

/-- The game state consists of both players' numbers and the history of announcements -/
structure GameState :=
  (player_a_number : ℕ)
  (player_b_number : ℕ)
  (announcements : List ℕ)

/-- A game is valid if both players' numbers are in S and the sum of announcements is 20 -/
def valid_game (g : GameState) : Prop :=
  g.player_a_number ∈ S ∧ g.player_b_number ∈ S ∧ g.announcements.sum = 20

/-- A strategy is winning if it allows both players to determine each other's number -/
def winning_strategy (strat_a strat_b : Strategy) : Prop :=
  ∀ (g : GameState), valid_game g →
    ∃ (n : ℕ), strat_a g.player_a_number (g.announcements.take n) = g.player_b_number ∧
               strat_b g.player_b_number (g.announcements.take n) = g.player_a_number

/-- There exists a winning strategy for the game -/
theorem exists_winning_strategy : ∃ (strat_a strat_b : Strategy), winning_strategy strat_a strat_b :=
sorry

end exists_winning_strategy_l757_75766


namespace largest_integer_negative_quadratic_six_satisfies_condition_seven_does_not_satisfy_l757_75701

theorem largest_integer_negative_quadratic :
  ∀ n : ℤ, n^2 - 11*n + 28 < 0 → n ≤ 6 :=
by sorry

theorem six_satisfies_condition :
  6^2 - 11*6 + 28 < 0 :=
by sorry

theorem seven_does_not_satisfy :
  7^2 - 11*7 + 28 ≥ 0 :=
by sorry

end largest_integer_negative_quadratic_six_satisfies_condition_seven_does_not_satisfy_l757_75701


namespace trip_distance_l757_75750

theorem trip_distance (speed1 speed2 time_saved : ℝ) (h1 : speed1 = 50) (h2 : speed2 = 60) (h3 : time_saved = 4) :
  let distance := speed1 * speed2 * time_saved / (speed2 - speed1)
  distance = 1200 := by sorry

end trip_distance_l757_75750


namespace area_cosine_plus_one_l757_75747

/-- The area enclosed by y = 1 + cos x and the x-axis over [-π, π] is 2π -/
theorem area_cosine_plus_one : 
  (∫ x in -π..π, (1 + Real.cos x)) = 2 * π := by
  sorry

end area_cosine_plus_one_l757_75747


namespace Z_in_third_quadrant_l757_75723

-- Define the complex number Z
def Z : ℂ := -1 + (1 - Complex.I)^2

-- Theorem statement
theorem Z_in_third_quadrant : 
  Z.re < 0 ∧ Z.im < 0 := by
  sorry

end Z_in_third_quadrant_l757_75723


namespace geometric_sequence_problem_l757_75754

theorem geometric_sequence_problem (a : ℝ) (h1 : a > 0) :
  (∃ r : ℝ, 20 * r = a ∧ a * r = 5/4) → a = 5 := by
  sorry

end geometric_sequence_problem_l757_75754


namespace product_calculation_l757_75782

theorem product_calculation : 
  (1 / 3) * 6 * (1 / 12) * 24 * (1 / 48) * 96 * (1 / 192) * 384 = 16 := by
  sorry

end product_calculation_l757_75782


namespace drive_time_between_towns_l757_75783

/-- Proves that the time to drive between two towns is 4 hours given the map distance, scale, and average speed. -/
theorem drive_time_between_towns
  (map_distance : ℝ)
  (scale_distance : ℝ)
  (scale_miles : ℝ)
  (average_speed : ℝ)
  (h1 : map_distance = 12)
  (h2 : scale_distance = 0.5)
  (h3 : scale_miles = 10)
  (h4 : average_speed = 60)
  : (map_distance * scale_miles / scale_distance) / average_speed = 4 :=
by
  sorry

#check drive_time_between_towns

end drive_time_between_towns_l757_75783


namespace decimal_multiplication_equivalence_l757_75757

theorem decimal_multiplication_equivalence (given : 268 * 74 = 19832) :
  ∃ x : ℝ, 2.68 * x = 1.9832 ∧ x = 0.74 := by
  sorry

end decimal_multiplication_equivalence_l757_75757


namespace f_derivative_positive_implies_a_bound_l757_75724

open Real

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * exp x - 2 * x^2

theorem f_derivative_positive_implies_a_bound (a : ℝ) :
  (∀ x₀ ∈ Set.Ioo 0 1, ∀ ε > 0, ∃ δ > 0, ∀ x ∈ Set.Ioo (x₀ - δ) (x₀ + δ),
    x ≠ x₀ → (f a x - f a x₀ - x + x₀) / (x - x₀) > 0) →
  a > 4 / exp (3/4) :=
sorry

end f_derivative_positive_implies_a_bound_l757_75724


namespace quadratic_function_range_l757_75794

/-- Given a quadratic function f(x) = ax² - c satisfying certain conditions,
    prove that f(3) is within a specific range. -/
theorem quadratic_function_range (a c : ℝ) (f : ℝ → ℝ) 
    (h_def : ∀ x, f x = a * x^2 - c)
    (h_1 : -4 ≤ f 1 ∧ f 1 ≤ -1)
    (h_2 : -1 ≤ f 2 ∧ f 2 ≤ 5) :
  -1 ≤ f 3 ∧ f 3 ≤ 20 := by
sorry

end quadratic_function_range_l757_75794


namespace sin_sum_max_value_l757_75706

open Real

theorem sin_sum_max_value (x₁ x₂ x₃ : ℝ) 
  (h₁ : 0 < x₁ ∧ x₁ < π) 
  (h₂ : 0 < x₂ ∧ x₂ < π) 
  (h₃ : 0 < x₃ ∧ x₃ < π) 
  (h_sum : x₁ + x₂ + x₃ = π) : 
  sin x₁ + sin x₂ + sin x₃ ≤ 2 * sqrt 3 / 3 := by
  sorry

#check sin_sum_max_value

end sin_sum_max_value_l757_75706


namespace cost_calculation_l757_75700

theorem cost_calculation (N P M : ℚ) 
  (eq1 : 13 * N + 26 * P + 19 * M = 25)
  (eq2 : 27 * N + 18 * P + 31 * M = 31) :
  24 * N + 120 * P + 52 * M = 88 := by
sorry

end cost_calculation_l757_75700


namespace triple_hash_40_l757_75745

-- Define the # operation
def hash (N : ℝ) : ℝ := 0.6 * N + 2

-- Theorem statement
theorem triple_hash_40 : hash (hash (hash 40)) = 12.56 := by
  sorry

end triple_hash_40_l757_75745


namespace intersection_A_B_union_A_complement_B_l757_75728

-- Define the sets A and B
def A : Set ℝ := {x | 2 ≤ x ∧ x < 4}
def B : Set ℝ := {x | 3 * x - 7 ≥ 8 - 2 * x}

-- Theorem for the intersection of A and B
theorem intersection_A_B : A ∩ B = {x : ℝ | 3 ≤ x ∧ x < 4} := by sorry

-- Theorem for the union of A and complement of B
theorem union_A_complement_B : A ∪ (Set.univ \ B) = {x : ℝ | x < 4} := by sorry

end intersection_A_B_union_A_complement_B_l757_75728


namespace greater_number_problem_l757_75763

theorem greater_number_problem (a b : ℕ+) (h1 : a * b = 4107) (h2 : Nat.gcd a b = 37) (h3 : a > b) : a = 111 := by
  sorry

end greater_number_problem_l757_75763


namespace fraction_to_decimal_l757_75719

/-- Proves that 37/80 is equal to 0.4625 -/
theorem fraction_to_decimal : (37 : ℚ) / 80 = 0.4625 := by
  sorry

end fraction_to_decimal_l757_75719


namespace cube_root_function_l757_75765

theorem cube_root_function (k : ℝ) :
  (∀ x, x > 0 → ∃ y, y = k * x^(1/3)) →
  (4 * Real.sqrt 3 = k * 64^(1/3)) →
  (2 * Real.sqrt 3 = k * 8^(1/3)) := by
  sorry

end cube_root_function_l757_75765
