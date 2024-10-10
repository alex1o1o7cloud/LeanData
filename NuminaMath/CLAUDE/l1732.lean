import Mathlib

namespace consecutive_numbers_product_l1732_173298

theorem consecutive_numbers_product (A B : Nat) : 
  A ≠ B → 
  A < 10 → 
  B < 10 → 
  35 * 36 * 37 * 38 * 39 = 120 * (100000 * A + 10000 * B + 1000 * A + 100 * B + 10 * A + B) → 
  A = 5 ∧ B = 7 := by
  sorry

end consecutive_numbers_product_l1732_173298


namespace initial_number_proof_l1732_173247

theorem initial_number_proof (x : ℝ) : 
  x + 12.808 - 47.80600000000004 = 3854.002 ↔ x = 3889 := by
  sorry

end initial_number_proof_l1732_173247


namespace truck_speed_on_dirt_road_l1732_173256

/-- A semi truck travels on two types of roads. This theorem proves the speed on the dirt road. -/
theorem truck_speed_on_dirt_road :
  ∀ (v : ℝ),
  (3 * v) + (2 * (v + 20)) = 200 →
  v = 32 := by
sorry

end truck_speed_on_dirt_road_l1732_173256


namespace brick_factory_workers_l1732_173287

/-- The maximum number of workers that can be hired at a brick factory -/
def max_workers : ℕ := 8

theorem brick_factory_workers :
  ∀ n : ℕ,
  n ≤ max_workers ↔
  (10 * n - n * n ≥ 13) ∧
  ∀ m : ℕ, m > n → (10 * m - m * m < 13) :=
by sorry

end brick_factory_workers_l1732_173287


namespace lcm_24_36_40_l1732_173263

theorem lcm_24_36_40 : Nat.lcm (Nat.lcm 24 36) 40 = 360 := by
  sorry

end lcm_24_36_40_l1732_173263


namespace necessary_condition_for_P_l1732_173248

-- Define the set A
def A : Set ℝ := {x : ℝ | -1 < x ∧ x < 2}

-- Define the proposition P(a)
def P (a : ℝ) : Prop := ∀ x ∈ A, x^2 - a < 0

-- Theorem statement
theorem necessary_condition_for_P :
  (∃ a : ℝ, P a) → (∀ a : ℝ, P a → a ≥ 1) ∧ ¬(∀ a : ℝ, a ≥ 1 → P a) := by
  sorry

end necessary_condition_for_P_l1732_173248


namespace quadratic_equation_from_roots_l1732_173232

theorem quadratic_equation_from_roots (x₁ x₂ : ℝ) (hx₁ : x₁ = 1) (hx₂ : x₂ = 2) :
  ∃ a b c : ℝ, a ≠ 0 ∧ a * x₁^2 + b * x₁ + c = 0 ∧ a * x₂^2 + b * x₂ + c = 0 ∧
  a * x^2 + b * x + c = x^2 - 3*x + 2 :=
sorry

end quadratic_equation_from_roots_l1732_173232


namespace alloy_cut_theorem_l1732_173254

/-- Represents an alloy piece with its mass and copper concentration -/
structure AlloyPiece where
  mass : ℝ
  copper_concentration : ℝ

/-- Represents the result of cutting and swapping parts of two alloy pieces -/
def cut_and_swap (piece1 piece2 : AlloyPiece) (cut_mass : ℝ) : Prop :=
  let new_piece1 := AlloyPiece.mk piece1.mass 
    ((cut_mass * piece2.copper_concentration + (piece1.mass - cut_mass) * piece1.copper_concentration) / piece1.mass)
  let new_piece2 := AlloyPiece.mk piece2.mass 
    ((cut_mass * piece1.copper_concentration + (piece2.mass - cut_mass) * piece2.copper_concentration) / piece2.mass)
  new_piece1.copper_concentration = new_piece2.copper_concentration

theorem alloy_cut_theorem (piece1 piece2 : AlloyPiece) (cut_mass : ℝ) :
  piece1.mass = piece2.mass →
  piece1.copper_concentration ≠ piece2.copper_concentration →
  cut_and_swap piece1 piece2 cut_mass →
  cut_mass = piece1.mass / 2 :=
sorry

end alloy_cut_theorem_l1732_173254


namespace compound_interest_problem_l1732_173215

/-- Calculates the compound interest earned given the initial principal, interest rate, 
    compounding frequency, time, and final amount -/
def compound_interest_earned (principal : ℝ) (rate : ℝ) (n : ℝ) (time : ℝ) (final_amount : ℝ) : ℝ :=
  final_amount - principal

/-- Theorem stating that for an investment with 8% annual interest rate compounded annually 
    for 2 years, resulting in a total of 19828.80, the interest earned is 2828.80 -/
theorem compound_interest_problem :
  ∃ (principal : ℝ),
    principal > 0 ∧
    (principal * (1 + 0.08)^2 = 19828.80) ∧
    (compound_interest_earned principal 0.08 1 2 19828.80 = 2828.80) := by
  sorry


end compound_interest_problem_l1732_173215


namespace factorization_1_factorization_2_l1732_173219

-- Factorization of -4a²x + 12ax - 9x
theorem factorization_1 (a x : ℝ) : -4 * a^2 * x + 12 * a * x - 9 * x = -x * (2*a - 3)^2 := by
  sorry

-- Factorization of (2x + y)² - (x + 2y)²
theorem factorization_2 (x y : ℝ) : (2*x + y)^2 - (x + 2*y)^2 = 3 * (x + y) * (x - y) := by
  sorry

end factorization_1_factorization_2_l1732_173219


namespace min_value_of_f_l1732_173249

noncomputable def f (x : ℝ) : ℝ := 9 / (8 * Real.cos (2 * x) + 16) - Real.sin x ^ 2

theorem min_value_of_f (x : ℝ) :
  f x ≥ 0 ∧ (f x = 0 ↔ Real.cos (2 * x) = -1/2) :=
sorry

end min_value_of_f_l1732_173249


namespace parabola_equation_l1732_173273

/-- A parabola with axis of symmetry parallel to the y-axis -/
structure Parabola where
  a : ℝ
  eq : ℝ → ℝ
  eq_def : ∀ x, eq x = a * (x - 1) * (x - 4)

/-- The line y = 2x -/
def line (x : ℝ) : ℝ := 2 * x

theorem parabola_equation (p : Parabola) (h1 : p.eq 1 = 0) (h2 : p.eq 4 = 0)
  (h_tangent : ∃ x, p.eq x = line x ∧ ∀ y ≠ x, p.eq y ≠ line y) :
  p.a = -2/9 ∨ p.a = -2 :=
sorry

end parabola_equation_l1732_173273


namespace set_membership_implies_m_values_l1732_173271

theorem set_membership_implies_m_values (m : ℝ) : 
  let A : Set ℝ := {1, m + 2, m^2 + 4}
  5 ∈ A → (m = 3 ∨ m = 1) :=
by sorry

end set_membership_implies_m_values_l1732_173271


namespace rebecca_eggs_count_l1732_173210

/-- Given that Rebecca wants to split eggs into 3 groups with 5 eggs in each group,
    prove that the total number of eggs is 15. -/
theorem rebecca_eggs_count :
  let num_groups : ℕ := 3
  let eggs_per_group : ℕ := 5
  num_groups * eggs_per_group = 15 := by sorry

end rebecca_eggs_count_l1732_173210


namespace a_equals_permutation_l1732_173238

-- Define a as the product n(n-1)(n-2)...(n-50)
def a (n : ℕ) : ℕ := (List.range 51).foldl (λ acc i => acc * (n - i)) n

-- Define the permutation function A_n^k
def permutation (n k : ℕ) : ℕ := (List.range k).foldl (λ acc i => acc * (n - i)) 1

-- Theorem statement
theorem a_equals_permutation (n : ℕ) : a n = permutation n 51 := by sorry

end a_equals_permutation_l1732_173238


namespace A_intersect_B_eq_set_l1732_173200

def A : Set ℝ := {-2, -1, 0, 1}

def B : Set ℝ := {y | ∃ x, y = 1 / (2^x - 2)}

theorem A_intersect_B_eq_set : A ∩ B = {-2, -1, 1} := by sorry

end A_intersect_B_eq_set_l1732_173200


namespace tangent_line_product_l1732_173216

/-- A cubic function with parameters a and b -/
def f (a b : ℝ) (x : ℝ) : ℝ := x^3 - 3*a*x + b

/-- The derivative of f with respect to x -/
def f_derivative (a : ℝ) (x : ℝ) : ℝ := 3*x^2 - 3*a

theorem tangent_line_product (a b : ℝ) (h1 : a ≠ 0) :
  f_derivative a 2 = 0 ∧ f a b 2 = 8 → a * b = 128 := by
  sorry

end tangent_line_product_l1732_173216


namespace samara_alligators_l1732_173289

theorem samara_alligators (group_size : ℕ) (friends_count : ℕ) (friends_average : ℕ) (total_alligators : ℕ) :
  group_size = friends_count + 1 →
  friends_count = 3 →
  friends_average = 10 →
  total_alligators = 50 →
  total_alligators = friends_count * friends_average + (total_alligators - friends_count * friends_average) →
  (total_alligators - friends_count * friends_average) = 20 := by
  sorry

end samara_alligators_l1732_173289


namespace business_ownership_l1732_173252

theorem business_ownership (total_value : ℝ) (sold_fraction : ℝ) (sold_value : ℝ) 
  (h1 : total_value = 75000)
  (h2 : sold_fraction = 3/5)
  (h3 : sold_value = 15000) :
  (sold_value / sold_fraction) / total_value = 1/3 := by
  sorry

end business_ownership_l1732_173252


namespace emu_count_correct_l1732_173235

/-- Represents the number of emus in Farmer Brown's flock -/
def num_emus : ℕ := 20

/-- Represents the total number of heads and legs in the flock -/
def total_heads_and_legs : ℕ := 60

/-- Represents the number of parts (head + legs) per emu -/
def parts_per_emu : ℕ := 3

/-- Theorem stating that the number of emus is correct given the total number of heads and legs -/
theorem emu_count_correct : num_emus * parts_per_emu = total_heads_and_legs := by
  sorry

end emu_count_correct_l1732_173235


namespace hilt_friends_cant_go_l1732_173253

/-- Given a total number of friends and the number of friends that can go to the movies,
    calculate the number of friends who can't go to the movies. -/
def friends_cant_go (total_friends : ℕ) (friends_going : ℕ) : ℕ :=
  total_friends - friends_going

/-- Theorem stating that with 25 total friends and 6 friends going to the movies,
    19 friends can't go to the movies. -/
theorem hilt_friends_cant_go :
  friends_cant_go 25 6 = 19 := by
  sorry

end hilt_friends_cant_go_l1732_173253


namespace polynomial_remainder_zero_l1732_173214

theorem polynomial_remainder_zero (x : ℝ) : 
  (x^3 - 5*x^2 + 2*x + 8) % (x - 2) = 0 := by
  sorry

end polynomial_remainder_zero_l1732_173214


namespace inequality_solution_l1732_173226

open Set
open Function
open Real

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the condition f'(x) > f(x)
variable (h : ∀ x : ℝ, deriv f x > f x)

-- Define the solution set
def solution_set := {x : ℝ | Real.exp (f (Real.log x)) - x * f 1 < 0}

-- Theorem statement
theorem inequality_solution :
  solution_set f = Ioo 0 (Real.exp 1) :=
sorry

end inequality_solution_l1732_173226


namespace paper_division_l1732_173213

/-- Represents the number of pieces after n divisions -/
def pieces (n : ℕ) : ℕ := 3 * n + 1

/-- The main theorem about paper division -/
theorem paper_division :
  (∀ n : ℕ, pieces n = 3 * n + 1) ∧
  (∃ n : ℕ, pieces n = 2011) :=
by sorry

end paper_division_l1732_173213


namespace range_of_a_l1732_173218

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, x^2 + 2*a*x + a > 0) → 0 < a ∧ a < 1 := by
  sorry

end range_of_a_l1732_173218


namespace marching_band_ratio_l1732_173251

theorem marching_band_ratio (total_students : ℕ) (alto_sax_players : ℕ)
  (h_total : total_students = 600)
  (h_alto : alto_sax_players = 4)
  (h_sax : ∃ sax_players : ℕ, 3 * alto_sax_players = sax_players)
  (h_brass : ∃ brass_players : ℕ, 5 * sax_players = brass_players)
  (h_band : ∃ band_students : ℕ, 2 * brass_players = band_students) :
  band_students / total_students = 1 / 5 :=
by sorry

end marching_band_ratio_l1732_173251


namespace proposition_A_sufficient_not_necessary_for_B_l1732_173260

theorem proposition_A_sufficient_not_necessary_for_B :
  (∀ x : ℝ, 0 < x ∧ x < 5 → |x - 2| < 3) ∧
  (∃ x : ℝ, |x - 2| < 3 ∧ ¬(0 < x ∧ x < 5)) :=
by sorry

end proposition_A_sufficient_not_necessary_for_B_l1732_173260


namespace find_x_l1732_173239

theorem find_x : ∃ x : ℝ, 
  (∃ y : ℝ, y = 1.5 * x ∧ 0.5 * x - 10 = 0.25 * y) → x = 80 := by
  sorry

end find_x_l1732_173239


namespace roots_of_equation_l1732_173217

theorem roots_of_equation (x : ℝ) : 
  (x - 3)^2 = 4 ↔ x = 5 ∨ x = 1 := by sorry

end roots_of_equation_l1732_173217


namespace g_1001_value_l1732_173245

/-- A function satisfying the given functional equation -/
def FunctionalEquation (g : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, g (x * y) + x = x * g y + g x

theorem g_1001_value
  (g : ℝ → ℝ)
  (h1 : FunctionalEquation g)
  (h2 : g 1 = -3) :
  g 1001 = -2001 := by
  sorry

end g_1001_value_l1732_173245


namespace total_snacks_weight_l1732_173259

-- Define the conversion rate from ounces to pounds
def ounces_to_pounds : ℚ → ℚ := (· / 16)

-- Define the weights of snacks
def peanuts_weight : ℚ := 0.1
def raisins_weight_oz : ℚ := 5
def almonds_weight : ℚ := 0.3

-- Theorem to prove
theorem total_snacks_weight :
  peanuts_weight + ounces_to_pounds raisins_weight_oz + almonds_weight = 0.7125 := by
  sorry

end total_snacks_weight_l1732_173259


namespace max_value_of_f_l1732_173284

-- Define the function f(x)
def f (x : ℝ) := x * (1 - x)

-- State the theorem
theorem max_value_of_f :
  ∃ (M : ℝ), M = 1/4 ∧ ∀ x, 0 < x → x < 1 → f x ≤ M :=
by
  sorry

end max_value_of_f_l1732_173284


namespace negation_equivalence_l1732_173244

theorem negation_equivalence (S : Set ℕ) :
  (¬ ∀ x ∈ S, x^2 ≠ 4) ↔ (∃ x ∈ S, x^2 = 4) :=
by sorry

end negation_equivalence_l1732_173244


namespace quadratic_function_negative_at_four_l1732_173295

def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

theorem quadratic_function_negative_at_four
  (a b c : ℝ)
  (h1 : f a b c (-1) = -3)
  (h2 : f a b c 0 = 1)
  (h3 : f a b c 1 = 3)
  (h4 : f a b c 3 = 1) :
  f a b c 4 < 0 := by
sorry

end quadratic_function_negative_at_four_l1732_173295


namespace sequence_length_problem_solution_l1732_173202

def arithmetic_sequence (a₁ : ℤ) (d : ℤ) (n : ℕ) : List ℤ :=
  List.range n |>.map (fun i => a₁ + d * i)

theorem sequence_length (a₁ a_n d : ℤ) (h : d ≠ 0) :
  ∃ n : ℕ, arithmetic_sequence a₁ d n = List.reverse (arithmetic_sequence a_n (-d) n) ∧
           a₁ = a_n + (n - 1) * d :=
by sorry

theorem problem_solution :
  let a₁ := 160
  let a_n := 28
  let d := -4
  ∃ n : ℕ, arithmetic_sequence a₁ d n = List.reverse (arithmetic_sequence a_n (-d) n) ∧
           n = 34 :=
by sorry

end sequence_length_problem_solution_l1732_173202


namespace volume_of_solid_l1732_173240

/-- Volume of a solid with specific dimensions -/
theorem volume_of_solid (a : ℝ) (h1 : a = 3 * Real.sqrt 2) : 
  2 * a^3 = 108 * Real.sqrt 2 := by
  sorry

#check volume_of_solid

end volume_of_solid_l1732_173240


namespace year_2023_ad_representation_l1732_173201

/-- Represents a year in the Gregorian calendar. -/
structure Year where
  value : Int
  is_ad : Bool

/-- Converts a Year to its numerical representation. -/
def Year.to_int (y : Year) : Int :=
  if y.is_ad then y.value else -y.value

/-- The year 500 BC -/
def year_500_bc : Year := { value := 500, is_ad := false }

/-- The year 2023 AD -/
def year_2023_ad : Year := { value := 2023, is_ad := true }

/-- Theorem stating that given 500 BC is denoted as -500, 2023 AD is denoted as +2023 -/
theorem year_2023_ad_representation :
  (year_500_bc.to_int = -500) → (year_2023_ad.to_int = 2023) := by
  sorry

end year_2023_ad_representation_l1732_173201


namespace one_more_stork_than_birds_l1732_173293

/-- Given the initial number of storks and birds on a fence, and additional birds that join,
    prove that there is one more stork than the total number of birds. -/
theorem one_more_stork_than_birds 
  (initial_storks : ℕ) 
  (initial_birds : ℕ) 
  (new_birds : ℕ) 
  (h1 : initial_storks = 6) 
  (h2 : initial_birds = 2) 
  (h3 : new_birds = 3) :
  initial_storks - (initial_birds + new_birds) = 1 := by
  sorry

end one_more_stork_than_birds_l1732_173293


namespace equation_solution_l1732_173274

/-- Converts a number from base b to base 10 -/
def to_base_10 (digits : List Nat) (b : Nat) : Nat :=
  digits.enum.foldr (fun (i, d) acc => acc + d * b^i) 0

/-- Checks if the equation 3△_4 = △2_11 is satisfied -/
def equation_satisfied (triangle : Nat) : Prop :=
  to_base_10 [3, triangle] 4 = to_base_10 [triangle, 2] 11

/-- Theorem stating that the equation is satisfied when triangle is 1 -/
theorem equation_solution :
  equation_satisfied 1 := by sorry

end equation_solution_l1732_173274


namespace min_races_for_top_3_l1732_173288

/-- Represents a horse in the race. -/
structure Horse :=
  (id : Nat)

/-- Represents a race with up to 6 horses. -/
structure Race :=
  (horses : Finset Horse)
  (condition : Nat)  -- Represents different race conditions

/-- A function to determine the ranking of horses in a race. -/
def raceResult (r : Race) : List Horse := sorry

/-- The total number of horses. -/
def totalHorses : Nat := 30

/-- The maximum number of horses that can race together. -/
def maxHorsesPerRace : Nat := 6

/-- A function to determine if we have found the top 3 horses. -/
def hasTop3 (races : List Race) : Bool := sorry

/-- Theorem stating the minimum number of races needed. -/
theorem min_races_for_top_3 :
  ∃ (races : List Race),
    races.length = 7 ∧
    hasTop3 races ∧
    ∀ (other_races : List Race),
      hasTop3 other_races → other_races.length ≥ 7 := by sorry

end min_races_for_top_3_l1732_173288


namespace circle_line_tangent_problem_l1732_173211

-- Define the circle C
def circle_C (x y : ℝ) : Prop :=
  x^2 + y^2 - 2*x - 4*y + 1 = 0

-- Define the line l
def line_l (x y : ℝ) (m : ℝ) : Prop :=
  x + m*y + 1 = 0

-- Define the point M
def point_M (m : ℝ) : ℝ × ℝ :=
  (m, m)

-- Define the symmetric property
def symmetric_points_exist (m : ℝ) : Prop :=
  ∃ (x1 y1 x2 y2 : ℝ), 
    circle_C x1 y1 ∧ circle_C x2 y2 ∧
    line_l ((x1 + x2)/2) ((y1 + y2)/2) m

-- Define the tangent property
def tangent_exists (m : ℝ) : Prop :=
  ∃ (x y : ℝ), circle_C x y ∧
    ∃ (k : ℝ), ∀ (t : ℝ),
      ¬(circle_C (m + k*t) (m + t))

-- Main theorem
theorem circle_line_tangent_problem (m : ℝ) :
  symmetric_points_exist m → tangent_exists m →
  m = -1 ∧ Real.sqrt ((m - 1)^2 + (m - 2)^2 - 4) = 3 :=
sorry

end circle_line_tangent_problem_l1732_173211


namespace garden_perimeter_garden_perimeter_proof_l1732_173224

/-- The perimeter of a rectangular garden with a width of 12 meters and an area equal to a 16x12 meter playground is 56 meters. -/
theorem garden_perimeter : ℝ → ℝ → ℝ → ℝ → Prop :=
  fun garden_width garden_length playground_length playground_width =>
    garden_width = 12 ∧
    garden_length * garden_width = playground_length * playground_width ∧
    playground_length = 16 ∧
    playground_width = 12 →
    2 * (garden_length + garden_width) = 56

-- The proof is omitted
theorem garden_perimeter_proof : garden_perimeter 12 16 16 12 := by sorry

end garden_perimeter_garden_perimeter_proof_l1732_173224


namespace gingers_size_l1732_173208

theorem gingers_size (anna_size becky_size ginger_size : ℕ) : 
  anna_size = 2 →
  becky_size = 3 * anna_size →
  ginger_size = 2 * becky_size - 4 →
  ginger_size = 8 := by
sorry

end gingers_size_l1732_173208


namespace inequality_proof_l1732_173242

theorem inequality_proof (a b c d : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d)
  (sum_eq_one : a + b + c + d = 1) : 
  b * c * d / (1 - a)^2 + c * d * a / (1 - b)^2 + 
  d * a * b / (1 - c)^2 + a * b * c / (1 - d)^2 ≤ 1 / 9 := by
sorry

end inequality_proof_l1732_173242


namespace alternating_draw_probability_l1732_173250

/-- The number of white balls in the box -/
def white_balls : Nat := 5

/-- The number of black balls in the box -/
def black_balls : Nat := 5

/-- The total number of balls in the box -/
def total_balls : Nat := white_balls + black_balls

/-- The number of ways to arrange white_balls white balls and black_balls black balls -/
def total_arrangements : Nat := Nat.choose total_balls white_balls

/-- The number of valid alternating color patterns -/
def valid_patterns : Nat := 2

/-- The probability of drawing all balls in an alternating color pattern -/
def alternating_probability : ℚ := valid_patterns / total_arrangements

theorem alternating_draw_probability : alternating_probability = 1 / 126 := by
  sorry

end alternating_draw_probability_l1732_173250


namespace function_not_in_first_quadrant_l1732_173220

theorem function_not_in_first_quadrant (a b : ℝ) (ha : 0 < a ∧ a < 1) (hb : b < -1) :
  ∀ x y : ℝ, x > 0 → y > 0 → a^x + b < y :=
by sorry

end function_not_in_first_quadrant_l1732_173220


namespace estate_value_l1732_173262

/-- Represents the estate distribution problem --/
structure EstateDistribution where
  total : ℝ
  daughter_share : ℝ
  son_share : ℝ
  wife_share : ℝ
  brother_share : ℝ
  nanny_share : ℝ

/-- Theorem stating the conditions and the result to be proved --/
theorem estate_value (e : EstateDistribution) : 
  e.daughter_share + e.son_share = (3/5) * e.total ∧ 
  e.daughter_share = (5/7) * (e.daughter_share + e.son_share) ∧
  e.son_share = (2/7) * (e.daughter_share + e.son_share) ∧
  e.wife_share = 3 * e.son_share ∧
  e.brother_share = e.daughter_share ∧
  e.nanny_share = 400 ∧
  e.total = e.daughter_share + e.son_share + e.wife_share + e.brother_share + e.nanny_share
  →
  e.total = 825 := by
  sorry

#eval 825 -- To display the result

end estate_value_l1732_173262


namespace mosaic_tiles_l1732_173228

/-- Calculates the number of square tiles needed to cover a rectangular area -/
def tilesNeeded (height_feet width_feet tile_side_inches : ℕ) : ℕ :=
  (height_feet * 12 * width_feet * 12) / (tile_side_inches * tile_side_inches)

/-- Theorem stating the number of 1-inch square tiles needed for a 10ft by 15ft mosaic -/
theorem mosaic_tiles : tilesNeeded 10 15 1 = 21600 := by
  sorry

end mosaic_tiles_l1732_173228


namespace two_books_cost_exceeds_min_preparation_l1732_173280

/-- The cost of one storybook in yuan -/
def storybook_cost : ℚ := 25.5

/-- The minimum amount Wang Hong needs to prepare in yuan -/
def min_preparation : ℚ := 50

/-- Theorem: The cost of two storybooks is greater than the minimum preparation amount -/
theorem two_books_cost_exceeds_min_preparation : 2 * storybook_cost > min_preparation := by
  sorry

end two_books_cost_exceeds_min_preparation_l1732_173280


namespace danai_decorations_l1732_173278

def total_decorations (skulls broomsticks spiderwebs cauldron additional_budget additional_left : ℕ) : ℕ :=
  skulls + broomsticks + spiderwebs + (2 * spiderwebs) + cauldron + additional_budget + additional_left

theorem danai_decorations :
  let skulls : ℕ := 12
  let broomsticks : ℕ := 4
  let spiderwebs : ℕ := 12
  let cauldron : ℕ := 1
  let additional_budget : ℕ := 20
  let additional_left : ℕ := 10
  total_decorations skulls broomsticks spiderwebs cauldron additional_budget additional_left = 83 :=
by
  sorry

end danai_decorations_l1732_173278


namespace bretschneiders_theorem_l1732_173272

-- Define a convex quadrilateral
structure ConvexQuadrilateral where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ
  m : ℝ
  n : ℝ
  A : ℝ
  C : ℝ
  a_positive : a > 0
  b_positive : b > 0
  c_positive : c > 0
  d_positive : d > 0
  m_positive : m > 0
  n_positive : n > 0
  A_range : 0 < A ∧ A < π
  C_range : 0 < C ∧ C < π

-- State Bretschneider's theorem
theorem bretschneiders_theorem (q : ConvexQuadrilateral) :
  q.m^2 * q.n^2 = q.a^2 * q.c^2 + q.b^2 * q.d^2 - 2 * q.a * q.b * q.c * q.d * Real.cos (q.A + q.C) :=
sorry

end bretschneiders_theorem_l1732_173272


namespace sin_A_value_side_c_value_l1732_173203

-- Define the triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- Define the conditions
def triangle_conditions (t : Triangle) : Prop :=
  t.C = 2 * Real.pi / 3 ∧ t.a = 6

-- Theorem 1
theorem sin_A_value (t : Triangle) (h : triangle_conditions t) (hc : t.c = 14) :
  Real.sin t.A = (3 / 14) * Real.sqrt 3 := by
  sorry

-- Theorem 2
theorem side_c_value (t : Triangle) (h : triangle_conditions t) (harea : 1/2 * t.a * t.b * Real.sin t.C = 3 * Real.sqrt 3) :
  t.c = 2 * Real.sqrt 13 := by
  sorry

end sin_A_value_side_c_value_l1732_173203


namespace rocket_max_height_l1732_173267

/-- The height function of the rocket -/
def h (t : ℝ) : ℝ := -20 * t^2 + 80 * t + 50

/-- The maximum height reached by the rocket -/
theorem rocket_max_height :
  ∃ (t_max : ℝ), ∀ (t : ℝ), h t ≤ h t_max ∧ h t_max = 130 :=
sorry

end rocket_max_height_l1732_173267


namespace exam_score_calculation_l1732_173241

theorem exam_score_calculation (total_questions : ℕ) (correct_answers : ℕ) (total_marks : ℤ) :
  total_questions = 120 →
  correct_answers = 75 →
  total_marks = 180 →
  (∃ (score_per_correct : ℕ),
    score_per_correct * correct_answers - (total_questions - correct_answers) = total_marks ∧
    score_per_correct = 3) :=
by
  sorry

end exam_score_calculation_l1732_173241


namespace tower_combinations_l1732_173285

/-- Represents the number of cubes of each color --/
structure CubeColors where
  red : Nat
  blue : Nat
  green : Nat
  yellow : Nat

/-- Calculates the number of different towers that can be built --/
def numTowers (colors : CubeColors) (towerHeight : Nat) : Nat :=
  if towerHeight ≠ colors.red + colors.blue + colors.green + colors.yellow - 1 then 0
  else if colors.yellow = 0 then 0
  else
    let n := towerHeight - 1
    Nat.factorial n / (Nat.factorial colors.red * Nat.factorial colors.blue * 
                       Nat.factorial colors.green * Nat.factorial (colors.yellow - 1))

/-- The main theorem to be proven --/
theorem tower_combinations : 
  let colors := CubeColors.mk 3 4 2 2
  numTowers colors 10 = 1260 := by
  sorry

end tower_combinations_l1732_173285


namespace factor_polynomial_l1732_173279

theorem factor_polynomial (x : ℝ) : 54 * x^5 - 135 * x^9 = 27 * x^5 * (2 - 5 * x^4) := by
  sorry

end factor_polynomial_l1732_173279


namespace bucket_filling_time_l1732_173246

theorem bucket_filling_time (total_time : ℝ) (total_fraction : ℝ) (partial_fraction : ℝ) : 
  total_time = 150 → total_fraction = 1 → partial_fraction = 2/3 →
  (partial_fraction * total_time) / total_fraction = 100 := by
sorry

end bucket_filling_time_l1732_173246


namespace max_integer_a_for_real_roots_l1732_173275

theorem max_integer_a_for_real_roots : 
  ∀ a : ℤ, (∃ x : ℝ, (a + 1 : ℝ) * x^2 - 2*x + 3 = 0) → a ≤ -2 :=
by sorry

end max_integer_a_for_real_roots_l1732_173275


namespace race_outcomes_six_participants_l1732_173236

/-- The number of different 1st-2nd-3rd-4th place outcomes in a race with 6 participants and no ties -/
def race_outcomes (n : ℕ) : ℕ :=
  if n ≥ 4 then n * (n - 1) * (n - 2) * (n - 3) else 0

/-- Theorem: The number of different 1st-2nd-3rd-4th place outcomes in a race with 6 participants and no ties is 360 -/
theorem race_outcomes_six_participants : race_outcomes 6 = 360 := by
  sorry

end race_outcomes_six_participants_l1732_173236


namespace student_rank_l1732_173270

theorem student_rank (total_students : ℕ) (rank_from_right : ℕ) (rank_from_left : ℕ) :
  total_students = 20 →
  rank_from_right = 13 →
  rank_from_left = total_students - rank_from_right + 1 →
  rank_from_left = 9 := by
  sorry

end student_rank_l1732_173270


namespace necessary_not_sufficient_condition_l1732_173277

theorem necessary_not_sufficient_condition :
  (∀ x : ℝ, |x - 2| < 1 → 1 < x ∧ x < 4) ∧
  ¬(∀ x : ℝ, 1 < x ∧ x < 4 → |x - 2| < 1) := by
  sorry

end necessary_not_sufficient_condition_l1732_173277


namespace basement_bulbs_l1732_173233

def light_bulbs_problem (bedroom bathroom kitchen basement garage : ℕ) : Prop :=
  bedroom = 2 ∧
  bathroom = 1 ∧
  kitchen = 1 ∧
  garage = basement / 2 ∧
  bedroom + bathroom + kitchen + basement + garage = 12

theorem basement_bulbs :
  ∃ (bedroom bathroom kitchen basement garage : ℕ),
    light_bulbs_problem bedroom bathroom kitchen basement garage ∧
    basement = 5 := by
  sorry

end basement_bulbs_l1732_173233


namespace tripled_rectangle_area_l1732_173296

/-- Theorem: New area of a tripled rectangle --/
theorem tripled_rectangle_area (k m : ℝ) (hk : k > 0) (hm : m > 0) : 
  let original_area := (6 * k) * (4 * m)
  let new_area := 3 * original_area
  new_area = 72 * k * m := by
  sorry


end tripled_rectangle_area_l1732_173296


namespace min_socks_for_ten_pairs_l1732_173204

/-- Represents the number of colors of socks in the drawer -/
def num_colors : ℕ := 4

/-- Represents the number of pairs we want to ensure -/
def required_pairs : ℕ := 10

/-- Calculates the minimum number of socks needed to ensure the required number of pairs -/
def min_socks (colors : ℕ) (pairs : ℕ) : ℕ :=
  3 + 2 * pairs

/-- Theorem stating that the minimum number of socks needed to ensure 10 pairs from 4 colors is 23 -/
theorem min_socks_for_ten_pairs : 
  min_socks num_colors required_pairs = 23 := by
  sorry

#eval min_socks num_colors required_pairs

end min_socks_for_ten_pairs_l1732_173204


namespace six_women_four_men_arrangements_l1732_173207

/-- The number of ways to arrange n indistinguishable objects of one type
    and m indistinguishable objects of another type in a row,
    such that no two objects of the same type are adjacent -/
def alternating_arrangements (n m : ℕ) : ℕ := sorry

/-- Theorem stating that there are 6 ways to arrange 6 women and 4 men
    alternately in a row -/
theorem six_women_four_men_arrangements :
  alternating_arrangements 6 4 = 6 := by sorry

end six_women_four_men_arrangements_l1732_173207


namespace max_min_difference_l1732_173223

theorem max_min_difference (a b : ℝ) : 
  a^2 + b^2 - 2*a - 4 = 0 → 
  (∃ (t_max t_min : ℝ), 
    (∀ t : ℝ, (∃ a' b' : ℝ, a'^2 + b'^2 - 2*a' - 4 = 0 ∧ t = 2*a' - b') → t_min ≤ t ∧ t ≤ t_max) ∧
    t_max - t_min = 10) :=
by sorry

end max_min_difference_l1732_173223


namespace chair_sequence_l1732_173292

theorem chair_sequence (seq : ℕ → ℕ) 
  (h1 : seq 1 = 14)
  (h2 : seq 2 = 23)
  (h3 : seq 3 = 32)
  (h4 : seq 4 = 41)
  (h6 : seq 6 = 59)
  (h_arithmetic : ∀ n : ℕ, n ≥ 1 → seq (n + 1) - seq n = seq 2 - seq 1) :
  seq 5 = 50 := by
  sorry

end chair_sequence_l1732_173292


namespace range_of_a_l1732_173255

theorem range_of_a (a : ℝ) : (∃ x : ℝ, (Real.exp x - a)^2 + x^2 - 2*a*x + a^2 ≤ 1/2) → a = 1/2 := by
  sorry

end range_of_a_l1732_173255


namespace calculation_proof_l1732_173294

theorem calculation_proof : 
  |Real.sqrt 3 - 1| - (-Real.sqrt 3)^2 - 12 * (-1/3) = Real.sqrt 3 := by
  sorry

end calculation_proof_l1732_173294


namespace inequality_solution_set_l1732_173222

theorem inequality_solution_set 
  (a : ℝ) 
  (h1 : 0 < a) 
  (h2 : a < 1) 
  (h3 : ∀ x : ℝ, x^2 - 2*a*x + a > 0) : 
  {x : ℝ | a^(x^2 - 3) < a^(2*x) ∧ a^(2*x) < 1} = {x : ℝ | x > 3} := by
  sorry

end inequality_solution_set_l1732_173222


namespace total_wheels_is_64_l1732_173282

/-- The number of wheels on a four-wheeler -/
def wheels_per_four_wheeler : ℕ := 4

/-- The number of four-wheelers parked in the school -/
def num_four_wheelers : ℕ := 16

/-- The total number of wheels for all four-wheelers parked in the school -/
def total_wheels_four_wheelers : ℕ := num_four_wheelers * wheels_per_four_wheeler

/-- Theorem: The total number of wheels for the four-wheelers parked in the school is 64 -/
theorem total_wheels_is_64 : total_wheels_four_wheelers = 64 := by
  sorry

end total_wheels_is_64_l1732_173282


namespace exists_monomial_neg5_deg2_l1732_173265

/-- A monomial is a product of a coefficient and variables raised to non-negative integer powers. -/
structure Monomial where
  coefficient : ℤ
  degree : ℕ

/-- Checks if a given monomial has the specified coefficient and degree. -/
def has_coeff_and_degree (m : Monomial) (c : ℤ) (d : ℕ) : Prop :=
  m.coefficient = c ∧ m.degree = d

/-- There exists a monomial with coefficient -5 and degree 2. -/
theorem exists_monomial_neg5_deg2 : ∃ m : Monomial, has_coeff_and_degree m (-5) 2 := by
  sorry

end exists_monomial_neg5_deg2_l1732_173265


namespace instantaneous_velocity_at_5_l1732_173205

-- Define the distance-time function
def s (t : ℝ) : ℝ := 4 * t^2 - 3

-- State the theorem
theorem instantaneous_velocity_at_5 :
  (deriv s) 5 = 40 := by sorry

end instantaneous_velocity_at_5_l1732_173205


namespace eggs_cooked_per_year_l1732_173276

/-- The number of eggs Lisa cooks for her family for breakfast in a year -/
def eggs_per_year : ℕ :=
  let days_per_week : ℕ := 5
  let weeks_per_year : ℕ := 52
  let num_children : ℕ := 4
  let eggs_per_child : ℕ := 2
  let eggs_for_husband : ℕ := 3
  let eggs_for_self : ℕ := 2
  let eggs_per_day : ℕ := num_children * eggs_per_child + eggs_for_husband + eggs_for_self
  eggs_per_day * days_per_week * weeks_per_year

theorem eggs_cooked_per_year :
  eggs_per_year = 3380 := by
  sorry

end eggs_cooked_per_year_l1732_173276


namespace total_shirts_bought_l1732_173286

theorem total_shirts_bought (cost_15 : ℕ) (price_15 : ℕ) (price_20 : ℕ) (total_cost : ℕ) :
  cost_15 = 3 →
  price_15 = 15 →
  price_20 = 20 →
  total_cost = 85 →
  ∃ (cost_20 : ℕ), cost_15 * price_15 + cost_20 * price_20 = total_cost ∧ cost_15 + cost_20 = 5 :=
by sorry

end total_shirts_bought_l1732_173286


namespace basketball_game_points_l1732_173258

/-- Calculate total points for a player given their shot counts -/
def playerPoints (twoPoints threePoints freeThrows : ℕ) : ℕ :=
  2 * twoPoints + 3 * threePoints + freeThrows

/-- Calculate total points for a team given two players' shot counts -/
def teamPoints (p1TwoPoints p1ThreePoints p1FreeThrows
                p2TwoPoints p2ThreePoints p2FreeThrows : ℕ) : ℕ :=
  playerPoints p1TwoPoints p1ThreePoints p1FreeThrows +
  playerPoints p2TwoPoints p2ThreePoints p2FreeThrows

/-- Theorem: The combined points of both teams is 128 -/
theorem basketball_game_points : 
  teamPoints 7 5 4 4 6 7 + teamPoints 9 4 5 6 3 6 = 128 := by
  sorry

#eval teamPoints 7 5 4 4 6 7 + teamPoints 9 4 5 6 3 6

end basketball_game_points_l1732_173258


namespace f_101_form_l1732_173281

def is_valid_f (f : ℕ → ℕ) : Prop :=
  ∀ m n : ℕ, m > 0 ∧ n > 0 → (m * n + 1) ∣ (f m * f n + 1)

theorem f_101_form (f : ℕ → ℕ) (h : is_valid_f f) :
  ∃ k : ℕ, k % 2 = 1 ∧ f 101 = 101^k :=
by sorry

end f_101_form_l1732_173281


namespace lesser_number_l1732_173234

theorem lesser_number (x y : ℤ) (sum_eq : x + y = 58) (diff_eq : x - y = 6) : 
  min x y = 26 := by
sorry

end lesser_number_l1732_173234


namespace smallest_odd_integer_triangle_perimeter_l1732_173297

/-- A function that generates the nth odd integer -/
def nthOddInteger (n : ℕ) : ℕ := 2 * n + 1

/-- A predicate that checks if three numbers form a valid triangle -/
def isValidTriangle (a b c : ℕ) : Prop := a + b > c ∧ a + c > b ∧ b + c > a

/-- The theorem stating the smallest possible perimeter of a triangle with consecutive odd integer sides -/
theorem smallest_odd_integer_triangle_perimeter :
  ∃ (n : ℕ), 
    isValidTriangle (nthOddInteger n) (nthOddInteger (n + 1)) (nthOddInteger (n + 2)) ∧
    (∀ (m : ℕ), m < n → ¬isValidTriangle (nthOddInteger m) (nthOddInteger (m + 1)) (nthOddInteger (m + 2))) ∧
    nthOddInteger n + nthOddInteger (n + 1) + nthOddInteger (n + 2) = 15 :=
sorry

end smallest_odd_integer_triangle_perimeter_l1732_173297


namespace square_area_from_diagonal_l1732_173291

/-- The area of a square given its diagonal length -/
theorem square_area_from_diagonal (d : ℝ) (h : d = 10) : 
  (d ^ 2 / 2 : ℝ) = 50 := by sorry

end square_area_from_diagonal_l1732_173291


namespace burger_cost_proof_l1732_173237

def total_cost : ℝ := 15
def fries_cost : ℝ := 2
def fries_quantity : ℕ := 2
def salad_cost_multiplier : ℕ := 3

theorem burger_cost_proof :
  let salad_cost := salad_cost_multiplier * fries_cost
  let fries_total_cost := fries_quantity * fries_cost
  let burger_cost := total_cost - (salad_cost + fries_total_cost)
  burger_cost = 5 := by sorry

end burger_cost_proof_l1732_173237


namespace empty_plane_speed_theorem_l1732_173230

/-- The speed of an empty plane given the conditions of the problem -/
def empty_plane_speed (p1 p2 p3 : ℕ) (speed_reduction : ℕ) (avg_speed : ℕ) : ℕ :=
  3 * avg_speed + p1 * speed_reduction + p2 * speed_reduction + p3 * speed_reduction

/-- Theorem stating the speed of an empty plane under the given conditions -/
theorem empty_plane_speed_theorem :
  empty_plane_speed 50 60 40 2 500 = 600 := by
  sorry

#eval empty_plane_speed 50 60 40 2 500

end empty_plane_speed_theorem_l1732_173230


namespace initial_amounts_given_final_state_l1732_173269

/-- Represents the state of the game after each round -/
structure GameState where
  player1 : ℤ
  player2 : ℤ
  player3 : ℤ

/-- Simulates one round of the game where the specified player loses -/
def playRound (state : GameState) (loser : Fin 3) : GameState :=
  match loser with
  | 0 => ⟨state.player1 - (state.player2 + state.player3), 
          state.player2 + state.player1, 
          state.player3 + state.player1⟩
  | 1 => ⟨state.player1 + state.player2, 
          state.player2 - (state.player1 + state.player3), 
          state.player3 + state.player2⟩
  | 2 => ⟨state.player1 + state.player3, 
          state.player2 + state.player3, 
          state.player3 - (state.player1 + state.player2)⟩

/-- Theorem stating the initial amounts given the final state -/
theorem initial_amounts_given_final_state 
  (x y z : ℤ) 
  (h1 : playRound (playRound (playRound ⟨x, y, z⟩ 0) 1) 2 = ⟨104, 104, 104⟩) :
  x = 169 ∧ y = 91 ∧ z = 52 := by
  sorry


end initial_amounts_given_final_state_l1732_173269


namespace polynomial_simplification_l1732_173212

theorem polynomial_simplification (x : ℝ) :
  4 * x^3 + 5 * x^2 + 2 * x + 8 - (3 * x^3 - 7 * x^2 + 4 * x - 6) =
  x^3 + 12 * x^2 - 2 * x + 14 := by
  sorry

end polynomial_simplification_l1732_173212


namespace max_rectangle_area_l1732_173290

/-- Given a wire of length 52 cm, the maximum area of a rectangle that can be formed is 169 cm². -/
theorem max_rectangle_area (wire_length : ℝ) (h : wire_length = 52) : 
  (∀ l w : ℝ, l > 0 → w > 0 → 2 * (l + w) ≤ wire_length → l * w ≤ 169) ∧ 
  (∃ l w : ℝ, l > 0 ∧ w > 0 ∧ 2 * (l + w) = wire_length ∧ l * w = 169) :=
sorry

end max_rectangle_area_l1732_173290


namespace simplify_fraction_l1732_173268

theorem simplify_fraction : (3^4 + 3^2) / (3^3 - 3) = 15 / 4 := by
  sorry

end simplify_fraction_l1732_173268


namespace min_value_expression_l1732_173266

theorem min_value_expression (a b : ℝ) (ha : a > 1) (hb : b > 2) :
  (a + b)^2 / (Real.sqrt (a^2 - 1) + Real.sqrt (b^2 - 4)) ≥ 6 := by
  sorry

end min_value_expression_l1732_173266


namespace g_of_3_l1732_173221

def g (x : ℝ) : ℝ := 3 * x^3 + 5 * x^2 - 2 * x - 7

theorem g_of_3 : g 3 = 113 := by
  sorry

end g_of_3_l1732_173221


namespace two_numbers_difference_l1732_173231

theorem two_numbers_difference (x y : ℚ) 
  (sum_eq : x + y = 40)
  (triple_minus_quadruple : 3 * y - 4 * x = 20) :
  |y - x| = 80 / 7 := by
  sorry

end two_numbers_difference_l1732_173231


namespace negation_of_implication_l1732_173225

theorem negation_of_implication (x : ℝ) : 
  ¬(x^2 = 1 → x = 1 ∨ x = -1) ↔ (x^2 ≠ 1 → x ≠ 1 ∧ x ≠ -1) := by sorry

end negation_of_implication_l1732_173225


namespace number_order_l1732_173257

theorem number_order (a : ℝ) (h1 : 0 < a) (h2 : a < 1) :
  (1 / a > Real.sqrt a) ∧ (Real.sqrt a > a) ∧ (a > a^2) := by
  sorry

end number_order_l1732_173257


namespace divisibility_rule_37_l1732_173264

/-- Represents a natural number as a list of its digits -/
def toDigits (n : ℕ) : List ℕ :=
  if n = 0 then [0] else
  let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
    if m = 0 then acc else aux (m / 10) ((m % 10) :: acc)
  aux n []

/-- Calculates the sum for the divisibility rule of 37 -/
def divSum (digits : List ℕ) : ℤ :=
  let rec aux (l : List ℕ) (k : ℕ) : ℤ :=
    match l with
    | [] => 0
    | [a] => a
    | [a, b] => a + 10 * b
    | a :: b :: c :: rest => a + 10 * b - 11 * c + aux rest (k + 1)
  aux digits 0

/-- The divisibility rule for 37 -/
theorem divisibility_rule_37 (n : ℕ) :
  n % 37 = 0 ↔ (divSum (toDigits n)) % 37 = 0 := by
  sorry

end divisibility_rule_37_l1732_173264


namespace parabola_directrix_l1732_173227

/-- The directrix of the parabola y = (x^2 - 4x + 4) / 8 is y = -1/4 -/
theorem parabola_directrix :
  let f : ℝ → ℝ := λ x => (x^2 - 4*x + 4) / 8
  ∃ (directrix : ℝ), directrix = -1/4 ∧
    ∀ (x y : ℝ), y = f x → 
      ∃ (focus : ℝ × ℝ), (x - focus.1)^2 + (y - focus.2)^2 = (y - directrix)^2 :=
by sorry

end parabola_directrix_l1732_173227


namespace cos_negative_thirteen_pi_over_four_l1732_173229

theorem cos_negative_thirteen_pi_over_four :
  Real.cos (-13 * π / 4) = -Real.sqrt 2 / 2 := by sorry

end cos_negative_thirteen_pi_over_four_l1732_173229


namespace extra_lambs_found_l1732_173299

def lambs_problem (initial_lambs : ℕ) (lambs_with_babies : ℕ) (babies_per_lamb : ℕ) 
                  (traded_lambs : ℕ) (final_lambs : ℕ) : ℕ :=
  let lambs_after_babies := initial_lambs + lambs_with_babies * babies_per_lamb
  let lambs_after_trade := lambs_after_babies - traded_lambs
  final_lambs - lambs_after_trade

theorem extra_lambs_found :
  lambs_problem 6 2 2 3 14 = 7 := by
  sorry

end extra_lambs_found_l1732_173299


namespace smallest_three_digit_candy_count_l1732_173243

theorem smallest_three_digit_candy_count (n : ℕ) : 
  (100 ≤ n ∧ n < 1000) →  -- n is a three-digit number
  ((n + 7) % 9 = 0) →     -- if Alicia gains 7 candies, she'll have a multiple of 9
  ((n - 9) % 7 = 0) →     -- if Alicia loses 9 candies, she'll have a multiple of 7
  (∀ m : ℕ, (100 ≤ m ∧ m < n ∧ (m + 7) % 9 = 0 ∧ (m - 9) % 7 = 0) → False) →  -- n is the smallest such number
  n = 101 :=
by sorry

end smallest_three_digit_candy_count_l1732_173243


namespace area_common_triangles_circle_l1732_173206

/-- The area of the region common to two inscribed equilateral triangles and an inscribed circle in a square -/
theorem area_common_triangles_circle (square_side : ℝ) (triangle_side : ℝ) (circle_radius : ℝ) : ℝ :=
  by
  -- Given conditions
  have h1 : square_side = 4 := by sorry
  have h2 : triangle_side = square_side := by sorry
  have h3 : circle_radius = square_side / 2 := by sorry
  
  -- Approximate area calculation
  have triangle_area : ℝ := by sorry
  have circle_area : ℝ := by sorry
  have overlap_per_triangle : ℝ := by sorry
  have total_overlap : ℝ := by sorry
  
  -- Prove the approximate area is 4π
  sorry

#check area_common_triangles_circle

end area_common_triangles_circle_l1732_173206


namespace circle_intersection_slope_range_l1732_173209

-- Define the circle C
def circle_C (x y : ℝ) : Prop :=
  (x - 1)^2 + y^2 = 25

-- Define the line that contains the center of C
def center_line (x y : ℝ) : Prop :=
  2*x - y - 2 = 0

-- Define the line l
def line_l (k : ℝ) (x y : ℝ) : Prop :=
  y - 5 = k*(x + 2)

-- Main theorem
theorem circle_intersection_slope_range :
  ∀ k : ℝ,
  (∃ x₁ y₁ x₂ y₂ : ℝ,
    -- C passes through M(-3,3) and N(1,-5)
    circle_C (-3) 3 ∧ circle_C 1 (-5) ∧
    -- Center of C lies on the given line
    ∃ xc yc : ℝ, circle_C xc yc ∧ center_line xc yc ∧
    -- l intersects C at two distinct points
    x₁ ≠ x₂ ∧ circle_C x₁ y₁ ∧ circle_C x₂ y₂ ∧
    line_l k x₁ y₁ ∧ line_l k x₂ y₂ ∧
    -- l passes through (-2,5)
    line_l k (-2) 5 ∧
    -- k > 0
    k > 0) →
  k > 15/8 :=
by sorry


end circle_intersection_slope_range_l1732_173209


namespace circle_area_with_diameter_l1732_173283

theorem circle_area_with_diameter (d : ℝ) (A : ℝ) :
  d = 7.5 →
  A = π * (d / 2)^2 →
  A = 14.0625 * π :=
by sorry

end circle_area_with_diameter_l1732_173283


namespace prob_three_odd_in_eight_rolls_l1732_173261

/-- The probability of getting an odd number on a single roll of a fair six-sided die -/
def prob_odd : ℚ := 1/2

/-- The number of times the die is rolled -/
def num_rolls : ℕ := 8

/-- The number of odd results we're interested in -/
def target_odd : ℕ := 3

/-- Binomial coefficient -/
def binom (n k : ℕ) : ℕ := Nat.choose n k

/-- The probability of exactly k successes in n trials with probability p of success on each trial -/
def binomial_probability (n k : ℕ) (p : ℚ) : ℚ :=
  (binom n k : ℚ) * p^k * (1 - p)^(n - k)

theorem prob_three_odd_in_eight_rolls :
  binomial_probability num_rolls target_odd prob_odd = 7/32 := by
  sorry

end prob_three_odd_in_eight_rolls_l1732_173261
