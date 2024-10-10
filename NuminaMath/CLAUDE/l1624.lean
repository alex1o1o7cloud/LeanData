import Mathlib

namespace solution_set_correct_l1624_162469

/-- An odd function f: ℝ → ℝ with specific properties -/
class OddFunction (f : ℝ → ℝ) :=
  (odd : ∀ x, f (-x) = -f x)
  (deriv_pos : ∀ x < 0, deriv f x > 0)
  (zero_at_neg_half : f (-1/2) = 0)

/-- The solution set for f(x) < 0 given an odd function with specific properties -/
def solution_set (f : ℝ → ℝ) [OddFunction f] : Set ℝ :=
  {x | x < -1/2 ∨ (0 < x ∧ x < 1/2)}

/-- Theorem stating that the solution set is correct -/
theorem solution_set_correct (f : ℝ → ℝ) [OddFunction f] :
  ∀ x, f x < 0 ↔ x ∈ solution_set f :=
sorry

end solution_set_correct_l1624_162469


namespace samuel_coaching_discontinue_date_l1624_162491

/-- Represents a date in a non-leap year -/
structure Date where
  month : Nat
  day : Nat

/-- Calculates the number of days from January 1st to a given date in a non-leap year -/
def daysFromNewYear (d : Date) : Nat :=
  sorry

/-- The date Samuel discontinued coaching -/
def discontinueDate : Date :=
  { month := 11, day := 3 }

theorem samuel_coaching_discontinue_date 
  (totalCost : Nat) 
  (dailyCharge : Nat) 
  (nonLeapYear : Bool) :
  totalCost = 7038 →
  dailyCharge = 23 →
  nonLeapYear = true →
  daysFromNewYear discontinueDate = totalCost / dailyCharge :=
by sorry

end samuel_coaching_discontinue_date_l1624_162491


namespace largest_integer_negative_quadratic_l1624_162403

theorem largest_integer_negative_quadratic :
  ∃ (n : ℤ), n^2 - 13*n + 40 < 0 ∧
  ∀ (m : ℤ), m^2 - 13*m + 40 < 0 → m ≤ 7 :=
by sorry

end largest_integer_negative_quadratic_l1624_162403


namespace game_ends_in_six_rounds_l1624_162405

/-- Represents a player in the token game -/
inductive Player : Type
| A
| B
| C

/-- Represents the state of the game at any given round -/
structure GameState :=
  (tokens : Player → ℕ)

/-- Determines if the game has ended (any player has 0 tokens) -/
def game_ended (state : GameState) : Prop :=
  ∃ p : Player, state.tokens p = 0

/-- Simulates one round of the game -/
def play_round (state : GameState) : GameState :=
  sorry

/-- The initial state of the game -/
def initial_state : GameState :=
  { tokens := λ p => match p with
    | Player.A => 16
    | Player.B => 14
    | Player.C => 12 }

/-- Theorem stating that the game ends after exactly 6 rounds -/
theorem game_ends_in_six_rounds :
  let final_state := (play_round^[6]) initial_state
  game_ended final_state ∧ ¬game_ended ((play_round^[5]) initial_state) :=
sorry

end game_ends_in_six_rounds_l1624_162405


namespace friend_team_assignment_l1624_162440

theorem friend_team_assignment (n : ℕ) (k : ℕ) (h1 : n = 8) (h2 : k = 4) :
  k ^ n = 65536 :=
sorry

end friend_team_assignment_l1624_162440


namespace one_real_root_condition_l1624_162437

theorem one_real_root_condition (a : ℝ) : 
  (∃! x : ℝ, Real.sqrt (a * x^2 + a * x + 2) = a * x + 2) ↔ 
  (a = -8 ∨ a ≥ 1) := by
sorry

end one_real_root_condition_l1624_162437


namespace min_h_12_l1624_162448

/-- A function h : ℕ+ → ℤ is quibbling if h(x) + h(y) ≥ x^2 + 10*y for all positive integers x and y -/
def IsQuibbling (h : ℕ+ → ℤ) : Prop :=
  ∀ x y : ℕ+, h x + h y ≥ x^2 + 10*y

/-- The sum of h(1) to h(15) -/
def SumH (h : ℕ+ → ℤ) : ℤ :=
  (Finset.range 15).sum (fun i => h ⟨i + 1, Nat.succ_pos i⟩)

theorem min_h_12 (h : ℕ+ → ℤ) (hQuib : IsQuibbling h) (hMin : ∀ g : ℕ+ → ℤ, IsQuibbling g → SumH g ≥ SumH h) :
  h ⟨12, by norm_num⟩ ≥ 144 := by
  sorry


end min_h_12_l1624_162448


namespace diophantine_equation_solutions_l1624_162457

def solution_set : Set (ℕ × ℕ × ℕ) :=
  {(5, 0, 4), (4, 1, 4), (3, 2, 4), (2, 3, 4), (1, 4, 4), (0, 5, 4),
   (3, 0, 0), (2, 1, 0), (1, 2, 0), (0, 3, 0)}

theorem diophantine_equation_solutions :
  {(x, y, z) : ℕ × ℕ × ℕ | x^2 + y^2 - z^2 = 9 - 2*x*y} = solution_set :=
by sorry

end diophantine_equation_solutions_l1624_162457


namespace min_packs_for_130_cans_l1624_162422

/-- Represents the number of cans in each pack type -/
inductive PackSize
  | small : PackSize
  | medium : PackSize
  | large : PackSize

/-- Returns the number of cans for a given pack size -/
def cansInPack (p : PackSize) : ℕ :=
  match p with
  | .small => 8
  | .medium => 15
  | .large => 30

/-- Represents a combination of packs -/
structure PackCombination where
  small : ℕ
  medium : ℕ
  large : ℕ

/-- Calculates the total number of cans in a pack combination -/
def totalCans (c : PackCombination) : ℕ :=
  c.small * cansInPack PackSize.small +
  c.medium * cansInPack PackSize.medium +
  c.large * cansInPack PackSize.large

/-- Calculates the total number of packs in a combination -/
def totalPacks (c : PackCombination) : ℕ :=
  c.small + c.medium + c.large

/-- Theorem: The minimum number of packs to buy 130 cans is 6 -/
theorem min_packs_for_130_cans :
  ∃ (c : PackCombination),
    totalCans c = 130 ∧
    totalPacks c = 6 ∧
    (∀ (d : PackCombination), totalCans d = 130 → totalPacks d ≥ 6) :=
by sorry

end min_packs_for_130_cans_l1624_162422


namespace parallel_vectors_x_value_l1624_162433

theorem parallel_vectors_x_value (x : ℝ) :
  let a : Fin 2 → ℝ := ![x, 1]
  let b : Fin 2 → ℝ := ![1, -1]
  (∃ (k : ℝ), a = k • b) → x = -1 := by
  sorry

end parallel_vectors_x_value_l1624_162433


namespace pet_store_bird_count_l1624_162462

theorem pet_store_bird_count :
  ∀ (num_cages : ℕ) (parrots_per_cage : ℕ) (parakeets_per_cage : ℕ),
    num_cages = 8 →
    parrots_per_cage = 2 →
    parakeets_per_cage = 7 →
    num_cages * (parrots_per_cage + parakeets_per_cage) = 72 :=
by
  sorry

end pet_store_bird_count_l1624_162462


namespace intersection_points_count_l1624_162482

-- Define the two equations
def equation1 (x y : ℝ) : Prop := (x - 2*y + 3) * (4*x + y - 5) = 0
def equation2 (x y : ℝ) : Prop := (x + 2*y - 3) * (3*x - 4*y + 6) = 0

-- Define a point as a pair of real numbers
def Point := ℝ × ℝ

-- Define what it means for a point to satisfy both equations
def satisfiesBothEquations (p : Point) : Prop :=
  equation1 p.1 p.2 ∧ equation2 p.1 p.2

-- Statement of the theorem
theorem intersection_points_count :
  ∃ (s : Finset Point), (∀ p ∈ s, satisfiesBothEquations p) ∧ s.card = 3 ∧
  (∀ p : Point, satisfiesBothEquations p → p ∈ s) := by
  sorry

end intersection_points_count_l1624_162482


namespace angle_at_intersection_point_l1624_162483

/-- In a 3x3 grid, given points A, B, C, D, and E where AB and CD intersect at E, 
    prove that the angle at E is 45 degrees. -/
theorem angle_at_intersection_point (A B C D E : ℝ × ℝ) : 
  A = (0, 0) → 
  B = (3, 3) → 
  C = (0, 3) → 
  D = (3, 0) → 
  (E.1 - A.1) / (E.2 - A.2) = (B.1 - A.1) / (B.2 - A.2) →  -- E is on line AB
  (E.1 - C.1) / (E.2 - C.2) = (D.1 - C.1) / (D.2 - C.2) →  -- E is on line CD
  Real.arctan ((B.2 - A.2) / (B.1 - A.1) - (D.2 - C.2) / (D.1 - C.1)) / 
    (1 + (B.2 - A.2) / (B.1 - A.1) * (D.2 - C.2) / (D.1 - C.1)) * (180 / Real.pi) = 45 :=
by sorry

end angle_at_intersection_point_l1624_162483


namespace chord_squared_sum_l1624_162454

-- Define the circle and points
def Circle : Type := Unit
def Point : Type := Unit
def O : Point := Unit.unit -- Center of the circle
def A : Point := Unit.unit
def B : Point := Unit.unit
def C : Point := Unit.unit
def D : Point := Unit.unit
def E : Point := Unit.unit

-- Define the radius of the circle
def radius : ℝ := 6

-- Define the distance function
def distance : Point → Point → ℝ := sorry

-- Define the angle function
def angle : Point → Point → Point → ℝ := sorry

-- State the theorem
theorem chord_squared_sum (c : Circle) :
  distance O A = radius ∧
  distance O B = radius ∧
  distance A B = 2 * radius ∧
  distance B E = 3 ∧
  angle A E C = π / 3 →
  (distance C E)^2 + (distance D E)^2 = 108 := by
  sorry

end chord_squared_sum_l1624_162454


namespace intersection_line_equation_l1624_162423

-- Define the two circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 + 4*x - 4*y = 0
def circle2 (x y : ℝ) : Prop := x^2 + y^2 + 2*x - 12 = 0

-- Define the line
def line (x y : ℝ) : Prop := x - 2*y + 6 = 0

-- Theorem statement
theorem intersection_line_equation :
  ∃ (A B : ℝ × ℝ),
    (A.1 ≠ B.1 ∨ A.2 ≠ B.2) ∧
    circle1 A.1 A.2 ∧ circle1 B.1 B.2 ∧
    circle2 A.1 A.2 ∧ circle2 B.1 B.2 ∧
    (∀ (x y : ℝ), circle1 x y ∧ circle2 x y → line x y) :=
by sorry

end intersection_line_equation_l1624_162423


namespace tangent_parallel_to_x_axis_l1624_162446

noncomputable def f (x : ℝ) : ℝ := (Real.log x) / x

theorem tangent_parallel_to_x_axis :
  ∃ (x₀ : ℝ), x₀ > 0 ∧ (deriv f x₀ = 0) ∧ (f x₀ = 1 / Real.exp 1) := by
  sorry

end tangent_parallel_to_x_axis_l1624_162446


namespace inequality_proofs_l1624_162473

theorem inequality_proofs (x : ℝ) :
  (x^2 - x - 2 ≥ 0 ∧ Real.sqrt (x^2 - x - 2) ≤ 2*x → x ≥ 2) ∧
  (x^2 - x - 2 ≥ 0 ∧ Real.sqrt (x^2 - x - 2) ≥ 2*x → x ≤ -1) := by
  sorry

end inequality_proofs_l1624_162473


namespace right_triangle_height_l1624_162410

theorem right_triangle_height (base height hypotenuse : ℝ) : 
  base = 4 →
  base + height + hypotenuse = 12 →
  base^2 + height^2 = hypotenuse^2 →
  height = 3 := by
sorry

end right_triangle_height_l1624_162410


namespace periodic_decimal_sum_l1624_162475

/-- The sum of 0.3̅, 0.0̅4̅, and 0.0̅0̅5̅ is equal to 14/37 -/
theorem periodic_decimal_sum : 
  (1 / 3 : ℚ) + (4 / 99 : ℚ) + (5 / 999 : ℚ) = 14 / 37 := by sorry

end periodic_decimal_sum_l1624_162475


namespace billys_restaurant_bill_l1624_162420

/-- The total bill for a group at Billy's Restaurant -/
def total_bill (adults children meal_cost : ℕ) : ℕ :=
  (adults + children) * meal_cost

/-- Theorem: The total bill for a group of 2 adults and 5 children, 
    with each meal costing 3 dollars, is 21 dollars -/
theorem billys_restaurant_bill : total_bill 2 5 3 = 21 := by
  sorry

end billys_restaurant_bill_l1624_162420


namespace min_spheres_to_cover_unit_cylinder_l1624_162429

/-- Represents a cylinder with given height and base radius -/
structure Cylinder where
  height : ℝ
  baseRadius : ℝ

/-- Represents a sphere with given radius -/
structure Sphere where
  radius : ℝ

/-- Function to determine the minimum number of spheres needed to cover a cylinder -/
def minSpheresToCoverCylinder (c : Cylinder) (s : Sphere) : ℕ :=
  sorry

/-- Theorem stating that a cylinder with height 1 and base radius 1 requires at least 3 unit spheres to cover it -/
theorem min_spheres_to_cover_unit_cylinder :
  let c := Cylinder.mk 1 1
  let s := Sphere.mk 1
  minSpheresToCoverCylinder c s = 3 :=
sorry

end min_spheres_to_cover_unit_cylinder_l1624_162429


namespace binomial_coefficient_problem_l1624_162463

theorem binomial_coefficient_problem (h1 : Nat.choose 18 11 = 31824)
                                     (h2 : Nat.choose 18 12 = 18564)
                                     (h3 : Nat.choose 20 13 = 77520) :
  Nat.choose 19 13 = 27132 := by
  sorry

end binomial_coefficient_problem_l1624_162463


namespace root_in_interval_l1624_162459

noncomputable def f (x : ℝ) := Real.exp x + x - 2

theorem root_in_interval : ∃ x ∈ Set.Ioo 0 1, f x = 0 := by
  sorry

end root_in_interval_l1624_162459


namespace profit_formula_l1624_162428

/-- Represents the cost and pricing structure of a shop selling bundles -/
structure ShopBundle where
  water_cost : ℚ  -- Cost of water bottle in dollars
  fruit_cost : ℚ  -- Cost of fruit in dollars
  snack_cost : ℚ  -- Cost of snack in dollars (unknown)
  regular_price : ℚ  -- Regular selling price of a bundle
  fifth_bundle_price : ℚ  -- Price of every 5th bundle
  water_per_bundle : ℕ  -- Number of water bottles per bundle
  fruit_per_bundle : ℕ  -- Number of fruits per bundle
  snack_per_bundle : ℕ  -- Number of snacks per regular bundle
  extra_snack : ℕ  -- Extra snacks given in 5th bundle

/-- Calculates the total profit for 5 bundles given the shop's pricing structure -/
def total_profit_five_bundles (shop : ShopBundle) : ℚ :=
  let regular_cost := shop.water_cost * shop.water_per_bundle +
                      shop.fruit_cost * shop.fruit_per_bundle +
                      shop.snack_cost * shop.snack_per_bundle
  let fifth_bundle_cost := shop.water_cost * shop.water_per_bundle +
                           shop.fruit_cost * shop.fruit_per_bundle +
                           shop.snack_cost * (shop.snack_per_bundle + shop.extra_snack)
  let regular_profit := shop.regular_price - regular_cost
  let fifth_bundle_profit := shop.fifth_bundle_price - fifth_bundle_cost
  4 * regular_profit + fifth_bundle_profit

/-- Theorem stating that the total profit for 5 bundles can be expressed as 15.40 - 16S -/
theorem profit_formula (shop : ShopBundle)
  (h1 : shop.water_cost = 0.5)
  (h2 : shop.fruit_cost = 0.25)
  (h3 : shop.regular_price = 4.6)
  (h4 : shop.fifth_bundle_price = 2)
  (h5 : shop.water_per_bundle = 1)
  (h6 : shop.fruit_per_bundle = 2)
  (h7 : shop.snack_per_bundle = 3)
  (h8 : shop.extra_snack = 1) :
  total_profit_five_bundles shop = 15.4 - 16 * shop.snack_cost := by
  sorry

end profit_formula_l1624_162428


namespace baseball_card_value_decrease_l1624_162470

theorem baseball_card_value_decrease : 
  ∀ (initial_value : ℝ), initial_value > 0 →
  let value_after_first_year := initial_value * (1 - 0.1)
  let value_after_second_year := value_after_first_year * (1 - 0.1)
  let total_decrease := initial_value - value_after_second_year
  let percent_decrease := (total_decrease / initial_value) * 100
  percent_decrease = 19 := by
sorry

end baseball_card_value_decrease_l1624_162470


namespace right_triangle_perimeter_l1624_162451

theorem right_triangle_perimeter : ∃ (a b c : ℕ), 
  a > 0 ∧ b > 0 ∧ c > 0 ∧  -- all sides are positive integers
  b = 4 ∧                   -- one leg measures 4
  a^2 + b^2 = c^2 ∧         -- right-angled triangle (Pythagorean theorem)
  a + b + c = 12            -- perimeter is 12
  := by sorry

end right_triangle_perimeter_l1624_162451


namespace robs_planned_reading_time_l1624_162434

/-- Proves that Rob's planned reading time was 3 hours given the conditions -/
theorem robs_planned_reading_time 
  (pages_read : ℕ) 
  (reading_rate : ℚ)  -- pages per minute
  (actual_time_ratio : ℚ) :
  pages_read = 9 →
  reading_rate = 1 / 15 →
  actual_time_ratio = 3 / 4 →
  (pages_read / reading_rate) / actual_time_ratio / 60 = 3 := by
sorry

end robs_planned_reading_time_l1624_162434


namespace jan_math_problem_l1624_162456

-- Define the operation of rounding to the nearest ten
def roundToNearestTen (x : ℤ) : ℤ :=
  10 * ((x + 5) / 10)

-- Theorem statement
theorem jan_math_problem :
  roundToNearestTen (83 - 29 + 58) = 110 := by
  sorry

end jan_math_problem_l1624_162456


namespace chocolate_cost_proof_l1624_162431

def candy_bar_cost : ℝ := 2
def chocolate_cost_difference : ℝ := 1

theorem chocolate_cost_proof :
  let chocolate_cost := candy_bar_cost + chocolate_cost_difference
  chocolate_cost = 3 := by sorry

end chocolate_cost_proof_l1624_162431


namespace unique_triple_existence_l1624_162425

theorem unique_triple_existence : 
  ∃! (a b c : ℝ), a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ 
  (1 / a = b + c) ∧ (1 / b = a + c) ∧ (1 / c = a + b) := by
sorry

end unique_triple_existence_l1624_162425


namespace trigonometric_identity_l1624_162408

theorem trigonometric_identity (α : ℝ) :
  2 * (Real.sin (3 * π - 2 * α))^2 * (Real.cos (5 * π + 2 * α))^2 =
  1/4 - 1/4 * Real.sin (5/2 * π - 8 * α) := by
  sorry

end trigonometric_identity_l1624_162408


namespace square_coverage_l1624_162443

theorem square_coverage (unit_square_area : ℝ) (large_square_side : ℝ) :
  unit_square_area = 1 →
  large_square_side = 5 / 4 →
  3 * unit_square_area ≥ large_square_side ^ 2 :=
by
  sorry

end square_coverage_l1624_162443


namespace shop_profit_calculation_l1624_162468

/-- The amount the shop makes off each jersey -/
def jersey_profit : ℕ := 210

/-- The additional cost of a t-shirt compared to a jersey -/
def tshirt_additional_cost : ℕ := 30

/-- The amount the shop makes off each t-shirt -/
def tshirt_profit : ℕ := jersey_profit + tshirt_additional_cost

theorem shop_profit_calculation :
  tshirt_profit = 240 :=
by sorry

end shop_profit_calculation_l1624_162468


namespace geometric_progression_ratio_l1624_162465

/-- For a geometric progression with first term b₁, common ratio q, n-th term bₙ, and sum of first n terms Sₙ, 
    the ratio (Sₙ - bₙ) / (Sₙ - b₁) is equal to 1/q for all q -/
theorem geometric_progression_ratio (n : ℕ) (b₁ q : ℝ) : 
  let bₙ := b₁ * q^(n - 1)
  let Sₙ := if q ≠ 1 then b₁ * (q^n - 1) / (q - 1) else n * b₁
  (Sₙ - bₙ) / (Sₙ - b₁) = 1 / q :=
by sorry

end geometric_progression_ratio_l1624_162465


namespace smallest_five_digit_multiple_l1624_162419

theorem smallest_five_digit_multiple : ∃ n : ℕ, 
  (n ≥ 10000 ∧ n < 100000) ∧  -- five-digit number
  n % 9 = 0 ∧                 -- multiple of 9
  n % 6 = 0 ∧                 -- multiple of 6
  n % 2 = 0 ∧                 -- multiple of 2
  (∀ m : ℕ, 
    (m ≥ 10000 ∧ m < 100000) ∧ 
    m % 9 = 0 ∧ 
    m % 6 = 0 ∧ 
    m % 2 = 0 → 
    n ≤ m) ∧
  n = 10008 := by
sorry

end smallest_five_digit_multiple_l1624_162419


namespace arithmetic_sequence_property_l1624_162455

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- Theorem: In an arithmetic sequence, if a_3 + a_5 = 12, then a_4 = 6 -/
theorem arithmetic_sequence_property (a : ℕ → ℝ) 
    (h_arith : ArithmeticSequence a) (h_sum : a 3 + a 5 = 12) : 
    a 4 = 6 := by
  sorry

end arithmetic_sequence_property_l1624_162455


namespace cookies_per_bag_l1624_162435

theorem cookies_per_bag (total_cookies : ℕ) (num_bags : ℕ) (h1 : total_cookies = 75) (h2 : num_bags = 25) :
  total_cookies / num_bags = 3 := by
  sorry

end cookies_per_bag_l1624_162435


namespace complex_multiplication_l1624_162472

theorem complex_multiplication (i : ℂ) : i^2 = -1 → (2 + i) * (3 + i) = 5 + 5*i := by
  sorry

end complex_multiplication_l1624_162472


namespace range_of_roots_difference_l1624_162432

-- Define the function g
def g (a b c d x : ℝ) : ℝ := a * x^3 + b * x^2 + c * x + d

-- Define the derivative of g as f
def f (a b c : ℝ) (x : ℝ) : ℝ := 3 * a * x^2 + 2 * b * x + c

-- State the theorem
theorem range_of_roots_difference
  (a b c d : ℝ)
  (ha : a ≠ 0)
  (hsum : a + 2 * b + 3 * c = 0)
  (hpos : f a b c 0 * f a b c 1 > 0)
  (x₁ x₂ : ℝ)
  (hroot₁ : f a b c x₁ = 0)
  (hroot₂ : f a b c x₂ = 0) :
  ∃ y, y ∈ Set.Icc 0 (2/3) ∧ |x₁ - x₂| = y :=
sorry

end range_of_roots_difference_l1624_162432


namespace comic_book_stacking_order_l1624_162424

theorem comic_book_stacking_order :
  let spiderman_comics := 7
  let archie_comics := 6
  let garfield_comics := 5
  let group_arrangements := 3
  (spiderman_comics.factorial * archie_comics.factorial * garfield_comics.factorial * group_arrangements.factorial) = 248832000 := by
  sorry

end comic_book_stacking_order_l1624_162424


namespace probability_of_valid_pair_l1624_162467

def ball_numbers : Finset ℕ := {1, 2, 3, 4, 5, 6, 7}

def is_valid_pair (x y : ℕ) : Bool :=
  x ∈ ball_numbers ∧ y ∈ ball_numbers ∧ Even (x * y) ∧ x * y > 14

def valid_pairs : Finset (ℕ × ℕ) :=
  Finset.filter (λ p => is_valid_pair p.1 p.2) (ball_numbers.product ball_numbers)

theorem probability_of_valid_pair :
  (valid_pairs.card : ℚ) / (ball_numbers.card * ball_numbers.card : ℚ) = 16 / 49 := by
  sorry

end probability_of_valid_pair_l1624_162467


namespace max_planes_for_10_points_l1624_162460

/-- The number of points in space -/
def n : ℕ := 10

/-- The number of points required to determine a plane -/
def k : ℕ := 3

/-- Assumption that no three points are collinear -/
axiom no_collinear : True

/-- The maximum number of planes determined by n points in space -/
def max_planes (n : ℕ) : ℕ := Nat.choose n k

theorem max_planes_for_10_points : max_planes n = 120 := by sorry

end max_planes_for_10_points_l1624_162460


namespace coefficient_x_squared_expansion_l1624_162471

theorem coefficient_x_squared_expansion : 
  let f : ℕ → ℕ → ℕ := fun n k => Nat.choose n k
  let g : ℕ → ℤ := fun n => (-1)^n
  (f 3 0) * (f 4 2) + (f 3 1) * (f 4 1) * (g 1) + (f 3 2) * 2^2 * (f 4 0) = -6 :=
by sorry

end coefficient_x_squared_expansion_l1624_162471


namespace gold_coin_percentage_is_55_25_percent_l1624_162430

/-- Represents the composition of objects in an urn -/
structure UrnComposition where
  bead_percent : Real
  gold_coin_percent_of_coins : Real

/-- Calculates the percentage of gold coins in the urn -/
def gold_coin_percent (urn : UrnComposition) : Real :=
  (1 - urn.bead_percent) * urn.gold_coin_percent_of_coins

/-- Theorem stating that the percentage of gold coins in the urn is 55.25% -/
theorem gold_coin_percentage_is_55_25_percent 
  (urn : UrnComposition) 
  (h1 : urn.bead_percent = 0.15) 
  (h2 : urn.gold_coin_percent_of_coins = 0.65) : 
  gold_coin_percent urn = 0.5525 := by
  sorry

#eval gold_coin_percent { bead_percent := 0.15, gold_coin_percent_of_coins := 0.65 }

end gold_coin_percentage_is_55_25_percent_l1624_162430


namespace average_of_last_three_l1624_162416

theorem average_of_last_three (numbers : Fin 6 → ℝ) 
  (h1 : (numbers 0 + numbers 1 + numbers 2 + numbers 3 + numbers 4 + numbers 5) / 6 = 30)
  (h2 : (numbers 0 + numbers 1 + numbers 2 + numbers 3) / 4 = 25)
  (h3 : numbers 3 = 25) :
  (numbers 3 + numbers 4 + numbers 5) / 3 = 35 := by
sorry

end average_of_last_three_l1624_162416


namespace floor_expression_equals_eight_l1624_162412

def n : ℕ := 1004

theorem floor_expression_equals_eight :
  ⌊(1005^3 : ℚ) / (1003 * 1004) - (1003^3 : ℚ) / (1004 * 1005)⌋ = 8 := by
  sorry

end floor_expression_equals_eight_l1624_162412


namespace parallel_vectors_magnitude_l1624_162487

/-- Given vectors a and b in ℝ³, where a is parallel to b, prove that the magnitude of b is 3√6 -/
theorem parallel_vectors_magnitude (a b : ℝ × ℝ × ℝ) : 
  a = (-1, 2, 1) → 
  b.1 = 3 → 
  ∃ (k : ℝ), b = k • a → 
  ‖b‖ = 3 * Real.sqrt 6 := by
  sorry


end parallel_vectors_magnitude_l1624_162487


namespace remainder_div_nine_l1624_162477

theorem remainder_div_nine (n : ℕ) (h : n % 18 = 11) : n % 9 = 2 := by
  sorry

end remainder_div_nine_l1624_162477


namespace average_of_numbers_is_ten_l1624_162418

def numbers : List ℝ := [6, 8, 9, 11, 16]

theorem average_of_numbers_is_ten :
  (List.sum numbers) / (List.length numbers) = 10 := by
  sorry

end average_of_numbers_is_ten_l1624_162418


namespace complex_3_minus_i_in_fourth_quadrant_l1624_162486

/-- A complex number z is in the fourth quadrant if its real part is positive and its imaginary part is negative -/
def in_fourth_quadrant (z : ℂ) : Prop := 0 < z.re ∧ z.im < 0

/-- The complex number 3 - i is in the fourth quadrant -/
theorem complex_3_minus_i_in_fourth_quadrant : 
  in_fourth_quadrant (3 - I) := by sorry

end complex_3_minus_i_in_fourth_quadrant_l1624_162486


namespace hill_height_correct_l1624_162479

/-- The height of the hill in feet -/
def hill_height : ℝ := 900

/-- The uphill speed in feet per second -/
def uphill_speed : ℝ := 9

/-- The downhill speed in feet per second -/
def downhill_speed : ℝ := 12

/-- The total time to run up and down the hill in seconds -/
def total_time : ℝ := 175

/-- Theorem stating that the given hill height satisfies the conditions -/
theorem hill_height_correct : 
  hill_height / uphill_speed + hill_height / downhill_speed = total_time :=
sorry

end hill_height_correct_l1624_162479


namespace fraction_change_with_addition_l1624_162492

theorem fraction_change_with_addition (a b n : ℕ) (h_b_pos : b > 0) :
  (a / b < 1 → (a + n) / (b + n) > a / b) ∧
  (a / b > 1 → (a + n) / (b + n) < a / b) := by
sorry

end fraction_change_with_addition_l1624_162492


namespace candy_inconsistency_l1624_162453

theorem candy_inconsistency :
  ¬∃ (K Y N B : ℕ),
    K + Y + N = 120 ∧
    N + B = 103 ∧
    K + Y + B = 152 :=
by sorry

end candy_inconsistency_l1624_162453


namespace impossible_cover_l1624_162426

/-- Represents a rectangular board -/
structure Board :=
  (rows : ℕ)
  (cols : ℕ)

/-- Represents a rectangular piece -/
structure Piece :=
  (width : ℕ)
  (height : ℕ)

/-- Checks if a board can be completely covered by pieces without overlapping or sticking out -/
def can_cover (b : Board) (p : Piece) : Prop :=
  ∃ (arrangement : ℕ), 
    (arrangement * p.width * p.height = b.rows * b.cols) ∧ 
    (b.rows % p.height = 0) ∧ 
    (b.cols % p.width = 0)

/-- The main theorem stating that specific boards cannot be covered by specific pieces -/
theorem impossible_cover : 
  ¬(can_cover (Board.mk 6 6) (Piece.mk 1 4)) ∧ 
  ¬(can_cover (Board.mk 12 9) (Piece.mk 2 2)) :=
sorry

end impossible_cover_l1624_162426


namespace trigonometric_equation_l1624_162421

theorem trigonometric_equation (x : Real) :
  2 * Real.cos x - 3 * Real.sin x = 2 →
  Real.sin x + 3 * Real.cos x = 3 ∨ Real.sin x + 3 * Real.cos x = -31/13 := by
  sorry

end trigonometric_equation_l1624_162421


namespace sum_coordinates_reflection_over_x_axis_l1624_162439

/-- A point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Reflect a point over the x-axis -/
def reflectOverXAxis (p : Point2D) : Point2D :=
  { x := p.x, y := -p.y }

/-- Sum of all coordinate values of two points -/
def sumCoordinates (p1 p2 : Point2D) : ℝ :=
  p1.x + p1.y + p2.x + p2.y

/-- Theorem: The sum of coordinates of a point (5, y) and its reflection over x-axis is 10 -/
theorem sum_coordinates_reflection_over_x_axis (y : ℝ) :
  let c : Point2D := { x := 5, y := y }
  let d : Point2D := reflectOverXAxis c
  sumCoordinates c d = 10 := by
  sorry

end sum_coordinates_reflection_over_x_axis_l1624_162439


namespace custom_operation_equation_l1624_162495

-- Define the custom operation *
def star (a b : ℝ) : ℝ := 4 * a - b

-- State the theorem
theorem custom_operation_equation :
  ∃ x : ℝ, (star 4 (star 3 x) = 2) ∧ (x = -2) := by sorry

end custom_operation_equation_l1624_162495


namespace arithmetic_sequence_constant_multiple_l1624_162415

/-- An arithmetic sequence with common difference d -/
def ArithmeticSequence (a : ℕ → ℝ) (d : ℝ) :=
  ∀ n : ℕ, a (n + 1) - a n = d

/-- The constant multiple of a sequence -/
def ConstantMultiple (a : ℕ → ℝ) (c : ℝ) : ℕ → ℝ :=
  fun n => c * a n

theorem arithmetic_sequence_constant_multiple
  (a : ℕ → ℝ) (d c : ℝ) (hc : c ≠ 0) (ha : ArithmeticSequence a d) :
  ArithmeticSequence (ConstantMultiple a c) (c * d) := by
sorry

end arithmetic_sequence_constant_multiple_l1624_162415


namespace newspaper_photos_l1624_162436

/-- The total number of photos in a newspaper with specified page types -/
def total_photos (pages_with_two_photos pages_with_three_photos : ℕ) : ℕ :=
  2 * pages_with_two_photos + 3 * pages_with_three_photos

/-- Theorem stating that the total number of photos in the newspaper is 51 -/
theorem newspaper_photos : total_photos 12 9 = 51 := by
  sorry

end newspaper_photos_l1624_162436


namespace certain_number_proof_l1624_162498

theorem certain_number_proof : ∃! x : ℕ, (x - 16) % 37 = 0 ∧ (x - 16) / 37 = 23 := by
  sorry

end certain_number_proof_l1624_162498


namespace quarterback_sacks_l1624_162450

theorem quarterback_sacks (total_attempts : ℕ) (no_throw_percentage : ℚ) (sack_ratio : ℚ) :
  total_attempts = 80 →
  no_throw_percentage = 30 / 100 →
  sack_ratio = 1 / 2 →
  ↑(total_attempts : ℕ) * no_throw_percentage * sack_ratio = 12 :=
by sorry

end quarterback_sacks_l1624_162450


namespace number_of_boys_l1624_162445

theorem number_of_boys (total : ℕ) (boys : ℕ) (girls : ℕ) : 
  total = 100 → 
  boys + girls = total → 
  girls = boys → 
  boys = 50 := by
sorry

end number_of_boys_l1624_162445


namespace sum_in_terms_of_x_l1624_162481

theorem sum_in_terms_of_x (x y z : ℝ) (h1 : y = 3 * x) (h2 : z = 4 * y) :
  x + y + z = 16 * x := by
sorry

end sum_in_terms_of_x_l1624_162481


namespace value_of_x_l1624_162444

theorem value_of_x (w u v x : ℤ) 
  (hw : w = 50)
  (hv : v = 3 * w + 30)
  (hu : u = v - 15)
  (hx : x = 2 * u + 12) : 
  x = 342 := by
  sorry

end value_of_x_l1624_162444


namespace angle_Q_is_90_degrees_l1624_162494

/-- A regular dodecagon with vertices ABCDEFGHIJKL -/
structure RegularDodecagon where
  vertices : Fin 12 → Point

/-- The point Q where extended sides AL and FG meet -/
def Q (d : RegularDodecagon) : Point := sorry

/-- The angle at point Q formed by the extended sides AL and FG -/
def angle_Q (d : RegularDodecagon) : AngularMeasure := sorry

/-- The theorem stating that the measure of angle Q is 90 degrees -/
theorem angle_Q_is_90_degrees (d : RegularDodecagon) : 
  angle_Q d = 90 := by sorry

end angle_Q_is_90_degrees_l1624_162494


namespace fixed_tangent_circle_l1624_162452

-- Define the main circle
def main_circle (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 1

-- Define the property of the chords
def chord_property (OA OB : ℝ) : Prop := OA * OB = 2

-- Define the tangent circle
def tangent_circle (x y : ℝ) : Prop := x^2 + y^2 = 1

-- Theorem statement
theorem fixed_tangent_circle 
  (O A B : ℝ × ℝ) 
  (hA : main_circle A.1 A.2) 
  (hB : main_circle B.1 B.2)
  (hOA : O = (0, 0))
  (hchord : chord_property (Real.sqrt ((A.1 - O.1)^2 + (A.2 - O.2)^2)) 
                           (Real.sqrt ((B.1 - O.1)^2 + (B.2 - O.2)^2)))
  (hAB : A ≠ B) :
  ∃ (P : ℝ × ℝ), tangent_circle P.1 P.2 ∧ 
    (P.1 - A.1) * (B.2 - A.2) = (P.2 - A.2) * (B.1 - A.1) :=
sorry

end fixed_tangent_circle_l1624_162452


namespace modified_cube_surface_area_l1624_162461

/-- Represents the dimensions of a cube -/
structure CubeDimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Represents the dimensions of a corner cube to be removed -/
structure CornerCubeDimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the surface area of the modified cube -/
def modifiedCubeSurfaceArea (originalCube : CubeDimensions) (cornerCube : CornerCubeDimensions) : ℝ :=
  sorry

/-- Theorem stating that the surface area of the modified cube is 96 sq.cm -/
theorem modified_cube_surface_area :
  let originalCube : CubeDimensions := ⟨4, 4, 4⟩
  let cornerCube : CornerCubeDimensions := ⟨1, 1, 1⟩
  modifiedCubeSurfaceArea originalCube cornerCube = 96 := by
  sorry

end modified_cube_surface_area_l1624_162461


namespace edwards_lawn_mowing_earnings_l1624_162409

/-- Edward's lawn mowing business earnings and expenses --/
theorem edwards_lawn_mowing_earnings 
  (spring_earnings : ℕ) 
  (summer_earnings : ℕ) 
  (supplies_cost : ℕ) 
  (h1 : spring_earnings = 2)
  (h2 : summer_earnings = 27)
  (h3 : supplies_cost = 5) :
  spring_earnings + summer_earnings - supplies_cost = 24 :=
by sorry

end edwards_lawn_mowing_earnings_l1624_162409


namespace stratified_sample_science_students_l1624_162493

theorem stratified_sample_science_students 
  (total_students : ℕ) 
  (science_students : ℕ) 
  (sample_size : ℕ) 
  (h1 : total_students = 140) 
  (h2 : science_students = 100) 
  (h3 : sample_size = 14) :
  (sample_size : ℚ) / total_students * science_students = 10 := by
  sorry

end stratified_sample_science_students_l1624_162493


namespace extended_tile_ratio_l1624_162478

theorem extended_tile_ratio (initial_black : ℕ) (initial_white : ℕ) 
  (h1 : initial_black = 7)
  (h2 : initial_white = 18)
  (h3 : initial_black + initial_white = 25) :
  let side_length : ℕ := (initial_black + initial_white).sqrt
  let extended_side_length : ℕ := side_length + 2
  let extended_black : ℕ := initial_black + 4 * side_length + 4
  let extended_white : ℕ := initial_white
  (extended_black : ℚ) / extended_white = 31 / 18 := by
sorry

end extended_tile_ratio_l1624_162478


namespace max_min_f_on_interval_l1624_162427

def f (x : ℝ) := x^3 - 3*x + 1

theorem max_min_f_on_interval :
  ∃ (max min : ℝ),
    (∀ x ∈ Set.Icc (-3) 0, f x ≤ max) ∧
    (∃ x ∈ Set.Icc (-3) 0, f x = max) ∧
    (∀ x ∈ Set.Icc (-3) 0, min ≤ f x) ∧
    (∃ x ∈ Set.Icc (-3) 0, f x = min) ∧
    max = 3 ∧ min = -17 :=
by sorry

end max_min_f_on_interval_l1624_162427


namespace not_p_sufficient_not_necessary_for_not_q_l1624_162480

-- Define the conditions
def p (x : ℝ) : Prop := (x - 1) / (x + 3) ≥ 0
def q (x : ℝ) : Prop := 5 * x - 6 > x^2

-- Define the relationship between ¬p and ¬q
def sufficient_not_necessary (P Q : Prop) : Prop :=
  (¬P → ¬Q) ∧ ¬(¬Q → ¬P)

-- Theorem statement
theorem not_p_sufficient_not_necessary_for_not_q :
  sufficient_not_necessary (∃ x, p x) (∃ x, q x) := by
  sorry

end not_p_sufficient_not_necessary_for_not_q_l1624_162480


namespace find_n_l1624_162485

def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

theorem find_n : ∃ n : ℕ, n * factorial (n + 1) + factorial (n + 1) = 5040 ∧ n = 5 := by
  sorry

end find_n_l1624_162485


namespace helly_theorem_2d_l1624_162406

-- Define a type for points in the plane
variable (Point : Type)

-- Define a type for convex sets in the plane
variable (ConvexSet : Type)

-- Define a function to check if a point is in a convex set
variable (isIn : Point → ConvexSet → Prop)

-- Define a function to check if a set is convex
variable (isConvex : ConvexSet → Prop)

-- Define the theorem
theorem helly_theorem_2d 
  (n : ℕ) 
  (h_n : n ≥ 4) 
  (A : Fin n → ConvexSet) 
  (h_convex : ∀ i, isConvex (A i)) 
  (h_intersection : ∀ i j k, ∃ p, isIn p (A i) ∧ isIn p (A j) ∧ isIn p (A k)) :
  ∃ p, ∀ i, isIn p (A i) :=
sorry

end helly_theorem_2d_l1624_162406


namespace class_grade_average_l1624_162484

theorem class_grade_average (N : ℕ) (X : ℝ) : 
  (X * N + 45 * (2 * N)) / (3 * N) = 48 → X = 54 := by
  sorry

end class_grade_average_l1624_162484


namespace parabola_focus_l1624_162441

/-- The focus of the parabola y² = -8x is at the point (-2, 0) -/
theorem parabola_focus (x y : ℝ) : 
  y^2 = -8*x → (x + 2)^2 + y^2 = 4 :=
by sorry

end parabola_focus_l1624_162441


namespace parking_lot_cars_l1624_162490

theorem parking_lot_cars (initial_cars : ℕ) (cars_left : ℕ) (extra_cars_entered : ℕ) :
  initial_cars = 80 →
  cars_left = 13 →
  extra_cars_entered = 5 →
  initial_cars - cars_left + (cars_left + extra_cars_entered) = 85 :=
by
  sorry

end parking_lot_cars_l1624_162490


namespace average_rounds_is_three_l1624_162488

/-- Represents the distribution of rounds played by golfers -/
structure GolfDistribution where
  rounds1 : Nat
  rounds2 : Nat
  rounds3 : Nat
  rounds4 : Nat
  rounds5 : Nat

/-- Calculates the average number of rounds played, rounded to the nearest whole number -/
def averageRounds (dist : GolfDistribution) : Nat :=
  let totalRounds := dist.rounds1 * 1 + dist.rounds2 * 2 + dist.rounds3 * 3 + dist.rounds4 * 4 + dist.rounds5 * 5
  let totalGolfers := dist.rounds1 + dist.rounds2 + dist.rounds3 + dist.rounds4 + dist.rounds5
  (totalRounds + totalGolfers / 2) / totalGolfers

theorem average_rounds_is_three (dist : GolfDistribution) 
  (h1 : dist.rounds1 = 4)
  (h2 : dist.rounds2 = 3)
  (h3 : dist.rounds3 = 3)
  (h4 : dist.rounds4 = 2)
  (h5 : dist.rounds5 = 6) :
  averageRounds dist = 3 := by
  sorry

#eval averageRounds { rounds1 := 4, rounds2 := 3, rounds3 := 3, rounds4 := 2, rounds5 := 6 }

end average_rounds_is_three_l1624_162488


namespace p_sufficient_not_necessary_for_q_l1624_162414

-- Define the sets M and N
def M : Set ℝ := {x | x^2 - x < 0}
def N : Set ℝ := {x | |x| < 2}

-- Define propositions p and q
def p (a : ℝ) : Prop := a ∈ M
def q (a : ℝ) : Prop := a ∈ N

-- Theorem stating that p is sufficient but not necessary for q
theorem p_sufficient_not_necessary_for_q :
  (∀ a : ℝ, p a → q a) ∧ (∃ a : ℝ, q a ∧ ¬p a) := by sorry

end p_sufficient_not_necessary_for_q_l1624_162414


namespace arithmetic_sequence_sum_l1624_162404

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a b : ℕ → ℝ) :
  ArithmeticSequence a → ArithmeticSequence b →
  (a 1 + b 1 = 7) → (a 3 + b 3 = 21) →
  (a 5 + b 5 = 35) := by
  sorry

end arithmetic_sequence_sum_l1624_162404


namespace parabola_bound_l1624_162411

theorem parabola_bound (a b c : ℝ) :
  (∀ x ∈ Set.Icc (-1 : ℝ) 1, |a * x^2 - b * x + c| < 1) →
  (∀ x ∈ Set.Icc (-1 : ℝ) 1, |(a + b) * x^2 + c| < 1) := by
sorry

end parabola_bound_l1624_162411


namespace basketball_game_theorem_l1624_162496

/-- Represents the scores of a team in a four-quarter basketball game -/
structure GameScores where
  q1 : ℕ
  q2 : ℕ
  q3 : ℕ
  q4 : ℕ

/-- Checks if the given scores form an arithmetic sequence -/
def is_arithmetic (s : GameScores) : Prop :=
  ∃ (a d : ℕ), s.q1 = a ∧ s.q2 = a + d ∧ s.q3 = a + 2*d ∧ s.q4 = a + 3*d

/-- Checks if the given scores form a geometric sequence -/
def is_geometric (s : GameScores) : Prop :=
  ∃ (b r : ℕ), r > 1 ∧ s.q1 = b ∧ s.q2 = b * r ∧ s.q3 = b * r^2 ∧ s.q4 = b * r^3

/-- Calculates the total score for a team -/
def total_score (s : GameScores) : ℕ := s.q1 + s.q2 + s.q3 + s.q4

/-- Calculates the first half score for a team -/
def first_half_score (s : GameScores) : ℕ := s.q1 + s.q2

/-- The main theorem stating the conditions and the result to be proved -/
theorem basketball_game_theorem (team1 team2 : GameScores) : 
  is_arithmetic team1 →
  is_geometric team2 →
  total_score team1 = total_score team2 + 2 →
  total_score team1 ≤ 100 →
  total_score team2 ≤ 100 →
  first_half_score team1 + first_half_score team2 = 30 := by
  sorry

end basketball_game_theorem_l1624_162496


namespace problem_solution_l1624_162407

-- Define the functions f and g
def f (x : ℝ) : ℝ := x^3 - x
def g (x : ℝ) : ℝ := 2*x - 3

-- Define the interval [0, 2]
def interval : Set ℝ := Set.Icc 0 2

theorem problem_solution :
  -- 1. Tangent line equation
  (∃ (m b : ℝ), ∀ x y, y = m*x + b ↔ 2*x - y - 2 = 0) ∧
  (∀ x, x ≠ 1 → (f x - f 1) / (x - 1) < 2) ∧
  (∀ x, x ≠ 1 → (f x - f 1) / (x - 1) > 2) ∧
  
  -- 2. Maximum value on the interval
  (∀ x ∈ interval, f x ≤ 6) ∧
  (∃ x ∈ interval, f x = 6) ∧
  
  -- 3. Existence of unique x₀
  (∃! x₀, f x₀ = g x₀) :=
by sorry

end problem_solution_l1624_162407


namespace range_f_is_closed_interval_l1624_162402

/-- The quadratic function f(x) = -x^2 + 4x + 1 -/
def f (x : ℝ) : ℝ := -x^2 + 4*x + 1

/-- The closed interval [0, 3] -/
def I : Set ℝ := { x | 0 ≤ x ∧ x ≤ 3 }

/-- The range of f over the interval I -/
def range_f : Set ℝ := { y | ∃ x ∈ I, f x = y }

theorem range_f_is_closed_interval :
  range_f = { y | 1 ≤ y ∧ y ≤ 5 } := by sorry

end range_f_is_closed_interval_l1624_162402


namespace water_balloon_packs_l1624_162466

/-- Represents the number of packs of water balloons --/
def num_own_packs : ℕ := 3

/-- Represents the number of balloons in each pack --/
def balloons_per_pack : ℕ := 6

/-- Represents the number of neighbor's packs used --/
def num_neighbor_packs : ℕ := 2

/-- Represents the extra balloons Milly takes --/
def extra_balloons : ℕ := 7

/-- Represents the number of balloons Floretta is left with --/
def floretta_balloons : ℕ := 8

theorem water_balloon_packs :
  num_own_packs * balloons_per_pack + num_neighbor_packs * balloons_per_pack =
  2 * (floretta_balloons + extra_balloons) :=
sorry

end water_balloon_packs_l1624_162466


namespace max_area_of_nonoverlapping_triangle_l1624_162499

/-- A triangle on a coordinate plane -/
structure Triangle where
  vertices : Fin 3 → ℝ × ℝ

/-- Check if two triangles overlap -/
def overlap (t1 t2 : Triangle) : Prop := sorry

/-- Translation of a triangle by an integer vector -/
def translate (t : Triangle) (v : ℤ × ℤ) : Triangle := sorry

/-- Area of a triangle -/
def area (t : Triangle) : ℝ := sorry

/-- A triangle is valid if its translations by integer vectors do not overlap -/
def valid_triangle (t : Triangle) : Prop :=
  ∀ v : ℤ × ℤ, ¬(overlap t (translate t v))

theorem max_area_of_nonoverlapping_triangle :
  ∃ (t : Triangle), valid_triangle t ∧ area t = 2/3 ∧
  ∀ (t' : Triangle), valid_triangle t' → area t' ≤ 2/3 := by sorry

end max_area_of_nonoverlapping_triangle_l1624_162499


namespace quadratic_equation_result_l1624_162400

theorem quadratic_equation_result (y : ℝ) (h : 6 * y^2 + 7 = 2 * y + 10) : 
  (12 * y - 4)^2 = 80 := by
  sorry

end quadratic_equation_result_l1624_162400


namespace min_value_of_f_l1624_162413

theorem min_value_of_f (x : Real) (h : x ∈ Set.Icc (π/4) (5*π/12)) : 
  let f := fun (x : Real) => (Real.sin x)^2 - 2*(Real.cos x)^2 / (Real.sin x * Real.cos x)
  ∃ (m : Real), m = -1 ∧ ∀ y ∈ Set.Icc (π/4) (5*π/12), f y ≥ m := by
  sorry

end min_value_of_f_l1624_162413


namespace problem_solution_l1624_162401

noncomputable def f (x : ℝ) : ℝ := Real.exp x - Real.exp (-x)
noncomputable def g (x : ℝ) : ℝ := Real.exp x + Real.exp (-x)

theorem problem_solution (x y : ℝ) :
  (∀ x, (f x)^2 - (g x)^2 = -4) ∧
  (f x * f y = 4 ∧ g x * g y = 8 → g (x + y) / g (x - y) = 3) :=
by sorry

end problem_solution_l1624_162401


namespace no_super_sudoku_l1624_162442

/-- Represents a 9x9 grid of integers -/
def Grid := Fin 9 → Fin 9 → Fin 9

/-- Checks if a given row contains each number 1-9 exactly once -/
def validRow (g : Grid) (row : Fin 9) : Prop :=
  ∀ n : Fin 9, ∃! col : Fin 9, g row col = n

/-- Checks if a given column contains each number 1-9 exactly once -/
def validColumn (g : Grid) (col : Fin 9) : Prop :=
  ∀ n : Fin 9, ∃! row : Fin 9, g row col = n

/-- Checks if a given 3x3 subsquare contains each number 1-9 exactly once -/
def validSubsquare (g : Grid) (startRow startCol : Fin 3) : Prop :=
  ∀ n : Fin 9, ∃! (row col : Fin 3), g (3 * startRow + row) (3 * startCol + col) = n

/-- Defines a super-sudoku grid -/
def isSuperSudoku (g : Grid) : Prop :=
  (∀ row : Fin 9, validRow g row) ∧
  (∀ col : Fin 9, validColumn g col) ∧
  (∀ startRow startCol : Fin 3, validSubsquare g startRow startCol)

/-- Theorem: There are no possible super-sudoku grids -/
theorem no_super_sudoku : ¬∃ g : Grid, isSuperSudoku g := by
  sorry

end no_super_sudoku_l1624_162442


namespace conference_handshakes_l1624_162476

/-- The number of handshakes in a conference of n people where each person
    shakes hands exactly once with every other person. -/
def handshakes (n : ℕ) : ℕ := n.choose 2

/-- Theorem stating that in a conference of 10 people where each person
    shakes hands exactly once with every other person, the total number
    of handshakes is 45. -/
theorem conference_handshakes :
  handshakes 10 = 45 := by
  sorry

end conference_handshakes_l1624_162476


namespace pentadecagon_triangles_l1624_162497

/-- The number of sides in a regular pentadecagon -/
def n : ℕ := 15

/-- The number of vertices required to form a triangle -/
def k : ℕ := 3

/-- The number of triangles that can be formed using the vertices of a regular pentadecagon -/
def num_triangles : ℕ := Nat.choose n k

theorem pentadecagon_triangles : num_triangles = 455 := by
  sorry

end pentadecagon_triangles_l1624_162497


namespace abc_value_for_factored_polynomial_l1624_162458

/-- If a polynomial ax^2 + bx + c can be factored as (x-1)(x-2), then abc = -6 -/
theorem abc_value_for_factored_polynomial (a b c : ℝ) :
  (∀ x, a * x^2 + b * x + c = (x - 1) * (x - 2)) →
  a * b * c = -6 := by
  sorry

end abc_value_for_factored_polynomial_l1624_162458


namespace rectangle_area_l1624_162438

/-- Given a rectangle ABCD divided into six identical squares with a perimeter of 160 cm,
    its area is 1536 square centimeters. -/
theorem rectangle_area (a : ℝ) (h1 : a > 0) : 
  (2 * (3 * a + 2 * a) = 160) → (3 * a) * (2 * a) = 1536 := by
  sorry

end rectangle_area_l1624_162438


namespace symmetric_line_correct_l1624_162464

/-- Given two lines in a plane, this function returns the equation of the line 
    that is symmetric to the first line with respect to the second line. -/
def symmetricLine (l₁ l₂ : ℝ → ℝ → Prop) : ℝ → ℝ → Prop :=
  sorry

/-- The first given line l₁ -/
def l₁ : ℝ → ℝ → Prop :=
  fun x y ↦ 3 * x - y - 3 = 0

/-- The second given line l₂ -/
def l₂ : ℝ → ℝ → Prop :=
  fun x y ↦ x + y - 1 = 0

/-- The expected symmetric line l₃ -/
def l₃ : ℝ → ℝ → Prop :=
  fun x y ↦ x - 3 * y - 1 = 0

theorem symmetric_line_correct :
  symmetricLine l₁ l₂ = l₃ := by sorry

end symmetric_line_correct_l1624_162464


namespace derivative_zero_at_one_l1624_162489

theorem derivative_zero_at_one (a : ℝ) : 
  let f : ℝ → ℝ := λ x => (x^2 + a) / (x + 1)
  (deriv f 1 = 0) → a = 3 := by
  sorry

end derivative_zero_at_one_l1624_162489


namespace triangle_intersection_theorem_l1624_162447

/-- A triangle in 2D space -/
structure Triangle :=
  (A B C : ℝ × ℝ)

/-- Checks if a triangle is acute-angled -/
def isAcute (t : Triangle) : Prop := sorry

/-- Constructs the next triangle from the given triangle -/
def nextTriangle (t : Triangle) : Triangle := sorry

/-- Counts the number of intersection points between two triangles -/
def intersectionPoints (t1 t2 : Triangle) : ℕ := sorry

/-- The main theorem -/
theorem triangle_intersection_theorem (A₀B₀C₀ : Triangle) (h : isAcute A₀B₀C₀) :
  ∀ n : ℕ, intersectionPoints ((nextTriangle^[n]) A₀B₀C₀) ((nextTriangle^[n+1]) A₀B₀C₀) = 6 := by
  sorry

end triangle_intersection_theorem_l1624_162447


namespace abs_neg_sqrt_two_l1624_162474

theorem abs_neg_sqrt_two : |(-Real.sqrt 2)| = Real.sqrt 2 := by sorry

end abs_neg_sqrt_two_l1624_162474


namespace yuna_has_biggest_number_l1624_162417

def yoongi_number : ℕ := 4
def yuna_number : ℕ := 5
def jungkook_number : ℕ := 6 - 3

theorem yuna_has_biggest_number :
  yuna_number > yoongi_number ∧ yuna_number > jungkook_number :=
by sorry

end yuna_has_biggest_number_l1624_162417


namespace sin_ratio_minus_sqrt3_over_sin_l1624_162449

theorem sin_ratio_minus_sqrt3_over_sin : 
  (Real.sin (80 * π / 180)) / (Real.sin (20 * π / 180)) - 
  (Real.sqrt 3) / (2 * Real.sin (80 * π / 180)) = 2 := by
  sorry

end sin_ratio_minus_sqrt3_over_sin_l1624_162449
