import Mathlib

namespace NUMINAMATH_CALUDE_inverse_square_function_l2125_212546

/-- A function that varies inversely as the square of its input -/
noncomputable def f (y : ℝ) : ℝ := 
  9 / (y * y)

/-- Theorem stating that if f(y) = 1 for some y and f(2) = 2.25, then f(3) = 1 -/
theorem inverse_square_function (h1 : ∃ y, f y = 1) (h2 : f 2 = 2.25) : f 3 = 1 := by
  sorry

end NUMINAMATH_CALUDE_inverse_square_function_l2125_212546


namespace NUMINAMATH_CALUDE_gcd_lcm_product_l2125_212532

theorem gcd_lcm_product (a b : ℕ+) (h : Nat.gcd a b * Nat.lcm a b = 252) :
  (∃ s : Finset ℕ+, s.card = 4 ∧ ∀ x : ℕ+, x ∈ s ↔ ∃ a b : ℕ+, Nat.gcd a b = x ∧ Nat.gcd a b * Nat.lcm a b = 252) :=
sorry

end NUMINAMATH_CALUDE_gcd_lcm_product_l2125_212532


namespace NUMINAMATH_CALUDE_cricket_team_throwers_l2125_212526

/-- Represents a cricket team with throwers and non-throwers -/
structure CricketTeam where
  total_players : ℕ
  throwers : ℕ
  right_handed : ℕ
  left_handed : ℕ

/-- Conditions for the cricket team problem -/
def valid_cricket_team (team : CricketTeam) : Prop :=
  team.total_players = 67 ∧
  team.throwers + team.right_handed + team.left_handed = team.total_players ∧
  team.right_handed + team.throwers = 57 ∧
  3 * team.left_handed = 2 * team.right_handed

theorem cricket_team_throwers (team : CricketTeam) 
  (h : valid_cricket_team team) : team.throwers = 37 := by
  sorry

end NUMINAMATH_CALUDE_cricket_team_throwers_l2125_212526


namespace NUMINAMATH_CALUDE_reflect_d_twice_l2125_212504

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Reflects a point over the y-axis -/
def reflectOverYAxis (p : Point) : Point :=
  { x := -p.x, y := p.y }

/-- Reflects a point over the line y = -x -/
def reflectOverYEqualNegX (p : Point) : Point :=
  { x := -p.y, y := -p.x }

/-- The main theorem stating that reflecting point D(5,1) over y-axis and then over y=-x results in D''(-1,5) -/
theorem reflect_d_twice :
  let d : Point := { x := 5, y := 1 }
  let d' := reflectOverYAxis d
  let d'' := reflectOverYEqualNegX d'
  d''.x = -1 ∧ d''.y = 5 := by sorry

end NUMINAMATH_CALUDE_reflect_d_twice_l2125_212504


namespace NUMINAMATH_CALUDE_sum_bottle_caps_l2125_212564

/-- The number of bottle caps for each child -/
def bottle_caps : Fin 9 → ℕ
  | ⟨0, _⟩ => 5
  | ⟨1, _⟩ => 8
  | ⟨2, _⟩ => 12
  | ⟨3, _⟩ => 7
  | ⟨4, _⟩ => 9
  | ⟨5, _⟩ => 10
  | ⟨6, _⟩ => 15
  | ⟨7, _⟩ => 4
  | ⟨8, _⟩ => 11
  | ⟨n+9, h⟩ => absurd h (Nat.not_lt_of_ge (Nat.le_add_left 9 n))

/-- The theorem stating that the sum of bottle caps is 81 -/
theorem sum_bottle_caps : (Finset.univ.sum bottle_caps) = 81 := by
  sorry

end NUMINAMATH_CALUDE_sum_bottle_caps_l2125_212564


namespace NUMINAMATH_CALUDE_solve_system_l2125_212540

theorem solve_system (p q : ℚ) 
  (eq1 : 5 * p + 6 * q = 10) 
  (eq2 : 6 * p + 5 * q = 17) : 
  p = 52 / 11 := by
sorry

end NUMINAMATH_CALUDE_solve_system_l2125_212540


namespace NUMINAMATH_CALUDE_product_three_reciprocal_squares_sum_l2125_212527

theorem product_three_reciprocal_squares_sum :
  ∀ a b : ℕ+, 
  (a * b : ℕ+) = 3 →
  (1 : ℚ) / (a : ℚ)^2 + (1 : ℚ) / (b : ℚ)^2 = 10 / 9 := by
sorry

end NUMINAMATH_CALUDE_product_three_reciprocal_squares_sum_l2125_212527


namespace NUMINAMATH_CALUDE_four_correct_propositions_l2125_212550

theorem four_correct_propositions 
  (a b c : ℝ) : 
  (((a < b → a + c < b + c) ∧                   -- Original proposition
    ((a + c < b + c) → (a < b)) ∧               -- Converse
    ((a ≥ b) → (a + c ≥ b + c)) ∧               -- Inverse
    ((a + c ≥ b + c) → (a ≥ b))) →              -- Contrapositive
   (4 = (Bool.toNat (a < b → a + c < b + c) +
         Bool.toNat ((a + c < b + c) → (a < b)) +
         Bool.toNat ((a ≥ b) → (a + c ≥ b + c)) +
         Bool.toNat ((a + c ≥ b + c) → (a ≥ b))))) :=
by sorry

end NUMINAMATH_CALUDE_four_correct_propositions_l2125_212550


namespace NUMINAMATH_CALUDE_intersection_complement_equality_l2125_212578

-- Define set A
def A : Set ℝ := {x | 1 < x ∧ x < 4}

-- Define set B
def B : Set ℝ := {x | x^2 - 2*x - 3 ≤ 0}

-- Theorem statement
theorem intersection_complement_equality : A ∩ (Set.univ \ B) = Set.Ioo 3 4 := by
  sorry

end NUMINAMATH_CALUDE_intersection_complement_equality_l2125_212578


namespace NUMINAMATH_CALUDE_not_in_range_quadratic_l2125_212563

theorem not_in_range_quadratic (b : ℝ) : 
  (∀ x : ℝ, x^2 + b*x + 3 ≠ -3) ↔ -Real.sqrt 24 < b ∧ b < Real.sqrt 24 := by
  sorry

end NUMINAMATH_CALUDE_not_in_range_quadratic_l2125_212563


namespace NUMINAMATH_CALUDE_triangle_value_l2125_212586

theorem triangle_value (triangle r : ℝ) 
  (h1 : triangle + r = 72)
  (h2 : (triangle + r) + r = 117) : 
  triangle = 27 := by
sorry

end NUMINAMATH_CALUDE_triangle_value_l2125_212586


namespace NUMINAMATH_CALUDE_quadratic_roots_range_l2125_212552

theorem quadratic_roots_range (m : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁^2 + m*x₁ + (m + 3) = 0 ∧ x₂^2 + m*x₂ + (m + 3) = 0) ↔
  m < -2 ∨ m > 6 :=
sorry

end NUMINAMATH_CALUDE_quadratic_roots_range_l2125_212552


namespace NUMINAMATH_CALUDE_min_value_of_absolute_sum_l2125_212593

theorem min_value_of_absolute_sum (x : ℚ) : 
  ∀ x : ℚ, |3 - x| + |x - 2| + |-1 + x| ≥ 2 ∧ 
  ∃ x : ℚ, |3 - x| + |x - 2| + |-1 + x| = 2 :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_absolute_sum_l2125_212593


namespace NUMINAMATH_CALUDE_infinitely_many_pairs_with_roots_product_one_l2125_212530

theorem infinitely_many_pairs_with_roots_product_one :
  ∀ n : ℕ, n > 2 →
  ∃ a b : ℤ,
    (∃ x y : ℝ, x ≠ y ∧ x * y = 1 ∧
      x^2019 = a * x + b ∧ y^2019 = a * y + b) ∧
    (∀ m : ℕ, m > 2 → m ≠ n →
      ∃ c d : ℤ, c ≠ a ∨ d ≠ b ∧
        (∃ u v : ℝ, u ≠ v ∧ u * v = 1 ∧
          u^2019 = c * u + d ∧ v^2019 = c * v + d)) :=
by sorry

end NUMINAMATH_CALUDE_infinitely_many_pairs_with_roots_product_one_l2125_212530


namespace NUMINAMATH_CALUDE_derivative_zero_necessary_not_sufficient_l2125_212549

-- Define a real-valued function
variable (f : ℝ → ℝ)

-- Define the property of being an extreme value point
def is_extreme_point (f : ℝ → ℝ) (x₀ : ℝ) : Prop :=
  ∀ x, f x ≤ f x₀ ∨ f x ≥ f x₀

-- Define the theorem
theorem derivative_zero_necessary_not_sufficient :
  (∀ x₀ : ℝ, is_extreme_point f x₀ → (deriv f) x₀ = 0) ∧
  (∃ x₀ : ℝ, (deriv f) x₀ = 0 ∧ ¬(is_extreme_point f x₀)) :=
sorry

end NUMINAMATH_CALUDE_derivative_zero_necessary_not_sufficient_l2125_212549


namespace NUMINAMATH_CALUDE_strawberry_supply_theorem_l2125_212568

/-- Represents the weekly strawberry requirements for each bakery -/
structure BakeryRequirements where
  first : ℕ
  second : ℕ
  third : ℕ

/-- Calculates the total number of sacks needed for all bakeries over a given period -/
def totalSacks (req : BakeryRequirements) (weeks : ℕ) : ℕ :=
  (req.first + req.second + req.third) * weeks

/-- The problem statement -/
theorem strawberry_supply_theorem (req : BakeryRequirements) (weeks : ℕ) 
    (h1 : req.first = 2)
    (h2 : req.second = 4)
    (h3 : req.third = 12)
    (h4 : weeks = 4) :
  totalSacks req weeks = 72 := by
  sorry

#check strawberry_supply_theorem

end NUMINAMATH_CALUDE_strawberry_supply_theorem_l2125_212568


namespace NUMINAMATH_CALUDE_cups_in_smaller_purchase_is_40_l2125_212575

/-- The cost of a single paper plate -/
def plate_cost : ℝ := sorry

/-- The cost of a single paper cup -/
def cup_cost : ℝ := sorry

/-- The number of cups in the smaller purchase -/
def cups_in_smaller_purchase : ℕ := sorry

/-- The total cost of 100 plates and 200 cups is $6.00 -/
axiom total_cost_large : 100 * plate_cost + 200 * cup_cost = 6

/-- The total cost of 20 plates and the unknown number of cups is $1.20 -/
axiom total_cost_small : 20 * plate_cost + cups_in_smaller_purchase * cup_cost = 1.2

theorem cups_in_smaller_purchase_is_40 : cups_in_smaller_purchase = 40 := by sorry

end NUMINAMATH_CALUDE_cups_in_smaller_purchase_is_40_l2125_212575


namespace NUMINAMATH_CALUDE_stone_slab_length_l2125_212536

/-- Given a total floor area covered by equal-sized square stone slabs,
    calculate the length of each slab in centimeters. -/
theorem stone_slab_length
  (total_area : ℝ)
  (num_slabs : ℕ)
  (h_area : total_area = 67.5)
  (h_num : num_slabs = 30)
  : ∃ (slab_length : ℝ),
    slab_length = 150 ∧
    slab_length^2 * num_slabs = total_area * 10000 := by
  sorry

#check stone_slab_length

end NUMINAMATH_CALUDE_stone_slab_length_l2125_212536


namespace NUMINAMATH_CALUDE_golden_ratio_properties_l2125_212539

theorem golden_ratio_properties (x y : ℝ) 
  (hx : x^2 = x + 1) 
  (hy : y^2 = y + 1) 
  (hxy : x ≠ y) : 
  (x + y = 1) ∧ (x^5 + y^5 = 11) := by
  sorry

end NUMINAMATH_CALUDE_golden_ratio_properties_l2125_212539


namespace NUMINAMATH_CALUDE_ice_harvest_theorem_l2125_212551

/-- Represents a team that harvests ice blocks -/
inductive Team
| A
| B
| C

/-- Represents the proportion of total ice harvested by each team -/
def teamProportion (t : Team) : ℝ :=
  match t with
  | Team.A => 0.3
  | Team.B => 0.3
  | Team.C => 0.4

/-- Represents the utilization rate of ice harvested by each team -/
def utilizationRate (t : Team) : ℝ :=
  match t with
  | Team.A => 0.8
  | Team.B => 0.75
  | Team.C => 0.6

/-- The number of random draws -/
def numDraws : ℕ := 3

/-- Theorem stating the expectation of Team C's blocks being selected and the probability of a usable block being from Team B -/
theorem ice_harvest_theorem :
  let p := teamProportion Team.C
  let expectation := p * numDraws
  let probUsableB := (teamProportion Team.B * utilizationRate Team.B) /
    (teamProportion Team.A * utilizationRate Team.A +
     teamProportion Team.B * utilizationRate Team.B +
     teamProportion Team.C * utilizationRate Team.C)
  expectation = 6/5 ∧ probUsableB = 15/47 := by
  sorry


end NUMINAMATH_CALUDE_ice_harvest_theorem_l2125_212551


namespace NUMINAMATH_CALUDE_symmetric_circle_equation_l2125_212537

/-- Given a circle and a line of symmetry, this theorem proves the equation of the symmetric circle. -/
theorem symmetric_circle_equation (x y : ℝ) :
  (x^2 + y^2 + 2*x - 2*y + 1 = 0) →  -- Given circle equation
  (x - y = 0) →                      -- Line of symmetry
  (x^2 + y^2 - 2*x + 2*y + 1 = 0)    -- Symmetric circle equation
:= by sorry

end NUMINAMATH_CALUDE_symmetric_circle_equation_l2125_212537


namespace NUMINAMATH_CALUDE_midpoint_coordinate_product_l2125_212519

/-- The product of the coordinates of the midpoint of a line segment
    with endpoints (5, -2) and (-3, 6) is equal to 2. -/
theorem midpoint_coordinate_product : 
  let x₁ : ℝ := 5
  let y₁ : ℝ := -2
  let x₂ : ℝ := -3
  let y₂ : ℝ := 6
  let midpoint_x : ℝ := (x₁ + x₂) / 2
  let midpoint_y : ℝ := (y₁ + y₂) / 2
  midpoint_x * midpoint_y = 2 := by
  sorry

end NUMINAMATH_CALUDE_midpoint_coordinate_product_l2125_212519


namespace NUMINAMATH_CALUDE_multiple_of_one_third_equals_two_ninths_l2125_212583

theorem multiple_of_one_third_equals_two_ninths :
  ∃ x : ℚ, x * (1/3 : ℚ) = 2/9 ∧ x = 2/3 := by sorry

end NUMINAMATH_CALUDE_multiple_of_one_third_equals_two_ninths_l2125_212583


namespace NUMINAMATH_CALUDE_perpendicular_lines_m_values_l2125_212520

theorem perpendicular_lines_m_values (m : ℝ) : 
  (∃ (x y : ℝ), mx + 2*y + 1 = 0 ∧ x - m^2*y + 1/2 = 0) →
  (m * 1 + 2 * (-m^2) = 0) →
  (m = 0 ∨ m = 1/2) :=
sorry

end NUMINAMATH_CALUDE_perpendicular_lines_m_values_l2125_212520


namespace NUMINAMATH_CALUDE_problem_solution_l2125_212507

theorem problem_solution (x y : ℝ) 
  (h1 : x = 151)
  (h2 : x^3 * y - 4 * x^2 * y + 4 * x * y = 342200) : 
  y = 342200 / 3354151 := by sorry

end NUMINAMATH_CALUDE_problem_solution_l2125_212507


namespace NUMINAMATH_CALUDE_lights_on_200_7_11_l2125_212577

/-- The number of lights that are on after the switching operation -/
def lights_on (total_lights : ℕ) (interval1 interval2 : ℕ) : ℕ :=
  (total_lights / interval1 + total_lights / interval2) -
  2 * (total_lights / (interval1 * interval2))

/-- Theorem stating the number of lights on after the switching operation -/
theorem lights_on_200_7_11 :
  lights_on 200 7 11 = 44 := by
sorry

end NUMINAMATH_CALUDE_lights_on_200_7_11_l2125_212577


namespace NUMINAMATH_CALUDE_divisible_by_twelve_l2125_212511

theorem divisible_by_twelve (n : ℤ) (h : n > 1) : ∃ k : ℤ, n^4 - n^2 = 12 * k := by
  sorry

end NUMINAMATH_CALUDE_divisible_by_twelve_l2125_212511


namespace NUMINAMATH_CALUDE_complement_of_at_least_two_defective_l2125_212594

/-- Represents the number of products in the sample -/
def sample_size : ℕ := 10

/-- Represents the event of having at least two defective products -/
def event_A (defective : ℕ) : Prop := defective ≥ 2

/-- Represents the complementary event of A -/
def complement_A (defective : ℕ) : Prop := defective ≤ 1

/-- Theorem stating that the complement of event A is "at most one defective product" -/
theorem complement_of_at_least_two_defective :
  ∀ (defective : ℕ), defective ≤ sample_size →
    (¬ event_A defective) ↔ complement_A defective := by
  sorry

end NUMINAMATH_CALUDE_complement_of_at_least_two_defective_l2125_212594


namespace NUMINAMATH_CALUDE_max_value_of_a_l2125_212561

theorem max_value_of_a (a b c : ℝ) 
  (sum_condition : a + b + c = 6)
  (product_sum_condition : a * b + a * c + b * c = 11) :
  a ≤ 2 + (2 * Real.sqrt 15) / 3 :=
sorry

end NUMINAMATH_CALUDE_max_value_of_a_l2125_212561


namespace NUMINAMATH_CALUDE_unique_positive_solution_num_positive_solutions_correct_l2125_212500

/-- The polynomial function f(x) = x^11 + 8x^10 + 15x^9 + 1000x^8 - 1200x^7 -/
def f (x : ℝ) : ℝ := x^11 + 8*x^10 + 15*x^9 + 1000*x^8 - 1200*x^7

/-- The number of positive real solutions to the equation f(x) = 0 -/
def num_positive_solutions : ℕ := 1

theorem unique_positive_solution : 
  ∃! (x : ℝ), x > 0 ∧ f x = 0 :=
sorry

theorem num_positive_solutions_correct : 
  (∃! (x : ℝ), x > 0 ∧ f x = 0) ↔ num_positive_solutions = 1 :=
sorry

end NUMINAMATH_CALUDE_unique_positive_solution_num_positive_solutions_correct_l2125_212500


namespace NUMINAMATH_CALUDE_function_equivalence_l2125_212587

open Real

theorem function_equivalence (x : ℝ) :
  2 * (cos x)^2 - Real.sqrt 3 * sin (2 * x) = 2 * sin (2 * (x + 5 * π / 12)) + 1 := by
  sorry

end NUMINAMATH_CALUDE_function_equivalence_l2125_212587


namespace NUMINAMATH_CALUDE_sum_three_digit_integers_mod_1000_l2125_212525

def sum_three_digit_integers : ℕ :=
  (45 * 100 * 100) + (45 * 100 * 10) + (45 * 100)

theorem sum_three_digit_integers_mod_1000 :
  sum_three_digit_integers % 1000 = 500 := by sorry

end NUMINAMATH_CALUDE_sum_three_digit_integers_mod_1000_l2125_212525


namespace NUMINAMATH_CALUDE_largest_number_is_312_base_4_l2125_212510

def base_to_decimal (digits : List Nat) (base : Nat) : Nat :=
  digits.reverse.enum.foldl (fun acc (i, d) => acc + d * base ^ i) 0

theorem largest_number_is_312_base_4 :
  let binary := [1, 1, 1, 1, 1]
  let ternary := [1, 2, 2, 1]
  let quaternary := [3, 1, 2]
  let octal := [5, 6]
  
  (base_to_decimal quaternary 4) = 54 ∧
  (base_to_decimal quaternary 4) > (base_to_decimal binary 2) ∧
  (base_to_decimal quaternary 4) > (base_to_decimal ternary 3) ∧
  (base_to_decimal quaternary 4) > (base_to_decimal octal 8) :=
by
  sorry

end NUMINAMATH_CALUDE_largest_number_is_312_base_4_l2125_212510


namespace NUMINAMATH_CALUDE_angle_sum_in_triangle_rectangle_l2125_212512

/-- A rectangle containing an equilateral triangle -/
structure TriangleInRectangle where
  /-- The measure of one angle between the rectangle and triangle sides -/
  x : ℝ
  /-- The measure of the other angle between the rectangle and triangle sides -/
  y : ℝ
  /-- The rectangle has right angles -/
  rectangle_right_angles : x + y + 60 + 90 + 90 = 540
  /-- The inner triangle is equilateral -/
  equilateral_triangle : True

/-- The sum of angles x and y in a rectangle containing an equilateral triangle is 60° -/
theorem angle_sum_in_triangle_rectangle (t : TriangleInRectangle) : t.x + t.y = 60 := by
  sorry

end NUMINAMATH_CALUDE_angle_sum_in_triangle_rectangle_l2125_212512


namespace NUMINAMATH_CALUDE_optimal_racket_purchase_l2125_212513

/-- Represents the purchase and selling prices of rackets -/
structure RacketPrices where
  tableTennisBuy : ℝ
  tableTennisSell : ℝ
  badmintonBuy : ℝ
  badmintonSell : ℝ

/-- Represents the quantity of rackets to purchase -/
structure RacketQuantities where
  tableTennis : ℝ
  badminton : ℝ

/-- Calculates the profit given prices and quantities -/
def calculateProfit (prices : RacketPrices) (quantities : RacketQuantities) : ℝ :=
  (prices.tableTennisSell - prices.tableTennisBuy) * quantities.tableTennis +
  (prices.badmintonSell - prices.badmintonBuy) * quantities.badminton

/-- The main theorem stating the optimal solution -/
theorem optimal_racket_purchase
  (prices : RacketPrices)
  (h1 : 2 * prices.tableTennisBuy + prices.badmintonBuy = 120)
  (h2 : 4 * prices.tableTennisBuy + 3 * prices.badmintonBuy = 270)
  (h3 : prices.tableTennisSell = 55)
  (h4 : prices.badmintonSell = 50)
  : ∃ (quantities : RacketQuantities),
    quantities.tableTennis + quantities.badminton = 300 ∧
    quantities.tableTennis ≥ (1/3) * quantities.badminton ∧
    prices.tableTennisBuy = 45 ∧
    prices.badmintonBuy = 30 ∧
    quantities.tableTennis = 75 ∧
    quantities.badminton = 225 ∧
    calculateProfit prices quantities = 5250 ∧
    ∀ (other : RacketQuantities),
      other.tableTennis + other.badminton = 300 →
      other.tableTennis ≥ (1/3) * other.badminton →
      calculateProfit prices quantities ≥ calculateProfit prices other := by
  sorry

end NUMINAMATH_CALUDE_optimal_racket_purchase_l2125_212513


namespace NUMINAMATH_CALUDE_aisha_shopping_money_l2125_212522

theorem aisha_shopping_money (initial_money : ℝ) : 
  let after_first := initial_money - (0.4 * initial_money + 4)
  let after_second := after_first - (0.5 * after_first + 5)
  let after_third := after_second - (0.6 * after_second + 6)
  after_third = 2 → initial_money = 90 := by
sorry

end NUMINAMATH_CALUDE_aisha_shopping_money_l2125_212522


namespace NUMINAMATH_CALUDE_min_value_of_sum_of_roots_l2125_212573

theorem min_value_of_sum_of_roots (x : ℝ) :
  ∃ (y : ℝ), ∀ (x : ℝ),
    Real.sqrt (5 * x^2 - 16 * x + 16) + Real.sqrt (5 * x^2 - 18 * x + 29) ≥ y ∧
    ∃ (z : ℝ), Real.sqrt (5 * z^2 - 16 * z + 16) + Real.sqrt (5 * z^2 - 18 * z + 29) = y :=
by
  use Real.sqrt 29
  sorry

end NUMINAMATH_CALUDE_min_value_of_sum_of_roots_l2125_212573


namespace NUMINAMATH_CALUDE_three_power_fraction_equals_41_40_l2125_212518

theorem three_power_fraction_equals_41_40 :
  (3^1008 + 3^1004) / (3^1008 - 3^1004) = 41/40 := by
  sorry

end NUMINAMATH_CALUDE_three_power_fraction_equals_41_40_l2125_212518


namespace NUMINAMATH_CALUDE_midpoint_expression_evaluation_l2125_212558

/-- Given two points P and Q in the plane, prove that the expression 3x - 5y 
    evaluates to -36 at their midpoint R(x, y). -/
theorem midpoint_expression_evaluation (P Q : ℝ × ℝ) (h1 : P = (12, 15)) (h2 : Q = (4, 9)) :
  let R : ℝ × ℝ := ((P.1 + Q.1) / 2, (P.2 + Q.2) / 2)
  3 * R.1 - 5 * R.2 = -36 := by
sorry

end NUMINAMATH_CALUDE_midpoint_expression_evaluation_l2125_212558


namespace NUMINAMATH_CALUDE_chinese_remainder_theorem_l2125_212521

theorem chinese_remainder_theorem (x : ℤ) : 
  (x ≡ 2 [ZMOD 6] ∧ x ≡ 3 [ZMOD 5] ∧ x ≡ 4 [ZMOD 7]) ↔ 
  (∃ k : ℤ, x = 210 * k - 52) := by
  sorry

end NUMINAMATH_CALUDE_chinese_remainder_theorem_l2125_212521


namespace NUMINAMATH_CALUDE_total_tiles_count_l2125_212505

def room_length : ℕ := 18
def room_width : ℕ := 15
def border_tile_size : ℕ := 2
def border_width : ℕ := 2
def inner_tile_size : ℕ := 3

def border_tiles : ℕ := 
  2 * (room_length / border_tile_size + room_width / border_tile_size) + 4

def inner_area : ℕ := (room_length - 2 * border_width) * (room_width - 2 * border_width)
def inner_tiles : ℕ := inner_area / (inner_tile_size * inner_tile_size)

theorem total_tiles_count :
  border_tiles + inner_tiles = 45 := by sorry

end NUMINAMATH_CALUDE_total_tiles_count_l2125_212505


namespace NUMINAMATH_CALUDE_problem_solution_l2125_212531

def f (a x : ℝ) : ℝ := 3 * |x - a| + |3 * x + 1|

def g (x : ℝ) : ℝ := |4 * x - 1| - |x + 2|

theorem problem_solution :
  (∀ x : ℝ, g x < 6 ↔ -7/5 < x ∧ x < 3) ∧
  (∃ x₁ x₂ : ℝ, f a x₁ = -g x₂) → -13/12 ≤ a ∧ a ≤ 5/12 :=
sorry

end NUMINAMATH_CALUDE_problem_solution_l2125_212531


namespace NUMINAMATH_CALUDE_initial_people_at_table_l2125_212529

theorem initial_people_at_table (initial : ℕ) 
  (h1 : initial ≥ 6)
  (h2 : initial - 6 + 5 = 10) : initial = 11 := by
  sorry

end NUMINAMATH_CALUDE_initial_people_at_table_l2125_212529


namespace NUMINAMATH_CALUDE_g_zero_at_three_l2125_212509

def g (x s : ℝ) : ℝ := 3 * x^4 + 2 * x^3 - x^2 - 4 * x + s

theorem g_zero_at_three (s : ℝ) : g 3 s = 0 ↔ s = -276 := by sorry

end NUMINAMATH_CALUDE_g_zero_at_three_l2125_212509


namespace NUMINAMATH_CALUDE_square_roots_problem_l2125_212585

theorem square_roots_problem (a : ℝ) : 
  (a + 3) ^ 2 = (2 * a - 6) ^ 2 → (a + 3) ^ 2 = 16 := by
  sorry

end NUMINAMATH_CALUDE_square_roots_problem_l2125_212585


namespace NUMINAMATH_CALUDE_largest_number_of_cubic_roots_l2125_212559

theorem largest_number_of_cubic_roots (p q r : ℝ) 
  (sum_eq : p + q + r = 3)
  (sum_prod_eq : p * q + p * r + q * r = -6)
  (prod_eq : p * q * r = -8) :
  max p (max q r) = (1 + Real.sqrt 17) / 2 := by
  sorry

end NUMINAMATH_CALUDE_largest_number_of_cubic_roots_l2125_212559


namespace NUMINAMATH_CALUDE_cube_surface_area_increase_l2125_212597

theorem cube_surface_area_increase (L : ℝ) (h : L > 0) :
  let original_surface_area := 6 * L^2
  let new_edge_length := 1.4 * L
  let new_surface_area := 6 * new_edge_length^2
  (new_surface_area - original_surface_area) / original_surface_area = 0.96 := by
sorry

end NUMINAMATH_CALUDE_cube_surface_area_increase_l2125_212597


namespace NUMINAMATH_CALUDE_unique_number_exists_l2125_212571

def is_valid_number (n : ℕ) : Prop :=
  ∃ (a b c d e : ℕ),
    n = a * 10000 + b * 1000 + c * 100 + d * 10 + e ∧
    a ∈ ({1, 2, 3, 4, 5} : Set ℕ) ∧
    b ∈ ({1, 2, 3, 4, 5} : Set ℕ) ∧
    c ∈ ({1, 2, 3, 4, 5} : Set ℕ) ∧
    d ∈ ({1, 2, 3, 4, 5} : Set ℕ) ∧
    e ∈ ({1, 2, 3, 4, 5} : Set ℕ) ∧
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧
    b ≠ c ∧ b ≠ d ∧ b ≠ e ∧
    c ≠ d ∧ c ≠ e ∧
    d ≠ e ∧
    (a * 100 + b * 10 + c) % 4 = 0 ∧
    (b * 100 + c * 10 + d) % 5 = 0 ∧
    (c * 100 + d * 10 + e) % 3 = 0

theorem unique_number_exists : ∃! n : ℕ, is_valid_number n :=
sorry

end NUMINAMATH_CALUDE_unique_number_exists_l2125_212571


namespace NUMINAMATH_CALUDE_smallest_non_special_number_l2125_212580

def is_triangular (n : ℕ) : Prop :=
  ∃ k : ℕ, n = k * (k + 1) / 2

def is_prime_power (n : ℕ) : Prop :=
  ∃ p k : ℕ, Nat.Prime p ∧ n = p ^ k

def is_prime_plus_one (n : ℕ) : Prop :=
  ∃ p : ℕ, Nat.Prime p ∧ n = p + 1

def is_product_of_distinct_primes (n : ℕ) : Prop :=
  ∃ p q : ℕ, Nat.Prime p ∧ Nat.Prime q ∧ p ≠ q ∧ n = p * q

theorem smallest_non_special_number : 
  (∀ n < 40, is_triangular n ∨ is_prime_power n ∨ is_prime_plus_one n ∨ is_product_of_distinct_primes n) ∧
  ¬(is_triangular 40 ∨ is_prime_power 40 ∨ is_prime_plus_one 40 ∨ is_product_of_distinct_primes 40) :=
sorry

end NUMINAMATH_CALUDE_smallest_non_special_number_l2125_212580


namespace NUMINAMATH_CALUDE_problem_solving_probability_l2125_212543

theorem problem_solving_probability (xavier_prob yvonne_prob zelda_prob : ℚ)
  (hx : xavier_prob = 1 / 4)
  (hy : yvonne_prob = 1 / 3)
  (hz : zelda_prob = 5 / 8) :
  xavier_prob * yvonne_prob * (1 - zelda_prob) = 1 / 32 := by
  sorry

end NUMINAMATH_CALUDE_problem_solving_probability_l2125_212543


namespace NUMINAMATH_CALUDE_no_solution_quadratic_inequality_l2125_212545

theorem no_solution_quadratic_inequality (x : ℝ) : 
  (5 * x^2 + 6 * x + 8 < 0) ∧ (abs x > 2) → False :=
by
  sorry


end NUMINAMATH_CALUDE_no_solution_quadratic_inequality_l2125_212545


namespace NUMINAMATH_CALUDE_probability_same_length_is_33_105_l2125_212544

/-- The number of sides in a regular hexagon -/
def num_sides : ℕ := 6

/-- The number of diagonals in a regular hexagon -/
def num_diagonals : ℕ := 9

/-- The number of longer diagonals in a regular hexagon -/
def num_longer_diagonals : ℕ := 6

/-- The number of shorter diagonals in a regular hexagon -/
def num_shorter_diagonals : ℕ := 3

/-- The set of all sides and diagonals of a regular hexagon -/
def S : Finset ℕ := Finset.range (num_sides + num_diagonals)

/-- The probability of selecting two segments of the same length from S -/
def probability_same_length : ℚ :=
  (Nat.choose num_sides 2 + Nat.choose num_longer_diagonals 2 + Nat.choose num_shorter_diagonals 2) /
  Nat.choose S.card 2

theorem probability_same_length_is_33_105 :
  probability_same_length = 33 / 105 := by sorry

end NUMINAMATH_CALUDE_probability_same_length_is_33_105_l2125_212544


namespace NUMINAMATH_CALUDE_negation_equivalence_l2125_212524

-- Define the universe of discourse
variable (U : Type)

-- Define predicates
variable (student : U → Prop)
variable (shares_truth : U → Prop)

-- Define the original statement
def every_student_shares_truth : Prop :=
  ∀ x, student x → shares_truth x

-- Define the negation
def negation_statement : Prop :=
  ∃ x, student x ∧ ¬(shares_truth x)

-- Theorem to prove
theorem negation_equivalence :
  ¬(every_student_shares_truth U student shares_truth) ↔ negation_statement U student shares_truth :=
by sorry

end NUMINAMATH_CALUDE_negation_equivalence_l2125_212524


namespace NUMINAMATH_CALUDE_expression_equals_14_l2125_212598

theorem expression_equals_14 (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0)
  (h_sum : x + y + z = 0) (h_prod : x*y + x*z + y*z ≠ 0) :
  (x^7 + y^7 + z^7) / (x*y*z*(x^2 + y^2 + z^2)) = 14 := by
  sorry

end NUMINAMATH_CALUDE_expression_equals_14_l2125_212598


namespace NUMINAMATH_CALUDE_circle_square_area_ratio_l2125_212569

theorem circle_square_area_ratio (r : ℝ) (h : r > 0) :
  let inner_square_side := 3 * r
  let outer_circle_radius := inner_square_side * Real.sqrt 2 / 2
  let outer_square_side := 2 * outer_circle_radius
  (π * r^2) / (outer_square_side^2) = π / 18 := by
  sorry

end NUMINAMATH_CALUDE_circle_square_area_ratio_l2125_212569


namespace NUMINAMATH_CALUDE_gcd_87654321_12345678_l2125_212553

theorem gcd_87654321_12345678 : Nat.gcd 87654321 12345678 = 75 := by
  sorry

end NUMINAMATH_CALUDE_gcd_87654321_12345678_l2125_212553


namespace NUMINAMATH_CALUDE_sum_a1_a5_l2125_212562

/-- Given a sequence {aₙ} where the sum of the first n terms Sₙ = n² + a₁/2,
    prove that a₁ + a₅ = 11 -/
theorem sum_a1_a5 (a : ℕ → ℝ) (S : ℕ → ℝ) 
    (h : ∀ n : ℕ, n > 0 → S n = n^2 + a 1 / 2) : 
    a 1 + a 5 = 11 := by
  sorry

end NUMINAMATH_CALUDE_sum_a1_a5_l2125_212562


namespace NUMINAMATH_CALUDE_geometric_sequence_fourth_term_l2125_212576

/-- Given a sequence a_n where a_1 = 2 and {1 + a_n} forms a geometric sequence
    with common ratio 3, prove that a_4 = 80. -/
theorem geometric_sequence_fourth_term (a : ℕ → ℝ) :
  a 1 = 2 ∧
  (∀ n : ℕ, (1 + a (n + 1)) = 3 * (1 + a n)) →
  a 4 = 80 :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_fourth_term_l2125_212576


namespace NUMINAMATH_CALUDE_simplified_expression_equals_half_l2125_212534

theorem simplified_expression_equals_half :
  let x : ℚ := 1/3
  let y : ℚ := -1/2
  (2*x + 3*y)^2 - (2*x + y)*(2*x - y) = 1/2 :=
by sorry

end NUMINAMATH_CALUDE_simplified_expression_equals_half_l2125_212534


namespace NUMINAMATH_CALUDE_least_N_congruence_l2125_212533

/-- Sum of digits in base 3 representation -/
def f (n : ℕ) : ℕ := sorry

/-- Sum of digits in base 8 representation of f(n) -/
def g (n : ℕ) : ℕ := sorry

/-- The least value of n such that g(n) ≥ 10 -/
def N : ℕ := sorry

theorem least_N_congruence : N ≡ 862 [MOD 1000] := by sorry

end NUMINAMATH_CALUDE_least_N_congruence_l2125_212533


namespace NUMINAMATH_CALUDE_slope_of_AA_l2125_212502

-- Define a point in 2D space
structure Point2D where
  x : ℝ
  y : ℝ

-- Define a triangle in 2D space
structure Triangle where
  A : Point2D
  B : Point2D
  C : Point2D

-- Define the transformation (shift right by 2 and reflect across y=x)
def transform (p : Point2D) : Point2D :=
  { x := p.y, y := p.x + 2 }

-- Theorem statement
theorem slope_of_AA'_is_one (t : Triangle)
  (h1 : t.A.x ≥ 0 ∧ t.A.y ≥ 0)  -- A is in first quadrant
  (h2 : t.B.x ≥ 0 ∧ t.B.y ≥ 0)  -- B is in first quadrant
  (h3 : t.C.x ≥ 0 ∧ t.C.y ≥ 0)  -- C is in first quadrant
  (h4 : t.A.x + 2 ≥ 0 ∧ t.A.y ≥ 0)  -- A+2 is in first quadrant
  (h5 : t.B.x + 2 ≥ 0 ∧ t.B.y ≥ 0)  -- B+2 is in first quadrant
  (h6 : t.C.x + 2 ≥ 0 ∧ t.C.y ≥ 0)  -- C+2 is in first quadrant
  (h7 : t.A.x ≠ t.A.y)  -- A not on y=x
  (h8 : t.B.x ≠ t.B.y)  -- B not on y=x
  (h9 : t.C.x ≠ t.C.y)  -- C not on y=x
  : (transform t.A).y - t.A.y = (transform t.A).x - t.A.x :=
by sorry

end NUMINAMATH_CALUDE_slope_of_AA_l2125_212502


namespace NUMINAMATH_CALUDE_no_triangle_from_divisibility_conditions_l2125_212554

theorem no_triangle_from_divisibility_conditions (a b c : ℕ+) :
  a ≠ b ∧ b ≠ c ∧ a ≠ c →
  Nat.gcd a.val (Nat.gcd b.val c.val) = 1 →
  a.val ∣ (b.val - c.val)^2 →
  b.val ∣ (a.val - c.val)^2 →
  c.val ∣ (a.val - b.val)^2 →
  ¬(a.val + b.val > c.val ∧ b.val + c.val > a.val ∧ c.val + a.val > b.val) := by
sorry

end NUMINAMATH_CALUDE_no_triangle_from_divisibility_conditions_l2125_212554


namespace NUMINAMATH_CALUDE_triangle_angle_calculation_l2125_212501

/-- Configuration of triangles ABC and CDE --/
structure TriangleConfig where
  -- Angles in triangle ABC
  angle_A : ℝ
  angle_B : ℝ
  -- Angle y in triangle CDE
  angle_y : ℝ
  -- Assertions about the configuration
  angle_A_eq : angle_A = 50
  angle_B_eq : angle_B = 70
  right_angle_E : True  -- Represents the right angle at E
  angle_C_eq : True  -- Represents that angle at C is same in both triangles

/-- Theorem stating that in the given configuration, y = 30° --/
theorem triangle_angle_calculation (config : TriangleConfig) : config.angle_y = 30 := by
  sorry


end NUMINAMATH_CALUDE_triangle_angle_calculation_l2125_212501


namespace NUMINAMATH_CALUDE_polynomial_ratio_equals_infinite_sum_l2125_212556

theorem polynomial_ratio_equals_infinite_sum (x : ℝ) (h : x ∈ Set.Ioo 0 1) :
  x / (1 - x) = ∑' n, x^(2^n) / (1 - x^(2^n + 1)) :=
sorry

end NUMINAMATH_CALUDE_polynomial_ratio_equals_infinite_sum_l2125_212556


namespace NUMINAMATH_CALUDE_trapezoid_perimeter_l2125_212560

/-- Represents a trapezoid ABCD with given side lengths and a right angle at BCD -/
structure Trapezoid where
  AB : ℝ
  CD : ℝ
  BC : ℝ
  AD : ℝ
  angle_BCD_is_right : Bool

/-- Calculate the perimeter of the trapezoid -/
def perimeter (t : Trapezoid) : ℝ :=
  t.AB + t.CD + t.BC + t.AD

/-- Theorem: The perimeter of the given trapezoid is 118 units -/
theorem trapezoid_perimeter : 
  ∀ (t : Trapezoid), 
  t.AB = 33 ∧ t.CD = 15 ∧ t.BC = 45 ∧ t.AD = 25 ∧ t.angle_BCD_is_right = true → 
  perimeter t = 118 := by
  sorry

#check trapezoid_perimeter

end NUMINAMATH_CALUDE_trapezoid_perimeter_l2125_212560


namespace NUMINAMATH_CALUDE_unpainted_area_crossed_boards_l2125_212596

/-- The area of the unpainted region when two boards cross -/
theorem unpainted_area_crossed_boards (width1 width2 : ℝ) (angle : ℝ) :
  width1 = 5 →
  width2 = 8 →
  angle = π / 4 →
  (width1 * width2 * Real.sqrt 2) / 2 = 40 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_unpainted_area_crossed_boards_l2125_212596


namespace NUMINAMATH_CALUDE_smallest_positive_quadratic_form_l2125_212548

def quadratic_form (x y : ℤ) : ℤ := 20 * x^2 + 80 * x * y + 95 * y^2

theorem smallest_positive_quadratic_form :
  (∃ x y : ℤ, quadratic_form x y = 67) ∧
  (∀ n : ℕ, n > 0 → n < 67 → ∀ x y : ℤ, quadratic_form x y ≠ n) :=
sorry

end NUMINAMATH_CALUDE_smallest_positive_quadratic_form_l2125_212548


namespace NUMINAMATH_CALUDE_highway_vehicles_l2125_212570

theorem highway_vehicles (total : ℕ) (trucks : ℕ) (cars : ℕ) 
  (h1 : total = 300)
  (h2 : cars = 2 * trucks)
  (h3 : total = cars + trucks) :
  trucks = 100 := by
  sorry

end NUMINAMATH_CALUDE_highway_vehicles_l2125_212570


namespace NUMINAMATH_CALUDE_alpha_value_l2125_212516

theorem alpha_value (α : Real) (h1 : 0 < α ∧ α < π / 2) 
  (h2 : Real.sin (α - π / 18) = Real.sqrt 3 / 2) : α = π / 180 * 70 := by
  sorry

end NUMINAMATH_CALUDE_alpha_value_l2125_212516


namespace NUMINAMATH_CALUDE_unique_equal_sums_l2125_212528

def arithmetic_sum (a₁ : ℚ) (d : ℚ) (n : ℕ) : ℚ :=
  (n : ℚ) / 2 * (2 * a₁ + (n - 1 : ℚ) * d)

theorem unique_equal_sums : ∃! (n : ℕ), n > 0 ∧ 
  arithmetic_sum 3 7 n = arithmetic_sum 5 3 n := by sorry

end NUMINAMATH_CALUDE_unique_equal_sums_l2125_212528


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l2125_212535

def A : Set (ℝ × ℝ) := {p | p.2 = p.1 + 1}
def B : Set (ℝ × ℝ) := {p | p.2 = 4 - 2*p.1}

theorem intersection_of_A_and_B :
  A ∩ B = {(1, 2)} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l2125_212535


namespace NUMINAMATH_CALUDE_plumber_pipe_cost_l2125_212542

/-- The total cost of copper and plastic pipe given specific quantities and prices -/
theorem plumber_pipe_cost (copper_length : ℕ) (plastic_length : ℕ) 
  (copper_price : ℕ) (plastic_price : ℕ) : 
  copper_length = 10 → 
  plastic_length = 15 → 
  copper_price = 5 → 
  plastic_price = 3 → 
  copper_length * copper_price + plastic_length * plastic_price = 95 := by
  sorry

#check plumber_pipe_cost

end NUMINAMATH_CALUDE_plumber_pipe_cost_l2125_212542


namespace NUMINAMATH_CALUDE_price_increase_percentage_l2125_212541

theorem price_increase_percentage (old_price new_price : ℝ) (h1 : old_price = 300) (h2 : new_price = 330) :
  (new_price - old_price) / old_price * 100 = 10 := by
  sorry

end NUMINAMATH_CALUDE_price_increase_percentage_l2125_212541


namespace NUMINAMATH_CALUDE_log_comparison_l2125_212572

theorem log_comparison : Real.log 4 / Real.log 3 > Real.log 5 / Real.log 4 := by sorry

end NUMINAMATH_CALUDE_log_comparison_l2125_212572


namespace NUMINAMATH_CALUDE_min_value_of_sum_of_squares_l2125_212538

theorem min_value_of_sum_of_squares (x y z : ℝ) (h : 2*x + 3*y + 4*z = 1) :
  ∃ (m : ℝ), (∀ a b c : ℝ, 2*a + 3*b + 4*c = 1 → a^2 + b^2 + c^2 ≥ m) ∧
             (x^2 + y^2 + z^2 = m) ∧
             (m = 1/29) :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_sum_of_squares_l2125_212538


namespace NUMINAMATH_CALUDE_power_mod_eight_l2125_212592

theorem power_mod_eight : 3^23 ≡ 3 [MOD 8] := by sorry

end NUMINAMATH_CALUDE_power_mod_eight_l2125_212592


namespace NUMINAMATH_CALUDE_nancy_math_problems_l2125_212523

/-- The number of math problems Nancy had to solve -/
def math_problems : ℝ := 17.0

/-- The number of spelling problems Nancy had to solve -/
def spelling_problems : ℝ := 15.0

/-- The number of problems Nancy can finish in an hour -/
def problems_per_hour : ℝ := 8.0

/-- The number of hours it took Nancy to finish all problems -/
def total_hours : ℝ := 4.0

/-- Theorem stating that the number of math problems Nancy had is 17.0 -/
theorem nancy_math_problems : 
  math_problems = 
    problems_per_hour * total_hours - spelling_problems :=
by sorry

end NUMINAMATH_CALUDE_nancy_math_problems_l2125_212523


namespace NUMINAMATH_CALUDE_fourth_week_miles_l2125_212591

-- Define the number of weeks
def num_weeks : ℕ := 4

-- Define the number of days walked per week
def days_per_week : ℕ := 6

-- Define the miles walked per day for each week
def miles_per_day (week : ℕ) : ℕ :=
  if week < 4 then week else 0  -- The 4th week is unknown, so we set it to 0 initially

-- Define the total miles walked
def total_miles : ℕ := 60

-- Theorem to prove
theorem fourth_week_miles :
  ∃ (x : ℕ), 
    (miles_per_day 1 * days_per_week +
     miles_per_day 2 * days_per_week +
     miles_per_day 3 * days_per_week +
     x * days_per_week = total_miles) ∧
    x = 4 := by
  sorry

end NUMINAMATH_CALUDE_fourth_week_miles_l2125_212591


namespace NUMINAMATH_CALUDE_range_of_a_l2125_212506

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, Real.exp x - a * Real.log (a * x - a) + a > 0) →
  a > 0 →
  0 < a ∧ a < Real.exp 2 :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l2125_212506


namespace NUMINAMATH_CALUDE_min_value_trig_expression_min_value_trig_expression_achievable_l2125_212503

theorem min_value_trig_expression (α β : ℝ) :
  (3 * Real.cos α + 4 * Real.sin β - 5)^2 + (3 * Real.sin α + 4 * Real.cos β - 12)^2 ≥ 36 :=
by sorry

theorem min_value_trig_expression_achievable :
  ∃ (α β : ℝ), (3 * Real.cos α + 4 * Real.sin β - 5)^2 + (3 * Real.sin α + 4 * Real.cos β - 12)^2 = 36 :=
by sorry

end NUMINAMATH_CALUDE_min_value_trig_expression_min_value_trig_expression_achievable_l2125_212503


namespace NUMINAMATH_CALUDE_range_of_m_l2125_212590

-- Define set A
def A : Set ℝ := {x : ℝ | (x + 1) * (x - 6) ≤ 0}

-- Define set B
def B (m : ℝ) : Set ℝ := {x : ℝ | m - 1 ≤ x ∧ x ≤ 2 * m + 1}

-- Theorem statement
theorem range_of_m (m : ℝ) : 
  (A ∩ B m = B m) ↔ (m < -2 ∨ (0 ≤ m ∧ m ≤ 5/2)) :=
sorry

end NUMINAMATH_CALUDE_range_of_m_l2125_212590


namespace NUMINAMATH_CALUDE_cross_symmetry_l2125_212595

/-- Represents a square in the cross shape --/
inductive Square
| TopLeft
| TopRight
| Center
| BottomLeft
| BottomRight

/-- Represents a cross shape made of 5 squares --/
def CrossShape := Square → Square

/-- Defines the diagonal reflection operation --/
def diagonalReflection (c : CrossShape) : CrossShape :=
  fun s => match s with
  | Square.TopLeft => c Square.BottomRight
  | Square.TopRight => c Square.BottomLeft
  | Square.Center => c Square.Center
  | Square.BottomLeft => c Square.TopRight
  | Square.BottomRight => c Square.TopLeft

/-- Theorem: A cross shape is symmetric with respect to diagonal reflection
    if and only if it satisfies the specified swap conditions --/
theorem cross_symmetry (c : CrossShape) :
  (∀ s : Square, diagonalReflection c s = c s) ↔
  (c Square.TopRight = Square.BottomLeft ∧
   c Square.BottomLeft = Square.TopRight ∧
   c Square.TopLeft = Square.BottomRight ∧
   c Square.BottomRight = Square.TopLeft ∧
   c Square.Center = Square.Center) :=
by sorry


end NUMINAMATH_CALUDE_cross_symmetry_l2125_212595


namespace NUMINAMATH_CALUDE_expression_equivalence_l2125_212599

theorem expression_equivalence :
  (5 + 2) * (5^2 + 2^2) * (5^4 + 2^4) * (5^8 + 2^8) * (5^16 + 2^16) * (5^32 + 2^32) * (5^64 + 2^64) = 5^128 - 2^128 := by
  sorry

end NUMINAMATH_CALUDE_expression_equivalence_l2125_212599


namespace NUMINAMATH_CALUDE_rectangular_plot_poles_l2125_212566

/-- Calculates the number of fence poles needed for a rectangular plot -/
def fence_poles (length width pole_distance : ℕ) : ℕ :=
  (2 * (length + width)) / pole_distance

/-- Theorem: A 60m by 50m rectangular plot with poles 5m apart needs 44 poles -/
theorem rectangular_plot_poles :
  fence_poles 60 50 5 = 44 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_plot_poles_l2125_212566


namespace NUMINAMATH_CALUDE_rectangle_existence_l2125_212557

/-- A point in a 2D plane -/
structure Point := (x : ℝ) (y : ℝ)

/-- A line in a 2D plane -/
structure Line := (a : ℝ) (b : ℝ) (c : ℝ)

/-- A triangle defined by three points -/
structure Triangle := (K : Point) (L : Point) (M : Point)

/-- A rectangle defined by four points -/
structure Rectangle := (A : Point) (B : Point) (C : Point) (D : Point)

/-- Check if a point lies on a line -/
def Point.on_line (P : Point) (l : Line) : Prop :=
  l.a * P.x + l.b * P.y + l.c = 0

/-- Check if a point lies on the extension of a line segment -/
def Point.on_extension (P : Point) (A : Point) (B : Point) : Prop :=
  ∃ (t : ℝ), t > 1 ∧ P.x = A.x + t * (B.x - A.x) ∧ P.y = A.y + t * (B.y - A.y)

/-- Theorem: Given a triangle and a point on the extension of one side, 
    there exists a rectangle with vertices on the triangle's sides -/
theorem rectangle_existence (T : Triangle) (A : Point) :
  A.on_extension T.L T.K →
  ∃ (R : Rectangle),
    R.A = A ∧
    R.B.on_line (Line.mk (T.M.y - T.K.y) (T.K.x - T.M.x) (T.M.x * T.K.y - T.K.x * T.M.y)) ∧
    R.C.on_line (Line.mk (T.L.y - T.K.y) (T.K.x - T.L.x) (T.L.x * T.K.y - T.K.x * T.L.y)) ∧
    R.D.on_line (Line.mk (T.M.y - T.L.y) (T.L.x - T.M.x) (T.M.x * T.L.y - T.L.x * T.M.y)) :=
by
  sorry

end NUMINAMATH_CALUDE_rectangle_existence_l2125_212557


namespace NUMINAMATH_CALUDE_meal_with_tip_l2125_212579

/-- Calculates the total amount spent on a meal including tip -/
theorem meal_with_tip (lunch_cost : ℝ) (tip_percentage : ℝ) : 
  lunch_cost = 50.50 → tip_percentage = 20 → lunch_cost * (1 + tip_percentage / 100) = 60.60 := by
  sorry

end NUMINAMATH_CALUDE_meal_with_tip_l2125_212579


namespace NUMINAMATH_CALUDE_triangle_area_proof_l2125_212517

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    prove that the area of the triangle is √7/4 under the given conditions. -/
theorem triangle_area_proof (A B C : Real) (a b c : Real) :
  sinA = 2 * sinB →
  c = Real.sqrt 2 →
  cosC = 3 / 4 →
  (1 / 2) * a * b * sinC = Real.sqrt 7 / 4 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_proof_l2125_212517


namespace NUMINAMATH_CALUDE_pitcher_juice_distribution_l2125_212555

theorem pitcher_juice_distribution (C : ℝ) (h : C > 0) : 
  let juice_volume := (2 / 3) * C
  let cups := 6
  let juice_per_cup := juice_volume / cups
  (juice_per_cup / C) * 100 = 11.11 := by
  sorry

end NUMINAMATH_CALUDE_pitcher_juice_distribution_l2125_212555


namespace NUMINAMATH_CALUDE_noah_lights_on_time_l2125_212581

def bedroom_wattage : ℝ := 6
def office_wattage : ℝ := 3 * bedroom_wattage
def living_room_wattage : ℝ := 4 * bedroom_wattage
def total_energy_used : ℝ := 96

def total_wattage_per_hour : ℝ := bedroom_wattage + office_wattage + living_room_wattage

theorem noah_lights_on_time :
  total_energy_used / total_wattage_per_hour = 2 := by sorry

end NUMINAMATH_CALUDE_noah_lights_on_time_l2125_212581


namespace NUMINAMATH_CALUDE_volleyball_advancement_l2125_212514

def can_advance (k : ℕ) (t : ℕ) : Prop :=
  t ≤ k ∧ t * (t - 1) ≤ 2 * t * 1 ∧ (k - t) * (k - t - 1) ≥ 2 * (k - t) * (1 - t + 1)

theorem volleyball_advancement (k : ℕ) (h : k = 5 ∨ k = 6) :
  ∃ t : ℕ, t ≥ 0 ∧ t ≤ 3 ∧ can_advance k t :=
sorry

end NUMINAMATH_CALUDE_volleyball_advancement_l2125_212514


namespace NUMINAMATH_CALUDE_rectangle_diagonal_l2125_212565

/-- Given a rectangle with perimeter 72 meters and length-to-width ratio of 5:2,
    its diagonal length is 194/7 meters. -/
theorem rectangle_diagonal (length width : ℝ) : 
  2 * (length + width) = 72 →
  length / width = 5 / 2 →
  Real.sqrt (length^2 + width^2) = 194 / 7 := by
sorry

end NUMINAMATH_CALUDE_rectangle_diagonal_l2125_212565


namespace NUMINAMATH_CALUDE_correct_precipitation_forecast_interpretation_l2125_212584

/-- Represents the possible interpretations of a precipitation forecast --/
inductive PrecipitationForecastInterpretation
  | RainDuration
  | AreaCoverage
  | Probability
  | NoMeaningfulForecast

/-- Represents a precipitation forecast --/
structure PrecipitationForecast where
  probability : ℝ
  interpretation : PrecipitationForecastInterpretation

/-- Asserts that a given interpretation is correct for a precipitation forecast --/
def is_correct_interpretation (forecast : PrecipitationForecast) : Prop :=
  forecast.interpretation = PrecipitationForecastInterpretation.Probability

/-- Theorem: Given an 80% precipitation forecast, the correct interpretation is that there's an 80% chance of rain --/
theorem correct_precipitation_forecast_interpretation 
  (forecast : PrecipitationForecast) 
  (h : forecast.probability = 0.8) :
  is_correct_interpretation forecast :=
sorry

end NUMINAMATH_CALUDE_correct_precipitation_forecast_interpretation_l2125_212584


namespace NUMINAMATH_CALUDE_point_p_coordinates_l2125_212574

/-- Given a linear function and points A and P, proves that P satisfies the conditions of the problem -/
theorem point_p_coordinates (x y : ℝ) : 
  let f : ℝ → ℝ := λ t => -3/2 * t - 3
  let A : ℝ × ℝ := (-5, 0)
  let B : ℝ × ℝ := (f⁻¹ 0, 0)
  let P : ℝ × ℝ := (x, y)
  (P.2 = f P.1) →  -- P lies on the linear function
  (P ≠ B) →        -- P does not coincide with B
  (abs ((A.1 - B.1) * P.2) / 2 = 6) →  -- Area of triangle ABP is 6
  ((x = -14/3 ∧ y = 4) ∨ (x = 2/3 ∧ y = -4)) :=
by sorry

end NUMINAMATH_CALUDE_point_p_coordinates_l2125_212574


namespace NUMINAMATH_CALUDE_min_value_of_expression_l2125_212588

theorem min_value_of_expression (a b c : ℝ) 
  (sum_condition : a + b + c = -1)
  (product_condition : a * b * c ≤ -3) :
  (a * b + 1) / (a + b) + (b * c + 1) / (b + c) + (c * a + 1) / (c + a) ≥ 3 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_expression_l2125_212588


namespace NUMINAMATH_CALUDE_p_necessary_not_sufficient_for_q_l2125_212567

-- Define the basic structures
structure Line :=
  (id : ℕ)

structure Plane :=
  (id : ℕ)

-- Define the perpendicular relationships
def perpendicularToCountlessLines (l : Line) (p : Plane) : Prop :=
  sorry

def perpendicularToPlane (l : Line) : Plane → Prop :=
  sorry

-- Define the conditions p and q
def p (a : Line) (α : Plane) : Prop :=
  perpendicularToCountlessLines a α

def q (a : Line) (α : Plane) : Prop :=
  perpendicularToPlane a α

-- Theorem statement
theorem p_necessary_not_sufficient_for_q :
  (∀ (a : Line) (α : Plane), q a α → p a α) ∧
  (∃ (a : Line) (α : Plane), p a α ∧ ¬(q a α)) :=
sorry

end NUMINAMATH_CALUDE_p_necessary_not_sufficient_for_q_l2125_212567


namespace NUMINAMATH_CALUDE_raft_problem_l2125_212508

/-- The number of people who can fit on a raft under specific conditions -/
def raft_capacity (capacity_without_jackets : ℕ) (capacity_reduction : ℕ) (people_needing_jackets : ℕ) : ℕ :=
  let capacity_with_jackets := capacity_without_jackets - capacity_reduction
  min capacity_with_jackets (people_needing_jackets + (capacity_with_jackets - people_needing_jackets))

/-- Theorem stating that under the given conditions, 14 people can fit on the raft -/
theorem raft_problem : raft_capacity 21 7 8 = 14 := by
  sorry

end NUMINAMATH_CALUDE_raft_problem_l2125_212508


namespace NUMINAMATH_CALUDE_complex_power_sum_l2125_212582

theorem complex_power_sum : ∃ (i : ℂ), i^2 = -1 ∧ (1 - i)^2016 + (1 + i)^2016 = 2^1009 := by
  sorry

end NUMINAMATH_CALUDE_complex_power_sum_l2125_212582


namespace NUMINAMATH_CALUDE_collinear_sufficient_not_necessary_for_coplanar_l2125_212589

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Four points in 3D space -/
structure FourPoints where
  p1 : Point3D
  p2 : Point3D
  p3 : Point3D
  p4 : Point3D

/-- Predicate for three points being collinear -/
def threeCollinear (fp : FourPoints) : Prop :=
  sorry

/-- Predicate for four points being coplanar -/
def fourCoplanar (fp : FourPoints) : Prop :=
  sorry

/-- Theorem stating that three collinear points is a sufficient but not necessary condition for four coplanar points -/
theorem collinear_sufficient_not_necessary_for_coplanar :
  (∀ fp : FourPoints, threeCollinear fp → fourCoplanar fp) ∧
  (∃ fp : FourPoints, fourCoplanar fp ∧ ¬threeCollinear fp) :=
sorry

end NUMINAMATH_CALUDE_collinear_sufficient_not_necessary_for_coplanar_l2125_212589


namespace NUMINAMATH_CALUDE_family_savings_correct_l2125_212515

def income_tax_rate : ℝ := 0.13

def ivan_salary : ℝ := 55000
def vasilisa_salary : ℝ := 45000
def vasilisa_mother_salary : ℝ := 18000
def vasilisa_father_salary : ℝ := 20000
def son_state_stipend : ℝ := 3000
def son_non_state_stipend : ℝ := 15000

def vasilisa_mother_pension : ℝ := 10000

def monthly_expenses : ℝ := 74000

def net_income (gross_income : ℝ) : ℝ :=
  gross_income * (1 - income_tax_rate)

def total_income_before_may2018 : ℝ :=
  net_income ivan_salary + net_income vasilisa_salary + 
  net_income vasilisa_mother_salary + net_income vasilisa_father_salary + 
  son_state_stipend

def total_income_may_to_aug2018 : ℝ :=
  net_income ivan_salary + net_income vasilisa_salary + 
  vasilisa_mother_pension + net_income vasilisa_father_salary + 
  son_state_stipend

def total_income_from_sep2018 : ℝ :=
  net_income ivan_salary + net_income vasilisa_salary + 
  vasilisa_mother_pension + net_income vasilisa_father_salary + 
  son_state_stipend + net_income son_non_state_stipend

theorem family_savings_correct :
  (total_income_before_may2018 - monthly_expenses = 49060) ∧
  (total_income_may_to_aug2018 - monthly_expenses = 43400) ∧
  (total_income_from_sep2018 - monthly_expenses = 56450) := by
  sorry

end NUMINAMATH_CALUDE_family_savings_correct_l2125_212515


namespace NUMINAMATH_CALUDE_triangle_area_implies_q_value_l2125_212547

/-- Given a triangle DEF with vertices D(3, 15), E(15, 0), and F(0, q),
    if the area of the triangle is 30, then q = 12.5 -/
theorem triangle_area_implies_q_value :
  ∀ q : ℝ,
  let D : ℝ × ℝ := (3, 15)
  let E : ℝ × ℝ := (15, 0)
  let F : ℝ × ℝ := (0, q)
  let triangle_area := abs ((3 * q + 15 * q - 45) / 2)
  triangle_area = 30 → q = 12.5 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_implies_q_value_l2125_212547
