import Mathlib

namespace NUMINAMATH_CALUDE_pen_distribution_l3019_301927

theorem pen_distribution (num_students : ℕ) (red_pens : ℕ) (black_pens : ℕ) 
  (month1 : ℕ) (month2 : ℕ) (month3 : ℕ) (month4 : ℕ) : 
  num_students = 6 →
  red_pens = 85 →
  black_pens = 92 →
  month1 = 77 →
  month2 = 89 →
  month3 = 102 →
  month4 = 68 →
  (num_students * (red_pens + black_pens) - (month1 + month2 + month3 + month4)) / num_students = 121 := by
  sorry

#check pen_distribution

end NUMINAMATH_CALUDE_pen_distribution_l3019_301927


namespace NUMINAMATH_CALUDE_angle_with_special_supplementary_complementary_relation_l3019_301968

theorem angle_with_special_supplementary_complementary_relation :
  ∀ x : ℝ, (180 - x = 3 * (90 - x)) → x = 45 := by
  sorry

end NUMINAMATH_CALUDE_angle_with_special_supplementary_complementary_relation_l3019_301968


namespace NUMINAMATH_CALUDE_cube_volume_from_surface_area_l3019_301984

theorem cube_volume_from_surface_area (surface_area : ℝ) (volume : ℝ) :
  surface_area = 150 → volume = (surface_area / 6) ^ (3/2) → volume = 125 := by
  sorry

end NUMINAMATH_CALUDE_cube_volume_from_surface_area_l3019_301984


namespace NUMINAMATH_CALUDE_paintable_area_is_1572_l3019_301986

/-- Calculate the total paintable wall area for multiple identical bedrooms -/
def total_paintable_area (num_rooms : ℕ) (length width height : ℝ) (unpaintable_area : ℝ) : ℝ :=
  let total_wall_area := 2 * (length * height + width * height)
  let paintable_area_per_room := total_wall_area - unpaintable_area
  num_rooms * paintable_area_per_room

/-- Theorem stating that the total paintable wall area for the given conditions is 1572 square feet -/
theorem paintable_area_is_1572 :
  total_paintable_area 4 15 11 9 75 = 1572 := by
  sorry

#eval total_paintable_area 4 15 11 9 75

end NUMINAMATH_CALUDE_paintable_area_is_1572_l3019_301986


namespace NUMINAMATH_CALUDE_ab_product_theorem_l3019_301910

theorem ab_product_theorem (a b : ℝ) 
  (h1 : (27 : ℝ) ^ a = 3 ^ (10 * (b + 2)))
  (h2 : (125 : ℝ) ^ b = 5 ^ (a - 3)) : 
  a * b = 330 := by sorry

end NUMINAMATH_CALUDE_ab_product_theorem_l3019_301910


namespace NUMINAMATH_CALUDE_equation_solution_l3019_301973

theorem equation_solution : 
  {x : ℝ | x^2 + 6*x + 11 = |2*x + 5 - 5*x|} = {-6, -1} := by sorry

end NUMINAMATH_CALUDE_equation_solution_l3019_301973


namespace NUMINAMATH_CALUDE_square_sum_problem_l3019_301931

theorem square_sum_problem (x₁ y₁ x₂ y₂ : ℝ) 
  (h1 : x₁^2 + 5*x₂^2 = 10)
  (h2 : x₂*y₁ - x₁*y₂ = 5)
  (h3 : x₁*y₁ + 5*x₂*y₂ = Real.sqrt 105) :
  y₁^2 + 5*y₂^2 = 23 := by
sorry

end NUMINAMATH_CALUDE_square_sum_problem_l3019_301931


namespace NUMINAMATH_CALUDE_max_toys_buyable_l3019_301967

def initial_amount : ℕ := 57
def game_cost : ℕ := 27
def toy_cost : ℕ := 6

theorem max_toys_buyable : 
  (initial_amount - game_cost) / toy_cost = 5 :=
by sorry

end NUMINAMATH_CALUDE_max_toys_buyable_l3019_301967


namespace NUMINAMATH_CALUDE_total_weight_moved_tom_total_weight_l3019_301941

/-- Calculate the total weight Tom is moving with. -/
theorem total_weight_moved (tom_weight : ℝ) (hand_weight_ratio : ℝ) (vest_weight_ratio : ℝ) : ℝ :=
  let vest_weight := vest_weight_ratio * tom_weight
  let hand_weight := hand_weight_ratio * tom_weight
  let total_hand_weight := 2 * hand_weight
  total_hand_weight + vest_weight

/-- Prove that Tom is moving a total weight of 525 kg. -/
theorem tom_total_weight :
  total_weight_moved 150 1.5 0.5 = 525 := by
  sorry

end NUMINAMATH_CALUDE_total_weight_moved_tom_total_weight_l3019_301941


namespace NUMINAMATH_CALUDE_sundae_price_l3019_301901

/-- Proves that the price of each sundae is $0.60 given the conditions of the catering order --/
theorem sundae_price (ice_cream_bars sundaes : ℕ) (total_price ice_cream_price : ℚ) :
  ice_cream_bars = 200 →
  sundaes = 200 →
  total_price = 200 →
  ice_cream_price = 0.4 →
  (total_price - ice_cream_bars * ice_cream_price) / sundaes = 0.6 :=
by
  sorry

#eval (200 : ℚ) - 200 * 0.4  -- Expected output: 120
#eval 120 / 200              -- Expected output: 0.6

end NUMINAMATH_CALUDE_sundae_price_l3019_301901


namespace NUMINAMATH_CALUDE_inequality_solution_set_l3019_301962

theorem inequality_solution_set :
  let S := {x : ℝ | (3*x + 1)*(1 - 2*x) > 0}
  S = {x : ℝ | -1/3 < x ∧ x < 1/2} := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l3019_301962


namespace NUMINAMATH_CALUDE_range_of_a_theorem_l3019_301994

/-- Proposition p: For any x ∈ ℝ, x^2 - 2x > a -/
def p (a : ℝ) : Prop := ∀ x : ℝ, x^2 - 2*x > a

/-- Proposition q: The function f(x) = x^2 + 2ax + 2 - a has a zero point on ℝ -/
def q (a : ℝ) : Prop := ∃ x : ℝ, x^2 + 2*a*x + 2 - a = 0

/-- The range of values for a given the conditions -/
def range_of_a : Set ℝ := {a | a ∈ Set.Ioo (-2) (-1) ∨ a ∈ Set.Ici 1}

theorem range_of_a_theorem (a : ℝ) :
  (p a ∨ q a) ∧ ¬(p a ∧ q a) → a ∈ range_of_a := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_theorem_l3019_301994


namespace NUMINAMATH_CALUDE_unique_lock_code_satisfies_conditions_unique_lock_code_is_unique_l3019_301907

/-- Represents a seven-digit lock code -/
structure LockCode where
  digits : Fin 7 → Nat
  first_three_same : ∀ i j, i < 3 → j < 3 → digits i = digits j
  last_four_same : ∀ i j, 3 ≤ i → i < 7 → 3 ≤ j → j < 7 → digits i = digits j
  all_digits : ∀ i, digits i < 10

/-- The sum of digits in a lock code -/
def digit_sum (code : LockCode) : Nat :=
  (Finset.range 7).sum (λ i => code.digits i)

/-- The unique lock code satisfying all conditions -/
def unique_lock_code : LockCode where
  digits := λ i => if i < 3 then 3 else 7
  first_three_same := by sorry
  last_four_same := by sorry
  all_digits := by sorry

theorem unique_lock_code_satisfies_conditions :
  let s := digit_sum unique_lock_code
  (10 ≤ s ∧ s < 100) ∧
  (s / 10 = unique_lock_code.digits 0) ∧
  (s % 10 = unique_lock_code.digits 6) :=
by sorry

theorem unique_lock_code_is_unique (code : LockCode) :
  let s := digit_sum code
  (10 ≤ s ∧ s < 100) →
  (s / 10 = code.digits 0) →
  (s % 10 = code.digits 6) →
  code = unique_lock_code :=
by sorry

end NUMINAMATH_CALUDE_unique_lock_code_satisfies_conditions_unique_lock_code_is_unique_l3019_301907


namespace NUMINAMATH_CALUDE_max_profit_is_270000_l3019_301922

/-- Represents the production and profit details for a company's two products. -/
structure ProductionProblem where
  materialA_for_A : ℝ  -- tons of Material A needed for 1 ton of Product A
  materialB_for_A : ℝ  -- tons of Material B needed for 1 ton of Product A
  materialA_for_B : ℝ  -- tons of Material A needed for 1 ton of Product B
  materialB_for_B : ℝ  -- tons of Material B needed for 1 ton of Product B
  profit_A : ℝ         -- profit (in RMB) for 1 ton of Product A
  profit_B : ℝ         -- profit (in RMB) for 1 ton of Product B
  max_materialA : ℝ    -- maximum available tons of Material A
  max_materialB : ℝ    -- maximum available tons of Material B

/-- Calculates the maximum profit given the production constraints. -/
def maxProfit (p : ProductionProblem) : ℝ :=
  sorry

/-- States that the maximum profit for the given problem is 270,000 RMB. -/
theorem max_profit_is_270000 (p : ProductionProblem) 
  (h1 : p.materialA_for_A = 3)
  (h2 : p.materialB_for_A = 2)
  (h3 : p.materialA_for_B = 1)
  (h4 : p.materialB_for_B = 3)
  (h5 : p.profit_A = 50000)
  (h6 : p.profit_B = 30000)
  (h7 : p.max_materialA = 13)
  (h8 : p.max_materialB = 18) :
  maxProfit p = 270000 :=
by sorry

end NUMINAMATH_CALUDE_max_profit_is_270000_l3019_301922


namespace NUMINAMATH_CALUDE_negative_one_to_zero_equals_one_l3019_301930

theorem negative_one_to_zero_equals_one :
  (-1 : ℝ) ^ (0 : ℝ) = 1 := by sorry

end NUMINAMATH_CALUDE_negative_one_to_zero_equals_one_l3019_301930


namespace NUMINAMATH_CALUDE_arithmetic_expression_equality_l3019_301918

theorem arithmetic_expression_equality : 3 + 15 / 3 - 2^2 + 1 = 5 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_expression_equality_l3019_301918


namespace NUMINAMATH_CALUDE_probability_no_adjacent_standing_is_correct_l3019_301955

/-- Represents the number of valid arrangements for n people where no two adjacent people stand. -/
def validArrangements : ℕ → ℕ
  | 0 => 1
  | 1 => 2
  | n + 2 => validArrangements (n + 1) + validArrangements n

/-- The number of people sitting around the circular table. -/
def numPeople : ℕ := 10

/-- The probability of no two adjacent people standing when numPeople flip fair coins. -/
def probabilityNoAdjacentStanding : ℚ :=
  validArrangements numPeople / 2^numPeople

theorem probability_no_adjacent_standing_is_correct :
  probabilityNoAdjacentStanding = 123 / 1024 := by
  sorry

end NUMINAMATH_CALUDE_probability_no_adjacent_standing_is_correct_l3019_301955


namespace NUMINAMATH_CALUDE_gabes_original_seat_l3019_301919

/-- Represents the seats in the movie theater --/
inductive Seat
| one
| two
| three
| four
| five
| six
| seven

/-- Represents the friends --/
inductive Friend
| gabe
| flo
| dan
| cal
| bea
| eva
| hal

/-- Represents the seating arrangement --/
def Arrangement := Friend → Seat

/-- Returns the seat to the right of the given seat --/
def seatToRight (s : Seat) : Seat :=
  match s with
  | Seat.one => Seat.two
  | Seat.two => Seat.three
  | Seat.three => Seat.four
  | Seat.four => Seat.five
  | Seat.five => Seat.six
  | Seat.six => Seat.seven
  | Seat.seven => Seat.seven

/-- Returns the seat to the left of the given seat --/
def seatToLeft (s : Seat) : Seat :=
  match s with
  | Seat.one => Seat.one
  | Seat.two => Seat.one
  | Seat.three => Seat.two
  | Seat.four => Seat.three
  | Seat.five => Seat.four
  | Seat.six => Seat.five
  | Seat.seven => Seat.six

/-- Theorem stating Gabe's original seat --/
theorem gabes_original_seat (initial : Arrangement) (final : Arrangement) :
  (∀ (f : Friend), initial f ≠ initial Friend.gabe) →
  (final Friend.flo = seatToRight (seatToRight (seatToRight (initial Friend.flo)))) →
  (final Friend.dan = seatToLeft (initial Friend.dan)) →
  (final Friend.cal = initial Friend.cal) →
  (final Friend.bea = initial Friend.eva ∧ final Friend.eva = initial Friend.bea) →
  (final Friend.hal = seatToRight (initial Friend.gabe)) →
  (final Friend.gabe = Seat.one ∨ final Friend.gabe = Seat.seven) →
  initial Friend.gabe = Seat.three :=
by sorry


end NUMINAMATH_CALUDE_gabes_original_seat_l3019_301919


namespace NUMINAMATH_CALUDE_sum_neq_two_implies_both_neq_one_l3019_301977

theorem sum_neq_two_implies_both_neq_one (x y : ℝ) : x + y ≠ 2 → x ≠ 1 ∧ y ≠ 1 := by
  sorry

end NUMINAMATH_CALUDE_sum_neq_two_implies_both_neq_one_l3019_301977


namespace NUMINAMATH_CALUDE_triangle_area_problem_l3019_301940

theorem triangle_area_problem (base_small : ℝ) (base_large : ℝ) (area_small : ℝ) :
  base_small = 14 →
  base_large = 24 →
  area_small = 35 →
  let height_small := (2 * area_small) / base_small
  let height_large := (height_small * base_large) / base_small
  (1/2 : ℝ) * base_large * height_large = 144 :=
by sorry

end NUMINAMATH_CALUDE_triangle_area_problem_l3019_301940


namespace NUMINAMATH_CALUDE_geometric_progression_solution_l3019_301956

/-- A geometric progression is defined by its first term and common ratio -/
structure GeometricProgression where
  b₁ : ℚ  -- First term
  q : ℚ   -- Common ratio

/-- The nth term of a geometric progression -/
def GeometricProgression.nthTerm (gp : GeometricProgression) (n : ℕ) : ℚ :=
  gp.b₁ * gp.q ^ (n - 1)

theorem geometric_progression_solution :
  ∃ (gp : GeometricProgression),
    gp.nthTerm 3 = -1 ∧
    gp.nthTerm 6 = 27/8 ∧
    gp.b₁ = -4/9 ∧
    gp.q = -3/2 := by
  sorry

end NUMINAMATH_CALUDE_geometric_progression_solution_l3019_301956


namespace NUMINAMATH_CALUDE_F_of_4_f_of_5_equals_68_l3019_301912

-- Define the function f
def f (a : ℝ) : ℝ := a + 3

-- Define the function F
def F (a b : ℝ) : ℝ := b^2 + a

-- Theorem to prove
theorem F_of_4_f_of_5_equals_68 : F 4 (f 5) = 68 := by
  sorry

end NUMINAMATH_CALUDE_F_of_4_f_of_5_equals_68_l3019_301912


namespace NUMINAMATH_CALUDE_equation_satisfaction_l3019_301999

theorem equation_satisfaction (a b c : ℤ) :
  a = c ∧ b - 1 = a →
  a * (a - b) + b * (b - c) + c * (c - a) = 2 := by
  sorry

end NUMINAMATH_CALUDE_equation_satisfaction_l3019_301999


namespace NUMINAMATH_CALUDE_sum_of_roots_quadratic_sum_of_roots_specific_quadratic_l3019_301980

theorem sum_of_roots_quadratic (a b c : ℝ) (h : a ≠ 0) :
  let r₁ := (-b + Real.sqrt (b^2 - 4*a*c)) / (2*a)
  let r₂ := (-b - Real.sqrt (b^2 - 4*a*c)) / (2*a)
  (a * r₁^2 + b * r₁ + c = 0) ∧ (a * r₂^2 + b * r₂ + c = 0) →
  r₁ + r₂ = -b / a :=
by sorry

theorem sum_of_roots_specific_quadratic :
  let r₁ := (4 + Real.sqrt (16 - 12)) / 2
  let r₂ := (4 - Real.sqrt (16 - 12)) / 2
  (r₁^2 - 4*r₁ + 3 = 0) ∧ (r₂^2 - 4*r₂ + 3 = 0) →
  r₁ + r₂ = 4 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_roots_quadratic_sum_of_roots_specific_quadratic_l3019_301980


namespace NUMINAMATH_CALUDE_oxygen_atoms_in_compound_l3019_301991

-- Define atomic weights
def atomic_weight_H : ℝ := 1
def atomic_weight_Br : ℝ := 79.9
def atomic_weight_O : ℝ := 16

-- Define the molecular weight of the compound
def molecular_weight : ℝ := 129

-- Define the number of atoms for H and Br
def num_H : ℕ := 1
def num_Br : ℕ := 1

-- Theorem to prove
theorem oxygen_atoms_in_compound :
  ∃ (n : ℕ), n * atomic_weight_O = molecular_weight - (num_H * atomic_weight_H + num_Br * atomic_weight_Br) ∧ n = 3 := by
  sorry

end NUMINAMATH_CALUDE_oxygen_atoms_in_compound_l3019_301991


namespace NUMINAMATH_CALUDE_count_negative_numbers_l3019_301902

def number_list : List ℝ := [0, -2, 3, -0.1, -(-5)]

theorem count_negative_numbers : 
  (number_list.filter (λ x => x < 0)).length = 2 := by sorry

end NUMINAMATH_CALUDE_count_negative_numbers_l3019_301902


namespace NUMINAMATH_CALUDE_min_value_2a_plus_b_l3019_301961

theorem min_value_2a_plus_b (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a * b * (a + b) = 4) :
  2 * a + b ≥ 2 * Real.sqrt 3 ∧ ∃ a b, a > 0 ∧ b > 0 ∧ a * b * (a + b) = 4 ∧ 2 * a + b = 2 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_min_value_2a_plus_b_l3019_301961


namespace NUMINAMATH_CALUDE_pears_picked_by_keith_l3019_301936

/-- The number of pears Keith picked -/
def keiths_pears : ℝ := 0

theorem pears_picked_by_keith :
  let mikes_apples : ℝ := 7.0
  let nancys_eaten_apples : ℝ := 3.0
  let keiths_apples : ℝ := 6.0
  let apples_left : ℝ := 10.0
  keiths_pears = 0 := by sorry

end NUMINAMATH_CALUDE_pears_picked_by_keith_l3019_301936


namespace NUMINAMATH_CALUDE_original_quadratic_equation_l3019_301934

/-- The original quadratic equation given Xiaoming and Xiaohua's mistakes -/
theorem original_quadratic_equation :
  ∀ (a b c : ℝ),
  (∃ (x y : ℝ), x * y = -6 ∧ x + y = 2 - (-3)) →  -- Xiaoming's roots condition
  (∃ (u v : ℝ), u + v = -2 + 5) →                 -- Xiaohua's roots condition
  a = 1 →                                         -- Coefficient of x^2 is 1
  (a * X^2 + b * X + c = 0 ↔ X^2 - 3 * X - 6 = 0) -- The original equation
  := by sorry

end NUMINAMATH_CALUDE_original_quadratic_equation_l3019_301934


namespace NUMINAMATH_CALUDE_perimeter_approx_40_l3019_301943

/-- Represents a figure composed of three squares and one rectangle -/
structure CompositeFigure where
  square_side : ℝ
  total_area : ℝ

/-- Checks if the CompositeFigure satisfies the given conditions -/
def is_valid_figure (f : CompositeFigure) : Prop :=
  f.total_area = 150 ∧ 
  3 * f.square_side^2 + 2 * f.square_side^2 = f.total_area

/-- Calculates the perimeter of the CompositeFigure -/
def perimeter (f : CompositeFigure) : ℝ :=
  8 * f.square_side

/-- Theorem stating that the perimeter of a valid CompositeFigure is approximately 40 -/
theorem perimeter_approx_40 (f : CompositeFigure) (h : is_valid_figure f) :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 1 ∧ abs (perimeter f - 40) < ε :=
sorry

end NUMINAMATH_CALUDE_perimeter_approx_40_l3019_301943


namespace NUMINAMATH_CALUDE_factorial_square_root_simplification_l3019_301932

theorem factorial_square_root_simplification :
  Real.sqrt ((4 * 3 * 2 * 1) * (4 * 3 * 2 * 1) + 4) = 2 * Real.sqrt 145 := by
  sorry

end NUMINAMATH_CALUDE_factorial_square_root_simplification_l3019_301932


namespace NUMINAMATH_CALUDE_first_expression_second_expression_l3019_301990

-- First expression
theorem first_expression (x y : ℝ) : (-3 * x^2 * y)^3 = -27 * x^6 * y^3 := by sorry

-- Second expression
theorem second_expression (a : ℝ) : (-2*a - 1) * (2*a - 1) = 1 - 4*a^2 := by sorry

end NUMINAMATH_CALUDE_first_expression_second_expression_l3019_301990


namespace NUMINAMATH_CALUDE_zero_not_in_empty_set_l3019_301916

theorem zero_not_in_empty_set : 0 ∉ (∅ : Set ℕ) := by
  sorry

end NUMINAMATH_CALUDE_zero_not_in_empty_set_l3019_301916


namespace NUMINAMATH_CALUDE_rectangle_circle_area_ratio_l3019_301923

/-- Given a rectangle with perimeter equal to the circumference of a circle,
    and the length of the rectangle is twice its width,
    prove that the ratio of the area of the rectangle to the area of the circle is 2π/9. -/
theorem rectangle_circle_area_ratio (w : ℝ) (r : ℝ) (h1 : w > 0) (h2 : r > 0) :
  let l := 2 * w
  let rectangle_perimeter := 2 * l + 2 * w
  let circle_circumference := 2 * Real.pi * r
  let rectangle_area := l * w
  let circle_area := Real.pi * r^2
  rectangle_perimeter = circle_circumference →
  rectangle_area / circle_area = 2 * Real.pi / 9 := by
sorry

end NUMINAMATH_CALUDE_rectangle_circle_area_ratio_l3019_301923


namespace NUMINAMATH_CALUDE_bag_counter_problem_l3019_301974

theorem bag_counter_problem (Y X : ℕ) : 
  (Y > 0) →  -- Y is positive
  (X > 0) →  -- X is positive
  (Y / (Y + 10) = (Y + 2) / (X + Y + 12)) →  -- Proportion remains unchanged
  (Y * X = 20) →  -- Derived from the equality of proportions
  (Y = 1 ∨ Y = 2 ∨ Y = 4 ∨ Y = 5 ∨ Y = 10 ∨ Y = 20) :=
by sorry

end NUMINAMATH_CALUDE_bag_counter_problem_l3019_301974


namespace NUMINAMATH_CALUDE_expanded_product_terms_l3019_301970

theorem expanded_product_terms (a b c : ℕ) (ha : a = 6) (hb : b = 7) (hc : c = 5) :
  a * b * c = 210 := by
  sorry

end NUMINAMATH_CALUDE_expanded_product_terms_l3019_301970


namespace NUMINAMATH_CALUDE_chase_theorem_l3019_301979

/-- Represents the chase scenario between a greyhound and a rabbit. -/
structure ChaseScenario where
  n : ℕ  -- Initial lead of the rabbit in rabbit hops
  a : ℕ  -- Number of rabbit hops
  b : ℕ  -- Number of greyhound hops
  c : ℕ  -- Equivalent rabbit hops
  d : ℕ  -- Greyhound hops

/-- Calculates the number of hops the rabbit can make before being caught. -/
def rabbit_hops (scenario : ChaseScenario) : ℚ :=
  (scenario.a * scenario.d * scenario.n : ℚ) / (scenario.b * scenario.c - scenario.a * scenario.d)

/-- Calculates the number of hops the greyhound makes before catching the rabbit. -/
def greyhound_hops (scenario : ChaseScenario) : ℚ :=
  (scenario.b * scenario.d * scenario.n : ℚ) / (scenario.b * scenario.c - scenario.a * scenario.d)

/-- Theorem stating the correctness of the chase calculations. -/
theorem chase_theorem (scenario : ChaseScenario) 
  (h : scenario.b * scenario.c ≠ scenario.a * scenario.d) : 
  rabbit_hops scenario * (scenario.b * scenario.c : ℚ) / (scenario.a * scenario.d) = 
  greyhound_hops scenario * (scenario.c : ℚ) / scenario.d + scenario.n := by
  sorry

end NUMINAMATH_CALUDE_chase_theorem_l3019_301979


namespace NUMINAMATH_CALUDE_geometric_sequence_properties_l3019_301957

/-- A monotonically decreasing geometric sequence -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, 0 < q ∧ q < 1 ∧ ∀ n : ℕ, a (n + 1) = q * a n

theorem geometric_sequence_properties
  (a : ℕ → ℝ)
  (h_geom : GeometricSequence a)
  (h_sum : a 2 + a 3 + a 4 = 28)
  (h_mean : a 3 + 2 = (a 2 + a 4) / 2) :
  (∃ q : ℝ, ∀ n : ℕ, a n = (1/2)^(n - 6)) ∧
  (∃ q : ℝ, ∀ n : ℕ, a (n + 1) = (1/2) * a n) :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_properties_l3019_301957


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l3019_301997

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℕ, x > 0) ↔ (∃ x : ℕ, x ≤ 0) := by sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l3019_301997


namespace NUMINAMATH_CALUDE_matching_pair_probability_for_sue_l3019_301989

/-- Represents the number of pairs of shoes for each color --/
structure ShoePairs :=
  (black : Nat)
  (brown : Nat)
  (gray : Nat)
  (red : Nat)

/-- Calculates the probability of picking a matching pair of shoes --/
def matchingPairProbability (shoes : ShoePairs) : Rat :=
  let totalShoes := 2 * (shoes.black + shoes.brown + shoes.gray + shoes.red)
  let matchingPairs := 
    shoes.black * (shoes.black - 1) + 
    shoes.brown * (shoes.brown - 1) + 
    shoes.gray * (shoes.gray - 1) + 
    shoes.red * (shoes.red - 1)
  matchingPairs / (totalShoes * (totalShoes - 1))

theorem matching_pair_probability_for_sue : 
  let sueShoes : ShoePairs := { black := 7, brown := 4, gray := 3, red := 2 }
  matchingPairProbability sueShoes = 39 / 248 := by
  sorry

end NUMINAMATH_CALUDE_matching_pair_probability_for_sue_l3019_301989


namespace NUMINAMATH_CALUDE_system_solution_l3019_301945

theorem system_solution (x y k : ℝ) 
  (eq1 : 2 * x - y = 5 * k + 6)
  (eq2 : 4 * x + 7 * y = k)
  (eq3 : x + y = 2024) :
  k = 2023 := by
sorry

end NUMINAMATH_CALUDE_system_solution_l3019_301945


namespace NUMINAMATH_CALUDE_root_equation_implies_sum_l3019_301909

theorem root_equation_implies_sum (a b : ℝ) : 
  (1 : ℝ)^2 + a * (1 : ℝ) - b = 0 → a - b + 2023 = 2022 := by
  sorry

end NUMINAMATH_CALUDE_root_equation_implies_sum_l3019_301909


namespace NUMINAMATH_CALUDE_triangle_side_length_l3019_301960

-- Define the triangle ABC
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define the function to calculate the distance between two points
def distance (p1 p2 : ℝ × ℝ) : ℝ := sorry

-- Define the function to calculate the angle between three points
def angle (p1 p2 p3 : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem triangle_side_length (t : Triangle) :
  distance t.A t.B = 7 →
  distance t.A t.C = 5 →
  angle t.A t.C t.B = 2 * Real.pi / 3 →
  distance t.B t.C = 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_length_l3019_301960


namespace NUMINAMATH_CALUDE_triangle_abc_properties_l3019_301972

theorem triangle_abc_properties (A B C : ℝ) (a b c : ℝ) :
  0 < A ∧ 0 < B ∧ 0 < C ∧ 
  A + B + C = π ∧
  B = π/4 ∧ 
  b = Real.sqrt 10 ∧
  Real.cos C = 2 * Real.sqrt 5 / 5 →
  Real.sin A = 3 * Real.sqrt 10 / 10 ∧
  a = 3 * Real.sqrt 2 ∧
  1/2 * a * b * Real.sin C = 3 :=
by sorry

end NUMINAMATH_CALUDE_triangle_abc_properties_l3019_301972


namespace NUMINAMATH_CALUDE_halfway_between_fractions_l3019_301971

theorem halfway_between_fractions :
  (1 / 8 : ℚ) + ((1 / 3 : ℚ) - (1 / 8 : ℚ)) / 2 = 11 / 48 := by
  sorry

end NUMINAMATH_CALUDE_halfway_between_fractions_l3019_301971


namespace NUMINAMATH_CALUDE_toy_robot_shipment_l3019_301988

theorem toy_robot_shipment (displayed_percentage : ℚ) (stored : ℕ) : 
  displayed_percentage = 30 / 100 →
  stored = 140 →
  (1 - displayed_percentage) * 200 = stored :=
by sorry

end NUMINAMATH_CALUDE_toy_robot_shipment_l3019_301988


namespace NUMINAMATH_CALUDE_rachel_math_problems_l3019_301920

def problems_per_minute : ℕ := 5
def minutes_solved : ℕ := 12
def problems_second_day : ℕ := 16

theorem rachel_math_problems :
  problems_per_minute * minutes_solved + problems_second_day = 76 := by
  sorry

end NUMINAMATH_CALUDE_rachel_math_problems_l3019_301920


namespace NUMINAMATH_CALUDE_system_solution_proof_l3019_301996

theorem system_solution_proof (x y z : ℝ) : 
  x = 0.38 ∧ y = 0.992 ∧ z = -0.7176 →
  4 * x - 6 * y + 2 * z = -3 ∧
  8 * x + 3 * y - z = 5.3 ∧
  -x + 4 * y + 5 * z = 0 := by
sorry

end NUMINAMATH_CALUDE_system_solution_proof_l3019_301996


namespace NUMINAMATH_CALUDE_expression_evaluation_l3019_301949

theorem expression_evaluation (m : ℝ) (h : m = -Real.sqrt 5) :
  (2 * m - 1)^2 - (m - 5) * (m + 1) = 21 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3019_301949


namespace NUMINAMATH_CALUDE_sector_area_l3019_301976

/-- The area of a circular sector with a central angle of 150° and a radius of √3 is 5π/4 -/
theorem sector_area (α : Real) (r : Real) : 
  α = 150 * π / 180 →  -- Convert 150° to radians
  r = Real.sqrt 3 →
  (1 / 2) * α * r^2 = (5 * π) / 4 := by
  sorry

end NUMINAMATH_CALUDE_sector_area_l3019_301976


namespace NUMINAMATH_CALUDE_sqrt_product_simplification_l3019_301966

theorem sqrt_product_simplification (p : ℝ) : 
  Real.sqrt (12 * p) * Real.sqrt (20 * p) * Real.sqrt (15 * p^2) = 60 * p^2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_product_simplification_l3019_301966


namespace NUMINAMATH_CALUDE_ExistsSpecialSequence_l3019_301915

-- Define the sequence type
def InfiniteSequence := ℕ → ℕ

-- Define the properties of the sequence
def NoDivisibility (seq : InfiniteSequence) :=
  ∀ i j, i ≠ j → ¬(seq i ∣ seq j)

def CommonDivisorGreaterThanOne (seq : InfiniteSequence) :=
  ∀ i j, i ≠ j → ∃ k, k > 1 ∧ k ∣ seq i ∧ k ∣ seq j

def NoCommonDivisorGreaterThanOne (seq : InfiniteSequence) :=
  ¬∃ k, k > 1 ∧ (∀ i, k ∣ seq i)

-- Main theorem
theorem ExistsSpecialSequence :
  ∃ seq : InfiniteSequence,
    NoDivisibility seq ∧
    CommonDivisorGreaterThanOne seq ∧
    NoCommonDivisorGreaterThanOne seq :=
by sorry


end NUMINAMATH_CALUDE_ExistsSpecialSequence_l3019_301915


namespace NUMINAMATH_CALUDE_smallest_with_20_divisors_l3019_301933

/-- The number of positive divisors of a positive integer n -/
def num_divisors (n : ℕ+) : ℕ := sorry

/-- n has exactly 20 positive divisors -/
def has_20_divisors (n : ℕ+) : Prop := num_divisors n = 20

theorem smallest_with_20_divisors : 
  has_20_divisors 432 ∧ ∀ m : ℕ+, m < 432 → ¬(has_20_divisors m) := by sorry

end NUMINAMATH_CALUDE_smallest_with_20_divisors_l3019_301933


namespace NUMINAMATH_CALUDE_loan_interest_percentage_l3019_301951

theorem loan_interest_percentage 
  (loan_amount : ℝ) 
  (monthly_payment : ℝ) 
  (num_months : ℕ) 
  (h1 : loan_amount = 150)
  (h2 : monthly_payment = 15)
  (h3 : num_months = 11) : 
  (monthly_payment * num_months - loan_amount) / loan_amount * 100 = 10 := by
  sorry

end NUMINAMATH_CALUDE_loan_interest_percentage_l3019_301951


namespace NUMINAMATH_CALUDE_books_sold_on_thursday_l3019_301906

theorem books_sold_on_thursday (initial_stock : ℕ) (sold_monday : ℕ) (sold_tuesday : ℕ)
  (sold_wednesday : ℕ) (sold_friday : ℕ) (unsold : ℕ) :
  initial_stock = 800 →
  sold_monday = 60 →
  sold_tuesday = 10 →
  sold_wednesday = 20 →
  sold_friday = 66 →
  unsold = 600 →
  initial_stock - (sold_monday + sold_tuesday + sold_wednesday + sold_friday + unsold) = 44 :=
by sorry

end NUMINAMATH_CALUDE_books_sold_on_thursday_l3019_301906


namespace NUMINAMATH_CALUDE_intersection_empty_at_m_zero_l3019_301964

theorem intersection_empty_at_m_zero :
  ∃ m : ℝ, m = 0 ∧ (Set.Icc 0 1 : Set ℝ) ∩ {x : ℝ | x^2 - 2*x + m > 0} = ∅ :=
by sorry

end NUMINAMATH_CALUDE_intersection_empty_at_m_zero_l3019_301964


namespace NUMINAMATH_CALUDE_parallelogram_area_l3019_301914

/-- The area of a parallelogram with base 12 cm and height 48 cm is 576 square centimeters. -/
theorem parallelogram_area : 
  ∀ (base height area : ℝ), 
    base = 12 → 
    height = 48 → 
    area = base * height → 
    area = 576 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_area_l3019_301914


namespace NUMINAMATH_CALUDE_mapping_properties_l3019_301946

-- Define the sets A and B
variable {A B : Type}

-- Define the mapping f from A to B
variable (f : A → B)

-- Theorem stating the properties of the mapping
theorem mapping_properties :
  (∀ a : A, ∃! b : B, f a = b) ∧
  (∃ a₁ a₂ : A, a₁ ≠ a₂ ∧ f a₁ = f a₂) :=
by sorry

end NUMINAMATH_CALUDE_mapping_properties_l3019_301946


namespace NUMINAMATH_CALUDE_p_q_ratio_equals_ways_ratio_l3019_301925

/-- The number of balls -/
def n : ℕ := 20

/-- The number of bins -/
def k : ℕ := 4

/-- The probability of a 3-5-6-6 distribution -/
def p : ℚ := sorry

/-- The probability of a 5-5-5-5 distribution -/
def q : ℚ := sorry

/-- The number of ways to distribute n balls into k bins with a given distribution -/
def ways_to_distribute (n : ℕ) (k : ℕ) (distribution : List ℕ) : ℕ := sorry

/-- The ratio of p to q is equal to the ratio of the number of ways to achieve each distribution -/
theorem p_q_ratio_equals_ways_ratio : 
  p / q = (ways_to_distribute n k [3, 5, 6, 6] * 12) / ways_to_distribute n k [5, 5, 5, 5] := by
  sorry

end NUMINAMATH_CALUDE_p_q_ratio_equals_ways_ratio_l3019_301925


namespace NUMINAMATH_CALUDE_prob_three_correct_five_l3019_301982

/-- The number of houses and packages --/
def n : ℕ := 5

/-- The probability of exactly 3 out of n packages being delivered to the correct houses --/
def prob_three_correct (n : ℕ) : ℚ :=
  (n.choose 3 : ℚ) * (1 / n) * (1 / (n - 1)) * (1 / (n - 2)) * (1 / 2)

/-- Theorem stating that the probability of exactly 3 out of 5 packages 
    being delivered to the correct houses is 1/12 --/
theorem prob_three_correct_five : prob_three_correct n = 1 / 12 := by
  sorry

end NUMINAMATH_CALUDE_prob_three_correct_five_l3019_301982


namespace NUMINAMATH_CALUDE_max_ab_bisecting_line_l3019_301981

theorem max_ab_bisecting_line (a b : ℝ) : 
  (∀ x y : ℝ, 2*a*x - b*y + 2 = 0 → (x+1)^2 + (y-2)^2 = 4) → 
  (a * b ≤ 1/4) ∧ (∃ a₀ b₀ : ℝ, a₀ * b₀ = 1/4 ∧ 
    (∀ x y : ℝ, 2*a₀*x - b₀*y + 2 = 0 → (x+1)^2 + (y-2)^2 = 4)) :=
by sorry

end NUMINAMATH_CALUDE_max_ab_bisecting_line_l3019_301981


namespace NUMINAMATH_CALUDE_inequality_proof_l3019_301954

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a / Real.sqrt (a^2 + 8*b*c)) + (b / Real.sqrt (b^2 + 8*c*a)) + (c / Real.sqrt (c^2 + 8*a*b)) ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3019_301954


namespace NUMINAMATH_CALUDE_negation_equivalence_l3019_301917

theorem negation_equivalence :
  (¬ ∃ x ∈ Set.Ioo (-1 : ℝ) 1, x^2 + 2*x ≤ 1) ↔ 
  (∀ x ∈ Set.Ioo (-1 : ℝ) 1, x^2 + 2*x > 1) :=
by sorry

end NUMINAMATH_CALUDE_negation_equivalence_l3019_301917


namespace NUMINAMATH_CALUDE_ryan_learning_days_l3019_301947

def daily_english_hours : ℕ := 6
def daily_chinese_hours : ℕ := 7
def total_hours : ℕ := 65

theorem ryan_learning_days : 
  total_hours / (daily_english_hours + daily_chinese_hours) = 5 := by
sorry

end NUMINAMATH_CALUDE_ryan_learning_days_l3019_301947


namespace NUMINAMATH_CALUDE_consecutive_points_length_l3019_301992

/-- Given 5 consecutive points on a straight line, prove that ab = 5 -/
theorem consecutive_points_length (a b c d e : ℝ) : 
  (c - b = 3 * (d - c)) →  -- bc = 3 * cd
  (e - d = 8) →            -- de = 8
  (c - a = 11) →           -- ac = 11
  (e - a = 21) →           -- ae = 21
  (b - a = 5) :=           -- ab = 5
by sorry

end NUMINAMATH_CALUDE_consecutive_points_length_l3019_301992


namespace NUMINAMATH_CALUDE_johns_purchase_cost_l3019_301928

/-- Calculates the total cost of John's metal purchase in USD -/
def total_cost (silver_oz : ℝ) (gold_oz : ℝ) (platinum_oz : ℝ) 
                (silver_price_usd : ℝ) (gold_multiplier : ℝ) 
                (platinum_price_gbp : ℝ) (usd_gbp_rate : ℝ) : ℝ :=
  let silver_cost := silver_oz * silver_price_usd
  let gold_cost := gold_oz * (silver_price_usd * gold_multiplier)
  let platinum_cost := platinum_oz * (platinum_price_gbp * usd_gbp_rate)
  silver_cost + gold_cost + platinum_cost

/-- Theorem stating that John's total cost is $5780.5 -/
theorem johns_purchase_cost : 
  total_cost 2.5 3.5 4.5 25 60 80 1.3 = 5780.5 := by
  sorry

end NUMINAMATH_CALUDE_johns_purchase_cost_l3019_301928


namespace NUMINAMATH_CALUDE_cricket_target_runs_l3019_301944

/-- Calculates the target number of runs in a cricket game -/
def targetRuns (totalOvers runRateFirst8 runRateRemaining : ℕ) : ℕ :=
  let runsFirst8 := (runRateFirst8 * 8) / 10
  let runsRemaining := (runRateRemaining * 20) / 10
  runsFirst8 + runsRemaining

/-- Theorem stating the target number of runs for the given conditions -/
theorem cricket_target_runs :
  targetRuns 28 23 120 = 259 := by
  sorry

#eval targetRuns 28 23 120

end NUMINAMATH_CALUDE_cricket_target_runs_l3019_301944


namespace NUMINAMATH_CALUDE_dune_buggy_speed_l3019_301950

theorem dune_buggy_speed (S : ℝ) : 
  (1/3 : ℝ) * S + (1/3 : ℝ) * (S + 12) + (1/3 : ℝ) * (S - 18) = 58 → S = 60 := by
  sorry

end NUMINAMATH_CALUDE_dune_buggy_speed_l3019_301950


namespace NUMINAMATH_CALUDE_sqrt_a_div_sqrt_b_l3019_301937

theorem sqrt_a_div_sqrt_b (a b : ℝ) (h : (3/5)^2 + (2/7)^2 / ((2/9)^2 + (1/6)^2) = 28*a/(45*b)) :
  Real.sqrt a / Real.sqrt b = 2 * Real.sqrt 105 / 7 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_a_div_sqrt_b_l3019_301937


namespace NUMINAMATH_CALUDE_existence_of_m_satisfying_inequality_l3019_301985

theorem existence_of_m_satisfying_inequality (a t : ℝ) 
  (ha : a ∈ Set.Icc (-1 : ℝ) 1)
  (ht : t ∈ Set.Icc (-1 : ℝ) 1) :
  ∃ m : ℝ, (∀ x₁ x₂ : ℝ, 
    x₁ ≠ 0 ∧ x₂ ≠ 0 ∧ 
    (4 * x₁ + a * x₁^2 - (2/3) * x₁^3 = 2 * x₁ + (1/3) * x₁^3) ∧
    (4 * x₂ + a * x₂^2 - (2/3) * x₂^3 = 2 * x₂ + (1/3) * x₂^3) →
    m^2 + t * m + 1 ≥ |x₁ - x₂|) ∧
  (m ≥ 2 ∨ m ≤ -2) := by
  sorry

end NUMINAMATH_CALUDE_existence_of_m_satisfying_inequality_l3019_301985


namespace NUMINAMATH_CALUDE_line_segment_endpoint_l3019_301908

theorem line_segment_endpoint (x : ℝ) : 
  x > 0 → 
  Real.sqrt ((x - 2)^2 + (7 - 1)^2) = 8 → 
  x = 2 + 2 * Real.sqrt 7 := by
sorry

end NUMINAMATH_CALUDE_line_segment_endpoint_l3019_301908


namespace NUMINAMATH_CALUDE_table_area_proof_l3019_301952

theorem table_area_proof (total_runner_area : ℝ) (coverage_percentage : ℝ) 
  (two_layer_area : ℝ) (three_layer_area : ℝ) 
  (h1 : total_runner_area = 224) 
  (h2 : coverage_percentage = 0.8)
  (h3 : two_layer_area = 24)
  (h4 : three_layer_area = 30) : 
  ∃ (table_area : ℝ), table_area = 175 ∧ 
    coverage_percentage * table_area = 
      (total_runner_area - 2 * two_layer_area - 3 * three_layer_area) + 
      two_layer_area + three_layer_area := by
  sorry

end NUMINAMATH_CALUDE_table_area_proof_l3019_301952


namespace NUMINAMATH_CALUDE_least_positive_integer_for_reducible_fraction_l3019_301905

theorem least_positive_integer_for_reducible_fraction :
  ∃ (n : ℕ), n > 0 ∧ 
  (∀ m : ℕ, m > 0 → m < n → ¬(∃ (k : ℕ), k > 1 ∧ k ∣ (m - 10) ∧ k ∣ (9*m + 11))) ∧
  (∃ (k : ℕ), k > 1 ∧ k ∣ (n - 10) ∧ k ∣ (9*n + 11)) ∧
  n = 111 :=
by sorry

end NUMINAMATH_CALUDE_least_positive_integer_for_reducible_fraction_l3019_301905


namespace NUMINAMATH_CALUDE_inequality_holds_l3019_301975

theorem inequality_holds (a b : ℝ) (h1 : a > 1) (h2 : 1 > b) (h3 : b > 0) :
  (1 / Real.log a) > (1 / Real.log b) := by
  sorry

end NUMINAMATH_CALUDE_inequality_holds_l3019_301975


namespace NUMINAMATH_CALUDE_sphere_radius_when_volume_equals_surface_area_l3019_301978

theorem sphere_radius_when_volume_equals_surface_area :
  ∀ r : ℝ,
  (4 / 3) * Real.pi * r^3 = 4 * Real.pi * r^2 →
  r = 3 :=
by sorry

end NUMINAMATH_CALUDE_sphere_radius_when_volume_equals_surface_area_l3019_301978


namespace NUMINAMATH_CALUDE_smallest_solution_congruence_l3019_301965

theorem smallest_solution_congruence :
  ∃ (x : ℕ), x > 0 ∧ (5 * x) % 31 = 22 % 31 ∧
  ∀ (y : ℕ), y > 0 → (5 * y) % 31 = 22 % 31 → x ≤ y :=
by sorry

end NUMINAMATH_CALUDE_smallest_solution_congruence_l3019_301965


namespace NUMINAMATH_CALUDE_dilation_determinant_l3019_301913

/-- A dilation matrix with scale factor k -/
def dilationMatrix (k : ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  !![k, 0; 0, k]

/-- The determinant of a 2x2 matrix -/
def det2x2 (M : Matrix (Fin 2) (Fin 2) ℝ) : ℝ :=
  M 0 0 * M 1 1 - M 0 1 * M 1 0

theorem dilation_determinant :
  let E := dilationMatrix 12
  det2x2 E = 144 := by sorry

end NUMINAMATH_CALUDE_dilation_determinant_l3019_301913


namespace NUMINAMATH_CALUDE_conditional_probability_coin_flips_l3019_301900

-- Define the sample space for two coin flips
def CoinFlip := Bool × Bool

-- Define the probability measure
noncomputable def P : Set CoinFlip → ℝ := sorry

-- Define event A: heads on the first flip
def A : Set CoinFlip := {x | x.1 = true}

-- Define event B: heads on the second flip
def B : Set CoinFlip := {x | x.2 = true}

-- Define the intersection of events A and B
def AB : Set CoinFlip := A ∩ B

-- State the theorem
theorem conditional_probability_coin_flips :
  P B / P A = 1 / 2 := by sorry

end NUMINAMATH_CALUDE_conditional_probability_coin_flips_l3019_301900


namespace NUMINAMATH_CALUDE_quadratic_sum_l3019_301987

/-- Given a quadratic function f(x) = 8x^2 - 48x - 288, when expressed in the form a(x+b)^2 + c,
    the sum of a, b, and c is -355. -/
theorem quadratic_sum (f : ℝ → ℝ) (a b c : ℝ) :
  (∀ x, f x = 8 * x^2 - 48 * x - 288) →
  (∀ x, f x = a * (x + b)^2 + c) →
  a + b + c = -355 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_sum_l3019_301987


namespace NUMINAMATH_CALUDE_target_primes_are_5_13_17_29_l3019_301948

/-- The set of prime numbers less than 30 -/
def primes_less_than_30 : Set ℕ :=
  {p | p < 30 ∧ Nat.Prime p}

/-- A function that checks if a number becomes a multiple of 4 after adding 3 -/
def becomes_multiple_of_4 (n : ℕ) : Prop :=
  (n + 3) % 4 = 0

/-- The set of prime numbers less than 30 that become multiples of 4 after adding 3 -/
def target_primes : Set ℕ :=
  {p ∈ primes_less_than_30 | becomes_multiple_of_4 p}

theorem target_primes_are_5_13_17_29 : target_primes = {5, 13, 17, 29} := by
  sorry

end NUMINAMATH_CALUDE_target_primes_are_5_13_17_29_l3019_301948


namespace NUMINAMATH_CALUDE_auction_tv_initial_price_l3019_301921

/-- Given an auction event where:
    - The price of a TV increased by 2/5 times its initial price
    - The price of a phone, initially $400, increased by 40%
    - The total amount received after sale is $1260
    Prove that the initial price of the TV was $500 -/
theorem auction_tv_initial_price (tv_initial : ℝ) (phone_initial : ℝ) (total : ℝ) :
  phone_initial = 400 →
  total = 1260 →
  total = (tv_initial + 2/5 * tv_initial) + (phone_initial + 0.4 * phone_initial) →
  tv_initial = 500 := by
  sorry


end NUMINAMATH_CALUDE_auction_tv_initial_price_l3019_301921


namespace NUMINAMATH_CALUDE_committee_selection_ways_l3019_301926

theorem committee_selection_ways (n m : ℕ) (hn : n = 30) (hm : m = 5) :
  Nat.choose n m = 54810 := by
  sorry

end NUMINAMATH_CALUDE_committee_selection_ways_l3019_301926


namespace NUMINAMATH_CALUDE_problem_one_problem_two_l3019_301911

-- Problem 1
theorem problem_one : Real.sqrt 12 - Real.sqrt 3 + 3 * Real.sqrt (1/3) = Real.sqrt 3 + 3 := by
  sorry

-- Problem 2
theorem problem_two : Real.sqrt 18 / Real.sqrt 6 * Real.sqrt 3 = 3 := by
  sorry

end NUMINAMATH_CALUDE_problem_one_problem_two_l3019_301911


namespace NUMINAMATH_CALUDE_angle_supplement_theorem_l3019_301998

-- Define a type for angles in degrees and minutes
structure Angle :=
  (degrees : ℕ)
  (minutes : ℕ)

-- Define the complement of an angle
def complement (α : Angle) : Angle :=
  { degrees := 90 - α.degrees - 1,
    minutes := 60 - α.minutes }

-- Define the supplement of an angle
def supplement (α : Angle) : Angle :=
  { degrees := 180 - α.degrees - 1,
    minutes := 60 - α.minutes }

theorem angle_supplement_theorem (α : Angle) :
  complement α = { degrees := 54, minutes := 32 } →
  supplement α = { degrees := 144, minutes := 32 } :=
by sorry

end NUMINAMATH_CALUDE_angle_supplement_theorem_l3019_301998


namespace NUMINAMATH_CALUDE_hyperbola_focal_distance_l3019_301938

/-- The focal distance of a hyperbola with equation x²/20 - y²/5 = 1 is 10 -/
theorem hyperbola_focal_distance : 
  ∃ (c : ℝ), c > 0 ∧ c^2 = 25 ∧ 2*c = 10 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_focal_distance_l3019_301938


namespace NUMINAMATH_CALUDE_valid_arrangements_count_l3019_301995

def male_teachers : ℕ := 5
def female_teachers : ℕ := 4
def total_teachers : ℕ := male_teachers + female_teachers
def head_teachers_needed : ℕ := 3

def valid_arrangements : ℕ := 
  Nat.factorial total_teachers / Nat.factorial (total_teachers - head_teachers_needed) -
  (Nat.factorial male_teachers / Nat.factorial (male_teachers - head_teachers_needed) +
   Nat.factorial female_teachers / Nat.factorial (female_teachers - head_teachers_needed))

theorem valid_arrangements_count : valid_arrangements = 420 := by
  sorry

end NUMINAMATH_CALUDE_valid_arrangements_count_l3019_301995


namespace NUMINAMATH_CALUDE_negation_of_existence_negation_of_specific_proposition_l3019_301904

theorem negation_of_existence (P : ℝ → Prop) :
  (¬ ∃ x, x < 0 ∧ P x) ↔ (∀ x, x < 0 → ¬ P x) :=
by sorry

-- The specific proposition
def proposition (x : ℝ) : Prop := 3 * x < 4 * x

theorem negation_of_specific_proposition :
  (¬ ∃ x, x < 0 ∧ proposition x) ↔ (∀ x, x < 0 → 3 * x ≥ 4 * x) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_existence_negation_of_specific_proposition_l3019_301904


namespace NUMINAMATH_CALUDE_contest_participants_l3019_301958

theorem contest_participants (P : ℕ) 
  (h1 : (P / 2 : ℚ) = P * (1 / 2 : ℚ)) 
  (h2 : (P / 2 + P / 2 / 7 : ℚ) = P * (57.14285714285714 / 100 : ℚ)) : 
  ∃ k : ℕ, P = 7 * k :=
sorry

end NUMINAMATH_CALUDE_contest_participants_l3019_301958


namespace NUMINAMATH_CALUDE_prob_independent_of_trials_l3019_301903

/-- A random event. -/
structure RandomEvent where
  /-- The probability of the event occurring in a single trial. -/
  probability : ℝ
  /-- Assumption that the probability is between 0 and 1. -/
  prob_nonneg : 0 ≤ probability
  prob_le_one : probability ≤ 1

/-- The probability of the event not occurring in n trials. -/
def prob_not_occur (E : RandomEvent) (n : ℕ) : ℝ :=
  (1 - E.probability) ^ n

/-- The probability of the event occurring at least once in n trials. -/
def prob_occur_at_least_once (E : RandomEvent) (n : ℕ) : ℝ :=
  1 - prob_not_occur E n

/-- Theorem stating that the probability of a random event occurring
    is independent of the number of trials. -/
theorem prob_independent_of_trials (E : RandomEvent) :
  ∀ n : ℕ, prob_occur_at_least_once E (n + 1) - prob_occur_at_least_once E n = E.probability * (prob_not_occur E n) :=
sorry


end NUMINAMATH_CALUDE_prob_independent_of_trials_l3019_301903


namespace NUMINAMATH_CALUDE_additional_carrots_is_38_l3019_301959

/-- The number of additional carrots picked by Carol and her mother -/
def additional_carrots (carol_carrots mother_carrots total_bad_carrots : ℝ) : ℝ :=
  total_bad_carrots - (carol_carrots + mother_carrots)

/-- Theorem stating that the number of additional carrots picked is 38 -/
theorem additional_carrots_is_38 :
  additional_carrots 29 16 83 = 38 := by
  sorry

end NUMINAMATH_CALUDE_additional_carrots_is_38_l3019_301959


namespace NUMINAMATH_CALUDE_souvenir_profit_maximization_l3019_301942

/-- Represents the problem of maximizing profit for a souvenir seller --/
theorem souvenir_profit_maximization
  (cost_price : ℕ)
  (initial_price : ℕ)
  (initial_sales : ℕ)
  (price_increase : ℕ → ℕ)
  (sales_decrease : ℕ → ℕ)
  (profit : ℕ → ℕ)
  (h_cost : cost_price = 5)
  (h_initial_price : initial_price = 9)
  (h_initial_sales : initial_sales = 32)
  (h_price_increase : ∀ x, price_increase x = x)
  (h_sales_decrease : ∀ x, sales_decrease x = 4 * x)
  (h_profit : ∀ x, profit x = (initial_price + price_increase x - cost_price) * (initial_sales - sales_decrease x)) :
  ∃ (optimal_increase : ℕ),
    optimal_increase = 2 ∧
    ∀ x, x ≠ optimal_increase → profit x ≤ profit optimal_increase ∧
    profit optimal_increase = 144 := by
  sorry


end NUMINAMATH_CALUDE_souvenir_profit_maximization_l3019_301942


namespace NUMINAMATH_CALUDE_typists_problem_l3019_301993

theorem typists_problem (n : ℕ) : 
  (∃ (k : ℕ), k > 0 ∧ k * n = 46) → 
  (30 * (3 * 46) / n = 207) → 
  n = 20 := by
sorry

end NUMINAMATH_CALUDE_typists_problem_l3019_301993


namespace NUMINAMATH_CALUDE_least_product_of_distinct_primes_above_50_l3019_301924

theorem least_product_of_distinct_primes_above_50 : 
  ∃ (p q : ℕ), 
    p.Prime ∧ 
    q.Prime ∧ 
    p ≠ q ∧ 
    p > 50 ∧ 
    q > 50 ∧ 
    p * q = 3127 ∧ 
    ∀ (r s : ℕ), r.Prime → s.Prime → r ≠ s → r > 50 → s > 50 → r * s ≥ 3127 :=
by sorry

end NUMINAMATH_CALUDE_least_product_of_distinct_primes_above_50_l3019_301924


namespace NUMINAMATH_CALUDE_stating_height_represents_frequency_ratio_l3019_301969

/-- Represents a frequency distribution histogram --/
structure FrequencyHistogram where
  /-- The height of a bar in the histogram --/
  height : ℝ → ℝ
  /-- The frequency of individuals in a group --/
  frequency : ℝ → ℝ
  /-- The class interval for a group --/
  classInterval : ℝ → ℝ

/-- 
Theorem stating that the height of a frequency distribution histogram
represents the ratio of the frequency to the class interval
-/
theorem height_represents_frequency_ratio (h : FrequencyHistogram) :
  ∀ x, h.height x = h.frequency x / h.classInterval x := by
  sorry

end NUMINAMATH_CALUDE_stating_height_represents_frequency_ratio_l3019_301969


namespace NUMINAMATH_CALUDE_distinct_points_on_curve_l3019_301963

theorem distinct_points_on_curve : ∃ (a b : ℝ), 
  a ≠ b ∧ 
  (a^3 + Real.sqrt e^4 = 2 * (Real.sqrt e)^2 * a + 1) ∧
  (b^3 + Real.sqrt e^4 = 2 * (Real.sqrt e)^2 * b + 1) ∧
  |a - b| = 3 := by
  sorry

end NUMINAMATH_CALUDE_distinct_points_on_curve_l3019_301963


namespace NUMINAMATH_CALUDE_number_of_divisors_180_l3019_301953

theorem number_of_divisors_180 : ∃ (n : ℕ), n = 18 ∧ 
  (∀ d : ℕ, d > 0 ∧ (180 % d = 0) ↔ d ∈ Finset.range n) :=
by
  sorry

end NUMINAMATH_CALUDE_number_of_divisors_180_l3019_301953


namespace NUMINAMATH_CALUDE_cos_negative_sixty_degrees_l3019_301929

theorem cos_negative_sixty_degrees : Real.cos (-(60 * π / 180)) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_negative_sixty_degrees_l3019_301929


namespace NUMINAMATH_CALUDE_rotation_result_l3019_301983

-- Define the shapes
inductive Shape
  | SmallCircle
  | Triangle
  | Square
  | Pentagon

-- Define the rotation directions
inductive RotationDirection
  | Clockwise
  | Counterclockwise

-- Define the configuration of shapes
structure Configuration :=
  (smallCircle : ℝ)  -- Angle of rotation for small circle
  (triangle : ℝ)     -- Angle of rotation for triangle
  (pentagon : ℝ)     -- Angle of rotation for pentagon
  (overall : ℝ)      -- Overall rotation of the configuration

-- Define the rotation function
def rotate (shape : Shape) (angle : ℝ) (direction : RotationDirection) : ℝ :=
  match direction with
  | RotationDirection.Clockwise => angle
  | RotationDirection.Counterclockwise => -angle

-- Define the initial configuration
def initialConfig : Configuration :=
  { smallCircle := 0, triangle := 0, pentagon := 0, overall := 0 }

-- Define the final configuration after rotations
def finalConfig (initial : Configuration) : Configuration :=
  { smallCircle := initial.smallCircle + rotate Shape.SmallCircle 45 RotationDirection.Counterclockwise,
    triangle := initial.triangle + rotate Shape.Triangle 180 RotationDirection.Clockwise,
    pentagon := initial.pentagon + rotate Shape.Pentagon 120 RotationDirection.Clockwise,
    overall := initial.overall + rotate Shape.Square 90 RotationDirection.Clockwise }

-- Theorem statement
theorem rotation_result :
  let final := finalConfig initialConfig
  final.smallCircle = -45 ∧
  final.triangle = 180 ∧
  final.pentagon = 120 ∧
  final.overall = 90 :=
by sorry

end NUMINAMATH_CALUDE_rotation_result_l3019_301983


namespace NUMINAMATH_CALUDE_average_weight_ab_is_40_l3019_301935

def average_weight_abc : ℝ := 42
def average_weight_bc : ℝ := 43
def weight_b : ℝ := 40

theorem average_weight_ab_is_40 :
  let weight_c := 2 * average_weight_bc - weight_b
  let weight_a := 3 * average_weight_abc - weight_b - weight_c
  (weight_a + weight_b) / 2 = 40 := by sorry

end NUMINAMATH_CALUDE_average_weight_ab_is_40_l3019_301935


namespace NUMINAMATH_CALUDE_parking_garage_open_spots_l3019_301939

/-- Represents the number of open parking spots on each level of a parking garage -/
structure ParkingGarage where
  first_level : ℕ
  second_level : ℕ
  third_level : ℕ
  fourth_level : ℕ

/-- Theorem stating the number of open parking spots on the first level of the parking garage -/
theorem parking_garage_open_spots (g : ParkingGarage) : g.first_level = 58 :=
  by
  have h1 : g.second_level = g.first_level + 2 := sorry
  have h2 : g.third_level = g.second_level + 5 := sorry
  have h3 : g.fourth_level = 31 := sorry
  have h4 : g.first_level + g.second_level + g.third_level + g.fourth_level = 400 - 186 := sorry
  sorry

#check parking_garage_open_spots

end NUMINAMATH_CALUDE_parking_garage_open_spots_l3019_301939
