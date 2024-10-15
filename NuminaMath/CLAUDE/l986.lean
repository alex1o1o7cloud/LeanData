import Mathlib

namespace NUMINAMATH_CALUDE_truthful_dwarfs_count_l986_98629

theorem truthful_dwarfs_count :
  ∀ (total_dwarfs : ℕ) 
    (vanilla_hands chocolate_hands fruit_hands : ℕ),
  total_dwarfs = 10 →
  vanilla_hands = total_dwarfs →
  chocolate_hands = total_dwarfs / 2 →
  fruit_hands = 1 →
  ∃ (truthful_dwarfs : ℕ),
    truthful_dwarfs = 4 ∧
    truthful_dwarfs + (total_dwarfs - truthful_dwarfs) = total_dwarfs ∧
    vanilla_hands + chocolate_hands + fruit_hands = 
      total_dwarfs + (total_dwarfs - truthful_dwarfs) :=
by sorry

end NUMINAMATH_CALUDE_truthful_dwarfs_count_l986_98629


namespace NUMINAMATH_CALUDE_parabola_vertex_l986_98676

-- Define the parabola equation
def parabola_equation (x y : ℝ) : Prop :=
  y^2 + 4*y + 3*x + 8 = 0

-- Define the vertex of a parabola
def is_vertex (x y : ℝ) (eq : ℝ → ℝ → Prop) : Prop :=
  ∀ x' y', eq x' y' → x' ≥ x

-- Theorem stating that (-4/3, -2) is the vertex of the parabola
theorem parabola_vertex :
  is_vertex (-4/3) (-2) parabola_equation :=
sorry

end NUMINAMATH_CALUDE_parabola_vertex_l986_98676


namespace NUMINAMATH_CALUDE_polynomial_factorization_l986_98635

theorem polynomial_factorization (x : ℤ) :
  5 * (x + 4) * (x + 7) * (x + 11) * (x + 13) - 4 * x^2 =
  (5 * x^2 + 94 * x + 385) * (x + 3) * (x + 14) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l986_98635


namespace NUMINAMATH_CALUDE_blueberry_muffin_percentage_is_fifty_percent_l986_98673

/-- Calculates the percentage of blueberry muffins given the number of blueberry cartons,
    blueberries per carton, blueberries per muffin, and number of cinnamon muffins. -/
def blueberry_muffin_percentage (
  cartons : ℕ
  ) (blueberries_per_carton : ℕ
  ) (blueberries_per_muffin : ℕ
  ) (cinnamon_muffins : ℕ
  ) : ℚ :=
  let total_blueberries := cartons * blueberries_per_carton
  let blueberry_muffins := total_blueberries / blueberries_per_muffin
  let total_muffins := blueberry_muffins + cinnamon_muffins
  (blueberry_muffins : ℚ) / (total_muffins : ℚ) * 100

/-- Proves that given 3 cartons of 200 blueberries, making muffins with 10 blueberries each,
    and 60 additional cinnamon muffins, the percentage of blueberry muffins is 50% of the total muffins. -/
theorem blueberry_muffin_percentage_is_fifty_percent :
  blueberry_muffin_percentage 3 200 10 60 = 50 := by
  sorry

end NUMINAMATH_CALUDE_blueberry_muffin_percentage_is_fifty_percent_l986_98673


namespace NUMINAMATH_CALUDE_irrational_sqrt_three_rational_others_l986_98625

-- Define a rational number
def is_rational (x : ℝ) : Prop := ∃ (a b : ℤ), b ≠ 0 ∧ x = a / b

-- Define an irrational number
def is_irrational (x : ℝ) : Prop := ¬ is_rational x

theorem irrational_sqrt_three_rational_others : 
  is_irrational (Real.sqrt 3) ∧ 
  is_rational (-2) ∧ 
  is_rational (1/2) ∧ 
  is_rational 2 :=
sorry

end NUMINAMATH_CALUDE_irrational_sqrt_three_rational_others_l986_98625


namespace NUMINAMATH_CALUDE_quadratic_root_difference_l986_98606

theorem quadratic_root_difference : ∀ (x₁ x₂ : ℝ),
  (7 + 4 * Real.sqrt 3) * x₁^2 + (2 + Real.sqrt 3) * x₁ - 2 = 0 →
  (7 + 4 * Real.sqrt 3) * x₂^2 + (2 + Real.sqrt 3) * x₂ - 2 = 0 →
  x₁ ≠ x₂ →
  max x₁ x₂ - min x₁ x₂ = 6 - 3 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_root_difference_l986_98606


namespace NUMINAMATH_CALUDE_complex_number_equality_l986_98657

theorem complex_number_equality (Z : ℂ) (h : Z * (1 - Complex.I) = 3 - Complex.I) : 
  Z = 2 + Complex.I := by
sorry

end NUMINAMATH_CALUDE_complex_number_equality_l986_98657


namespace NUMINAMATH_CALUDE_area_inside_circle_outside_rectangle_l986_98690

/-- The area inside a circle but outside a rectangle with shared center --/
theorem area_inside_circle_outside_rectangle (π : Real) :
  let circle_radius : Real := 1 / 3
  let rectangle_length : Real := 3
  let rectangle_width : Real := 1.5
  let circle_area : Real := π * circle_radius ^ 2
  let rectangle_area : Real := rectangle_length * rectangle_width
  let rectangle_diagonal : Real := (rectangle_length ^ 2 + rectangle_width ^ 2).sqrt
  circle_radius < rectangle_diagonal / 2 →
  circle_area = π / 9 := by
  sorry

end NUMINAMATH_CALUDE_area_inside_circle_outside_rectangle_l986_98690


namespace NUMINAMATH_CALUDE_frog_mouse_jump_difference_l986_98667

/-- Represents the jumping contest between a grasshopper, a frog, and a mouse -/
def jumping_contest (grasshopper_jump mouse_jump frog_jump : ℕ) : Prop :=
  grasshopper_jump = 14 ∧
  frog_jump = grasshopper_jump + 37 ∧
  mouse_jump = grasshopper_jump + 21

/-- Theorem stating the difference between the frog's and mouse's jump distances -/
theorem frog_mouse_jump_difference 
  (grasshopper_jump mouse_jump frog_jump : ℕ)
  (h : jumping_contest grasshopper_jump mouse_jump frog_jump) :
  frog_jump - mouse_jump = 16 := by
  sorry

#check frog_mouse_jump_difference

end NUMINAMATH_CALUDE_frog_mouse_jump_difference_l986_98667


namespace NUMINAMATH_CALUDE_solution_analysis_l986_98645

def system_of_equations (a x y z : ℝ) : Prop :=
  a^3 * x + a * y + z = a^2 ∧
  x + y + z = 1 ∧
  8 * x + 2 * y + z = 4

theorem solution_analysis :
  (∀ x y z : ℝ, system_of_equations 2 x y z ↔ x = 1/5 ∧ y = 8/5 ∧ z = -2/5) ∧
  (∃ x₁ y₁ z₁ x₂ y₂ z₂ : ℝ, x₁ ≠ x₂ ∧ system_of_equations 1 x₁ y₁ z₁ ∧ system_of_equations 1 x₂ y₂ z₂) ∧
  (¬∃ x y z : ℝ, system_of_equations (-3) x y z) :=
sorry

end NUMINAMATH_CALUDE_solution_analysis_l986_98645


namespace NUMINAMATH_CALUDE_solution_count_decrease_l986_98632

/-- The system of equations has fewer than four solutions if and only if a = ±1 or a = ±√2 -/
theorem solution_count_decrease (a : ℝ) : 
  (∃ x y : ℝ, x^2 - y^2 = 0 ∧ (x - a)^2 + y^2 = 1) ∧ 
  (∀ x₁ y₁ x₂ y₂ x₃ y₃ x₄ y₄ : ℝ, 
    (x₁^2 - y₁^2 = 0 ∧ (x₁ - a)^2 + y₁^2 = 1) → 
    (x₂^2 - y₂^2 = 0 ∧ (x₂ - a)^2 + y₂^2 = 1) → 
    (x₃^2 - y₃^2 = 0 ∧ (x₃ - a)^2 + y₃^2 = 1) → 
    (x₄^2 - y₄^2 = 0 ∧ (x₄ - a)^2 + y₄^2 = 1) → 
    (x₁ = x₂ ∧ y₁ = y₂) ∨ (x₁ = x₃ ∧ y₁ = y₃) ∨ (x₁ = x₄ ∧ y₁ = y₄) ∨ 
    (x₂ = x₃ ∧ y₂ = y₃) ∨ (x₂ = x₄ ∧ y₂ = y₄) ∨ (x₃ = x₄ ∧ y₃ = y₄)) ↔ 
  a = 1 ∨ a = -1 ∨ a = Real.sqrt 2 ∨ a = -Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_solution_count_decrease_l986_98632


namespace NUMINAMATH_CALUDE_sin_cos_tan_product_l986_98698

theorem sin_cos_tan_product : 
  Real.sin (4/3 * Real.pi) * Real.cos (5/6 * Real.pi) * Real.tan (-4/3 * Real.pi) = -(3 * Real.sqrt 3) / 4 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_tan_product_l986_98698


namespace NUMINAMATH_CALUDE_franks_candy_bags_l986_98651

theorem franks_candy_bags (pieces_per_bag : ℕ) (total_pieces : ℕ) (h1 : pieces_per_bag = 11) (h2 : total_pieces = 22) :
  total_pieces / pieces_per_bag = 2 :=
by sorry

end NUMINAMATH_CALUDE_franks_candy_bags_l986_98651


namespace NUMINAMATH_CALUDE_trapezoid_area_l986_98670

/-- Given two equilateral triangles and four congruent trapezoids between them,
    this theorem proves that the area of one trapezoid is 8 square units. -/
theorem trapezoid_area
  (outer_triangle_area : ℝ)
  (inner_triangle_area : ℝ)
  (num_trapezoids : ℕ)
  (h_outer_area : outer_triangle_area = 36)
  (h_inner_area : inner_triangle_area = 4)
  (h_num_trapezoids : num_trapezoids = 4) :
  (outer_triangle_area - inner_triangle_area) / num_trapezoids = 8 := by
  sorry

end NUMINAMATH_CALUDE_trapezoid_area_l986_98670


namespace NUMINAMATH_CALUDE_kim_hard_round_correct_l986_98693

/-- A math contest with three rounds of questions --/
structure MathContest where
  easy_points : ℕ
  average_points : ℕ
  hard_points : ℕ
  easy_correct : ℕ
  average_correct : ℕ
  total_points : ℕ

/-- Kim's performance in the math contest --/
def kim_contest : MathContest :=
  { easy_points := 2
  , average_points := 3
  , hard_points := 5
  , easy_correct := 6
  , average_correct := 2
  , total_points := 38 }

/-- The number of correct answers in the hard round --/
def hard_round_correct (contest : MathContest) : ℕ :=
  (contest.total_points - (contest.easy_points * contest.easy_correct + contest.average_points * contest.average_correct)) / contest.hard_points

theorem kim_hard_round_correct :
  hard_round_correct kim_contest = 4 := by
  sorry

end NUMINAMATH_CALUDE_kim_hard_round_correct_l986_98693


namespace NUMINAMATH_CALUDE_hiking_rate_proof_l986_98669

-- Define the hiking scenario
def hiking_scenario (rate : ℝ) : Prop :=
  let initial_distance : ℝ := 2.5
  let total_distance : ℝ := 3.5
  let total_time : ℝ := 45
  let return_distance : ℝ := total_distance - initial_distance
  
  -- The time to hike the additional distance east
  (return_distance / rate) +
  -- The time to hike back the additional distance
  (return_distance / rate) +
  -- The time to hike back the initial distance
  (initial_distance / rate) = total_time

-- Theorem statement
theorem hiking_rate_proof :
  ∃ (rate : ℝ), hiking_scenario rate ∧ rate = 1/10 :=
sorry

end NUMINAMATH_CALUDE_hiking_rate_proof_l986_98669


namespace NUMINAMATH_CALUDE_right_triangle_and_modular_inverse_l986_98638

theorem right_triangle_and_modular_inverse : 
  (80^2 + 150^2 = 170^2) ∧ 
  (320 * 642 % 2879 = 1) := by sorry

end NUMINAMATH_CALUDE_right_triangle_and_modular_inverse_l986_98638


namespace NUMINAMATH_CALUDE_infinite_solutions_condition_l986_98660

theorem infinite_solutions_condition (b : ℝ) :
  (∀ x : ℝ, 5 * (3 * x - b) = 3 * (5 * x + 15)) ↔ b = -9 := by sorry

end NUMINAMATH_CALUDE_infinite_solutions_condition_l986_98660


namespace NUMINAMATH_CALUDE_seats_needed_for_zoo_trip_l986_98604

theorem seats_needed_for_zoo_trip (total_children : ℕ) (children_per_seat : ℕ) (seats_needed : ℕ) : 
  total_children = 58 → children_per_seat = 2 → seats_needed = total_children / children_per_seat → seats_needed = 29 := by
  sorry

end NUMINAMATH_CALUDE_seats_needed_for_zoo_trip_l986_98604


namespace NUMINAMATH_CALUDE_price_increase_condition_l986_98649

/-- Represents the fruit purchase and sale scenario -/
structure FruitSale where
  quantity : ℝ  -- Initial quantity in kg
  price : ℝ     -- Purchase price per kg
  loss : ℝ      -- Fraction of loss during transportation
  profit : ℝ    -- Desired minimum profit fraction
  increase : ℝ  -- Fraction of price increase

/-- Theorem stating the condition for the required price increase -/
theorem price_increase_condition (sale : FruitSale) 
  (h1 : sale.quantity = 200)
  (h2 : sale.price = 5)
  (h3 : sale.loss = 0.05)
  (h4 : sale.profit = 0.2) :
  (1 - sale.loss) * (1 + sale.increase) ≥ (1 + sale.profit) :=
sorry

end NUMINAMATH_CALUDE_price_increase_condition_l986_98649


namespace NUMINAMATH_CALUDE_candy_cost_per_pack_l986_98679

theorem candy_cost_per_pack (number_of_packs : ℕ) (amount_paid : ℕ) (change_received : ℕ) :
  number_of_packs = 3 →
  amount_paid = 20 →
  change_received = 11 →
  (amount_paid - change_received) / number_of_packs = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_candy_cost_per_pack_l986_98679


namespace NUMINAMATH_CALUDE_houses_with_neither_feature_l986_98603

theorem houses_with_neither_feature (total : ℕ) (garage : ℕ) (pool : ℕ) (both : ℕ) 
  (h1 : total = 85) 
  (h2 : garage = 50) 
  (h3 : pool = 40) 
  (h4 : both = 35) : 
  total - (garage + pool - both) = 30 := by
  sorry

end NUMINAMATH_CALUDE_houses_with_neither_feature_l986_98603


namespace NUMINAMATH_CALUDE_product_equality_l986_98652

theorem product_equality : 469111 * 9999 = 4690428889 := by
  sorry

end NUMINAMATH_CALUDE_product_equality_l986_98652


namespace NUMINAMATH_CALUDE_product_decrease_theorem_l986_98613

theorem product_decrease_theorem :
  ∃ (a b c d e : ℕ), 
    (a - 3) * (b - 3) * (c - 3) * (d - 3) * (e - 3) = 15 * (a * b * c * d * e) := by
  sorry

end NUMINAMATH_CALUDE_product_decrease_theorem_l986_98613


namespace NUMINAMATH_CALUDE_min_rings_to_connect_five_links_l986_98624

/-- Represents a chain link with a specific number of rings -/
structure ChainLink where
  rings : ℕ

/-- Represents a collection of chain links -/
structure ChainCollection where
  links : List ChainLink

/-- The minimum number of rings needed to connect a chain collection into a single chain -/
def minRingsToConnect (c : ChainCollection) : ℕ :=
  sorry

/-- The problem statement -/
theorem min_rings_to_connect_five_links :
  let links := List.replicate 5 (ChainLink.mk 3)
  let chain := ChainCollection.mk links
  minRingsToConnect chain = 3 := by
  sorry

end NUMINAMATH_CALUDE_min_rings_to_connect_five_links_l986_98624


namespace NUMINAMATH_CALUDE_complex_number_range_l986_98699

/-- The range of y/x for a complex number (x-2) + yi with modulus 1 -/
theorem complex_number_range (x y : ℝ) : 
  (x - 2)^2 + y^2 = 1 → 
  y ≠ 0 → 
  ∃ k : ℝ, y = k * x ∧ 
    (-Real.sqrt 3 / 3 ≤ k ∧ k < 0) ∨ (0 < k ∧ k ≤ Real.sqrt 3 / 3) :=
sorry

end NUMINAMATH_CALUDE_complex_number_range_l986_98699


namespace NUMINAMATH_CALUDE_percentage_85_89_is_40_3_l986_98633

/-- Represents the frequency distribution of test scores in a class -/
structure ScoreDistribution where
  range_90_100 : Nat
  range_85_89 : Nat
  range_75_84 : Nat
  range_65_74 : Nat
  range_below_65 : Nat

/-- Calculates the percentage of students in a specific score range -/
def percentageInRange (dist : ScoreDistribution) (rangeCount : Nat) : Rat :=
  let totalStudents := dist.range_90_100 + dist.range_85_89 + dist.range_75_84 + 
                       dist.range_65_74 + dist.range_below_65
  (rangeCount : Rat) / (totalStudents : Rat) * 100

/-- The main theorem stating that the percentage of students in the 85%-89% range is 40/3% -/
theorem percentage_85_89_is_40_3 (dist : ScoreDistribution) 
    (h : dist = ScoreDistribution.mk 6 4 7 10 3) : 
    percentageInRange dist dist.range_85_89 = 40 / 3 := by
  sorry

end NUMINAMATH_CALUDE_percentage_85_89_is_40_3_l986_98633


namespace NUMINAMATH_CALUDE_simplify_fraction_division_l986_98692

theorem simplify_fraction_division (x : ℝ) (h1 : x ≠ 0) (h2 : x ≠ 1) :
  ((1 - x) / x) / ((1 - x) / x^2) = x := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_division_l986_98692


namespace NUMINAMATH_CALUDE_line_of_symmetry_between_circles_l986_98637

-- Define the two circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 = 9
def circle2 (x y : ℝ) : Prop := x^2 + y^2 - 4*x + 4*y - 1 = 0

-- Define the line of symmetry
def line_l (x y : ℝ) : Prop := x - y - 2 = 0

-- Define symmetry between points with respect to a line
def symmetric_points (x1 y1 x2 y2 a b c : ℝ) : Prop :=
  a * (x1 + x2) + b * (y1 + y2) + 2 * c = 0

-- Theorem statement
theorem line_of_symmetry_between_circles :
  ∀ (x1 y1 x2 y2 : ℝ),
    circle1 x1 y1 → circle2 x2 y2 →
    symmetric_points x1 y1 x2 y2 1 (-1) (-2) →
    line_l ((x1 + x2) / 2) ((y1 + y2) / 2) :=
sorry

end NUMINAMATH_CALUDE_line_of_symmetry_between_circles_l986_98637


namespace NUMINAMATH_CALUDE_tan_negative_240_degrees_l986_98681

theorem tan_negative_240_degrees : Real.tan (-(240 * π / 180)) = -Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_negative_240_degrees_l986_98681


namespace NUMINAMATH_CALUDE_max_moves_in_grid_fourteen_fits_grid_fifteen_exceeds_grid_max_moves_is_fourteen_l986_98644

theorem max_moves_in_grid (n : ℕ) : n > 0 → n * (n + 1) ≤ 200 → n ≤ 14 := by
  sorry

theorem fourteen_fits_grid : 14 * (14 + 1) ≤ 200 := by
  sorry

theorem fifteen_exceeds_grid : 15 * (15 + 1) > 200 := by
  sorry

theorem max_moves_is_fourteen : 
  ∃ (n : ℕ), n > 0 ∧ n * (n + 1) ≤ 200 ∧ ∀ (m : ℕ), m > n → m * (m + 1) > 200 := by
  sorry

end NUMINAMATH_CALUDE_max_moves_in_grid_fourteen_fits_grid_fifteen_exceeds_grid_max_moves_is_fourteen_l986_98644


namespace NUMINAMATH_CALUDE_parabola_intersection_difference_l986_98640

/-- The difference between the x-coordinates of the intersection points of two parabolas -/
theorem parabola_intersection_difference :
  let f (x : ℝ) := 3 * x^2 - 6 * x + 3
  let g (x : ℝ) := -2 * x^2 - 4 * x + 5
  let a := (1 - Real.sqrt 11) / 5
  let c := (1 + Real.sqrt 11) / 5
  f a = g a ∧ f c = g c ∧ c ≥ a →
  c - a = 2 * Real.sqrt 11 / 5 := by
sorry

end NUMINAMATH_CALUDE_parabola_intersection_difference_l986_98640


namespace NUMINAMATH_CALUDE_arithmetic_expression_equality_l986_98663

theorem arithmetic_expression_equality :
  4^2 * 10 + 5 * 12 + 12 * 4 + 24 / 3 * 9 = 340 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_expression_equality_l986_98663


namespace NUMINAMATH_CALUDE_correct_contributions_l986_98697

/-- Represents the business contribution problem -/
structure BusinessContribution where
  total : ℝ
  a_months : ℝ
  b_months : ℝ
  a_received : ℝ
  b_received : ℝ

/-- Theorem stating the correct contributions of A and B -/
theorem correct_contributions (bc : BusinessContribution)
  (h_total : bc.total = 3400)
  (h_a_months : bc.a_months = 12)
  (h_b_months : bc.b_months = 16)
  (h_a_received : bc.a_received = 2070)
  (h_b_received : bc.b_received = 1920) :
  ∃ (a_contribution b_contribution : ℝ),
    a_contribution = 1800 ∧
    b_contribution = 1600 ∧
    a_contribution + b_contribution = bc.total ∧
    (bc.a_received - a_contribution) / (bc.b_received - (bc.total - a_contribution)) =
      (bc.a_months * a_contribution) / (bc.b_months * (bc.total - a_contribution)) :=
by sorry

end NUMINAMATH_CALUDE_correct_contributions_l986_98697


namespace NUMINAMATH_CALUDE_gas_pressure_change_l986_98628

/-- Represents the pressure-volume relationship for a gas at constant temperature -/
def pressure_volume_relation (p1 p2 v1 v2 : ℝ) : Prop :=
  p1 * v1 = p2 * v2

/-- Theorem: Given inverse proportionality of pressure and volume,
    if a gas initially at 8 kPa in a 3.5-liter container is transferred to a 7-liter container,
    its new pressure will be 4 kPa -/
theorem gas_pressure_change (p2 : ℝ) :
  pressure_volume_relation 8 p2 3.5 7 → p2 = 4 := by
  sorry

end NUMINAMATH_CALUDE_gas_pressure_change_l986_98628


namespace NUMINAMATH_CALUDE_barkley_buried_bones_l986_98616

/-- Proves that Barkley has buried 42 bones after 5 months -/
theorem barkley_buried_bones 
  (bones_per_month : ℕ) 
  (months_passed : ℕ) 
  (available_bones : ℕ) 
  (h1 : bones_per_month = 10)
  (h2 : months_passed = 5)
  (h3 : available_bones = 8) :
  bones_per_month * months_passed - available_bones = 42 := by
  sorry

end NUMINAMATH_CALUDE_barkley_buried_bones_l986_98616


namespace NUMINAMATH_CALUDE_puzzle_completion_l986_98683

/-- Puzzle completion problem -/
theorem puzzle_completion 
  (total_pieces : ℕ) 
  (num_children : ℕ) 
  (time_limit : ℕ) 
  (reyn_rate : ℚ) :
  total_pieces = 500 →
  num_children = 4 →
  time_limit = 120 →
  reyn_rate = 25 / 30 →
  (reyn_rate * time_limit + 
   2 * reyn_rate * time_limit + 
   3 * reyn_rate * time_limit + 
   4 * reyn_rate * time_limit) ≥ total_pieces := by
  sorry


end NUMINAMATH_CALUDE_puzzle_completion_l986_98683


namespace NUMINAMATH_CALUDE_N_equals_negative_fifteen_l986_98666

/-- A grid with arithmetic sequences in rows and columns -/
structure ArithmeticGrid where
  row_start : ℤ
  col1_second : ℤ
  col1_third : ℤ
  col2_last : ℤ

/-- The value N we're trying to determine -/
def N (grid : ArithmeticGrid) : ℤ :=
  grid.col2_last + (grid.col1_third - grid.col1_second)

/-- Theorem stating that N equals -15 for the given grid -/
theorem N_equals_negative_fifteen (grid : ArithmeticGrid) 
  (h1 : grid.row_start = 25)
  (h2 : grid.col1_second = 10)
  (h3 : grid.col1_third = 18)
  (h4 : grid.col2_last = -23) :
  N grid = -15 := by
  sorry


end NUMINAMATH_CALUDE_N_equals_negative_fifteen_l986_98666


namespace NUMINAMATH_CALUDE_largest_odd_digit_multiple_of_11_l986_98685

def is_odd_digit (d : ℕ) : Prop := d % 2 = 1 ∧ d ≤ 9

def has_only_odd_digits (n : ℕ) : Prop :=
  ∀ d, (n / 10^d % 10 ≠ 0 → is_odd_digit (n / 10^d % 10))

def alternating_sum (n : ℕ) : ℤ :=
  let digits := List.reverse (List.map (λ i => n / 10^i % 10) (List.range 4))
  List.foldl (λ sum (i, d) => sum + ((-1)^i : ℤ) * d) 0 (List.enumFrom 0 digits)

theorem largest_odd_digit_multiple_of_11 :
  9393 < 10000 ∧
  has_only_odd_digits 9393 ∧
  alternating_sum 9393 % 11 = 0 ∧
  ∀ n : ℕ, n < 10000 → has_only_odd_digits n → alternating_sum n % 11 = 0 → n ≤ 9393 :=
sorry

end NUMINAMATH_CALUDE_largest_odd_digit_multiple_of_11_l986_98685


namespace NUMINAMATH_CALUDE_grid_31_counts_l986_98658

/-- Represents a grid with n horizontal and vertical lines -/
structure Grid (n : ℕ) where
  horizontal_lines : ℕ
  vertical_lines : ℕ
  h_lines : horizontal_lines = n
  v_lines : vertical_lines = n

/-- Counts the number of rectangles in a grid -/
def count_rectangles (g : Grid n) : ℕ :=
  (n.choose 2) * (n.choose 2)

/-- Counts the number of squares in a grid with 1:2 distance ratio -/
def count_squares (g : Grid n) : ℕ :=
  let S (k : ℕ) := k * (k + 1) * (2 * k + 1) / 6
  S n - 2 * S (n / 2)

/-- The main theorem about the 31x31 grid -/
theorem grid_31_counts :
  ∃ (g : Grid 31),
    count_rectangles g = 216225 ∧
    count_squares g = 6975 :=
by sorry

end NUMINAMATH_CALUDE_grid_31_counts_l986_98658


namespace NUMINAMATH_CALUDE_g_sum_symmetric_l986_98636

-- Define the function g
def g (p q r : ℝ) (x : ℝ) : ℝ := p * x^8 + q * x^6 - r * x^4 + 5

-- State the theorem
theorem g_sum_symmetric (p q r : ℝ) :
  g p q r 12 = 3 → g p q r 12 + g p q r (-12) = 6 := by
  sorry

end NUMINAMATH_CALUDE_g_sum_symmetric_l986_98636


namespace NUMINAMATH_CALUDE_minimum_value_theorem_l986_98665

/-- A geometric sequence with positive terms -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q > 0 ∧ ∀ n : ℕ, a (n + 1) = q * a n

theorem minimum_value_theorem (a : ℕ → ℝ) (m n : ℕ) :
  geometric_sequence a →
  (a 7 = a 6 + 2 * a 5) →
  (∃ m n : ℕ, Real.sqrt (a m * a n) = 4 * a 1) →
  (∀ k l : ℕ, 1 / k + 9 / l ≥ 11 / 4) :=
by sorry

end NUMINAMATH_CALUDE_minimum_value_theorem_l986_98665


namespace NUMINAMATH_CALUDE_constant_width_from_circle_sum_l986_98656

/-- A curve in 2D space -/
structure Curve where
  -- Add necessary fields here
  convex : Bool

/-- Rotation of a curve by 180 degrees -/
def rotate180 (K : Curve) : Curve :=
  sorry

/-- Sum of two curves -/
def curveSum (K1 K2 : Curve) : Curve :=
  sorry

/-- Check if a curve is a circle -/
def isCircle (K : Curve) : Prop :=
  sorry

/-- Check if a curve has constant width -/
def hasConstantWidth (K : Curve) : Prop :=
  sorry

/-- Main theorem -/
theorem constant_width_from_circle_sum (K : Curve) (h : K.convex) :
  let K' := rotate180 K
  let K_star := curveSum K K'
  isCircle K_star → hasConstantWidth K :=
by
  sorry

end NUMINAMATH_CALUDE_constant_width_from_circle_sum_l986_98656


namespace NUMINAMATH_CALUDE_maximum_marks_calculation_l986_98682

/-- Given a student who needs to obtain 40% to pass, got 150 marks, and failed by 50 marks, 
    the maximum possible marks are 500. -/
theorem maximum_marks_calculation (passing_threshold : ℝ) (marks_obtained : ℝ) (marks_short : ℝ) :
  passing_threshold = 0.4 →
  marks_obtained = 150 →
  marks_short = 50 →
  ∃ (max_marks : ℝ), max_marks = 500 ∧ passing_threshold * max_marks = marks_obtained + marks_short :=
by sorry

end NUMINAMATH_CALUDE_maximum_marks_calculation_l986_98682


namespace NUMINAMATH_CALUDE_walkers_commute_l986_98626

/-- Ms. Walker's commute problem -/
theorem walkers_commute
  (speed_to_work : ℝ)
  (speed_from_work : ℝ)
  (total_time : ℝ)
  (h1 : speed_to_work = 60)
  (h2 : speed_from_work = 40)
  (h3 : total_time = 1) :
  ∃ (distance : ℝ), 
    distance / speed_to_work + distance / speed_from_work = total_time ∧ 
    distance = 24 := by
  sorry

end NUMINAMATH_CALUDE_walkers_commute_l986_98626


namespace NUMINAMATH_CALUDE_height_side_relation_l986_98687

/-- Triangle with sides and heights -/
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  H_A : ℝ
  H_B : ℝ
  H_C : ℝ

/-- Theorem: In a triangle, if one height is greater than another, the side opposite to the greater height is shorter than the side opposite to the smaller height -/
theorem height_side_relation (t : Triangle) :
  t.H_A > t.H_B → t.B < t.A :=
by sorry

end NUMINAMATH_CALUDE_height_side_relation_l986_98687


namespace NUMINAMATH_CALUDE_right_triangle_and_perimeter_range_l986_98675

/-- Triangle ABC with side lengths a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The theorem about the right triangle and its perimeter range -/
theorem right_triangle_and_perimeter_range (t : Triangle) (h : t.a * (Real.cos t.B + Real.cos t.C) = t.b + t.c) :
  t.A = π / 2 ∧ 
  (∀ (r : ℝ), r = 1 → 4 < t.a + t.b + t.c ∧ t.a + t.b + t.c ≤ 2 + 2 * Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_and_perimeter_range_l986_98675


namespace NUMINAMATH_CALUDE_equation_solution_l986_98619

theorem equation_solution (x : ℝ) :
  x ≠ -2 → x ≠ 2 →
  (1 / (x + 2) + (x + 6) / (x^2 - 4) = 1) ↔ x = 4 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l986_98619


namespace NUMINAMATH_CALUDE_total_earnings_calculation_l986_98653

theorem total_earnings_calculation 
  (x y : ℝ) 
  (h1 : 4 * x * (5 * y / 100) = 3 * x * (6 * y / 100) + 350) 
  (h2 : x * y = 17500) : 
  (3 * x * (6 * y / 100) + 4 * x * (5 * y / 100) + 5 * x * (4 * y / 100)) = 10150 := by
  sorry

end NUMINAMATH_CALUDE_total_earnings_calculation_l986_98653


namespace NUMINAMATH_CALUDE_parallel_vectors_x_value_l986_98605

/-- Two vectors are parallel if one is a scalar multiple of the other -/
def parallel (v w : ℝ × ℝ) : Prop :=
  ∃ (k : ℝ), v.1 = k * w.1 ∧ v.2 = k * w.2

theorem parallel_vectors_x_value :
  let m : ℝ × ℝ := (-2, 4)
  let n : ℝ × ℝ := (x, -1)
  parallel m n → x = 1/2 :=
by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_x_value_l986_98605


namespace NUMINAMATH_CALUDE_total_votes_cast_l986_98678

theorem total_votes_cast (total_votes : ℕ) (votes_for : ℕ) (votes_against : ℕ) : 
  votes_for = votes_against + 70 →
  votes_against = (40 : ℕ) * total_votes / 100 →
  total_votes = votes_for + votes_against →
  total_votes = 350 := by
sorry

end NUMINAMATH_CALUDE_total_votes_cast_l986_98678


namespace NUMINAMATH_CALUDE_triangle_angle_calculation_l986_98688

theorem triangle_angle_calculation (D E F : ℝ) : 
  D = 90 →
  E = 2 * F + 15 →
  D + E + F = 180 →
  F = 25 := by
sorry

end NUMINAMATH_CALUDE_triangle_angle_calculation_l986_98688


namespace NUMINAMATH_CALUDE_butterfat_solution_l986_98696

def butterfat_problem (x : ℝ) : Prop :=
  let initial_volume : ℝ := 8
  let added_volume : ℝ := 20
  let initial_butterfat : ℝ := x / 100
  let added_butterfat : ℝ := 10 / 100
  let final_butterfat : ℝ := 20 / 100
  let total_volume : ℝ := initial_volume + added_volume
  (initial_volume * initial_butterfat + added_volume * added_butterfat) / total_volume = final_butterfat

theorem butterfat_solution : butterfat_problem 45 := by
  sorry

end NUMINAMATH_CALUDE_butterfat_solution_l986_98696


namespace NUMINAMATH_CALUDE_base_ten_is_only_solution_l986_98612

/-- Represents the number in base b as a function of n -/
def number (b n : ℕ) : ℚ :=
  (b^(3*n) - b^(2*n+1) + 7 * b^(2*n) + b^(n+1) - 7 * b^n - 1) / (3 * (b - 1))

/-- Predicate to check if a rational number is a perfect cube -/
def is_perfect_cube (q : ℚ) : Prop :=
  ∃ m : ℤ, q = (m : ℚ)^3

theorem base_ten_is_only_solution :
  ∀ b : ℕ, b ≥ 9 →
  (∀ n : ℕ, ∃ N : ℕ, ∀ m : ℕ, m ≥ N → is_perfect_cube (number b m)) →
  b = 10 :=
sorry

end NUMINAMATH_CALUDE_base_ten_is_only_solution_l986_98612


namespace NUMINAMATH_CALUDE_farm_animals_l986_98674

theorem farm_animals (total_legs : ℕ) (total_animals : ℕ) (duck_legs : ℕ) (cow_legs : ℕ) :
  total_legs = 42 →
  total_animals = 15 →
  duck_legs = 2 →
  cow_legs = 4 →
  ∃ (num_ducks : ℕ) (num_cows : ℕ),
    num_ducks + num_cows = total_animals ∧
    num_ducks * duck_legs + num_cows * cow_legs = total_legs ∧
    num_cows = 6 :=
by sorry

end NUMINAMATH_CALUDE_farm_animals_l986_98674


namespace NUMINAMATH_CALUDE_cannot_finish_third_l986_98620

-- Define the set of runners
inductive Runner : Type
| A | B | C | D | E | F

-- Define the finish order relation
def finishes_before (x y : Runner) : Prop := sorry

-- Define the race conditions
axiom race_condition1 : finishes_before Runner.A Runner.B ∧ finishes_before Runner.A Runner.D
axiom race_condition2 : finishes_before Runner.B Runner.C ∧ finishes_before Runner.B Runner.F
axiom race_condition3 : finishes_before Runner.C Runner.D
axiom race_condition4 : finishes_before Runner.E Runner.F ∧ finishes_before Runner.A Runner.E

-- Define a function to represent the finishing position of a runner
def finishing_position (r : Runner) : ℕ := sorry

-- Define what it means to finish in third place
def finishes_third (r : Runner) : Prop := finishing_position r = 3

-- Theorem to prove
theorem cannot_finish_third : 
  ¬(finishes_third Runner.A) ∧ ¬(finishes_third Runner.F) := sorry

end NUMINAMATH_CALUDE_cannot_finish_third_l986_98620


namespace NUMINAMATH_CALUDE_double_base_exponent_problem_l986_98639

theorem double_base_exponent_problem (a b x : ℝ) 
  (ha : a > 0) (hb : b > 0) (hx : x > 0) :
  (2*a)^(2*b) = (a^2)^b * x^b → x = 4 := by
  sorry

end NUMINAMATH_CALUDE_double_base_exponent_problem_l986_98639


namespace NUMINAMATH_CALUDE_circular_table_dice_probability_l986_98614

/-- The number of people sitting around the circular table -/
def num_people : ℕ := 5

/-- The number of sides on the die -/
def die_sides : ℕ := 8

/-- The probability that no two adjacent people roll the same number -/
def no_adjacent_same_prob : ℚ := 441 / 8192

theorem circular_table_dice_probability :
  let n := num_people
  let s := die_sides
  (n : ℚ) > 0 ∧ (s : ℚ) > 0 →
  no_adjacent_same_prob = 441 / 8192 := by
  sorry

end NUMINAMATH_CALUDE_circular_table_dice_probability_l986_98614


namespace NUMINAMATH_CALUDE_francie_allowance_problem_l986_98600

/-- Represents the number of weeks Francie received her increased allowance -/
def weeks_of_increased_allowance : ℕ := 6

/-- Initial savings from the first 8 weeks -/
def initial_savings : ℕ := 5 * 8

/-- Increased allowance per week -/
def increased_allowance : ℕ := 6

/-- Total savings including both initial and increased allowance periods -/
def total_savings (x : ℕ) : ℕ := initial_savings + increased_allowance * x

theorem francie_allowance_problem :
  total_savings weeks_of_increased_allowance / 2 = 35 + 3 ∧
  weeks_of_increased_allowance * increased_allowance = total_savings weeks_of_increased_allowance - initial_savings :=
by sorry

end NUMINAMATH_CALUDE_francie_allowance_problem_l986_98600


namespace NUMINAMATH_CALUDE_percentage_of_defective_meters_l986_98622

theorem percentage_of_defective_meters (total_meters examined_meters : ℕ) 
  (h1 : total_meters = 120) (h2 : examined_meters = 12) :
  (examined_meters : ℝ) / total_meters * 100 = 10 := by
  sorry

end NUMINAMATH_CALUDE_percentage_of_defective_meters_l986_98622


namespace NUMINAMATH_CALUDE_circular_fields_area_comparison_l986_98621

theorem circular_fields_area_comparison (r₁ r₂ : ℝ) (h : r₂ = (5/2) * r₁) :
  (π * r₂^2 - π * r₁^2) / (π * r₁^2) = 2.25 := by
  sorry

end NUMINAMATH_CALUDE_circular_fields_area_comparison_l986_98621


namespace NUMINAMATH_CALUDE_min_value_sum_reciprocals_l986_98671

theorem min_value_sum_reciprocals (a b c d e f : ℝ) 
  (pos_a : 0 < a) (pos_b : 0 < b) (pos_c : 0 < c) 
  (pos_d : 0 < d) (pos_e : 0 < e) (pos_f : 0 < f)
  (sum_eq_10 : a + b + c + d + e + f = 10) :
  (4/a + 9/b + 16/c + 25/d + 36/e + 49/f) ≥ 72.9 := by
  sorry

end NUMINAMATH_CALUDE_min_value_sum_reciprocals_l986_98671


namespace NUMINAMATH_CALUDE_nathan_nickels_l986_98627

theorem nathan_nickels (n : ℕ) : 
  20 < n ∧ n < 200 ∧ 
  n % 4 = 2 ∧ 
  n % 5 = 2 ∧ 
  n % 7 = 2 → 
  n = 142 := by sorry

end NUMINAMATH_CALUDE_nathan_nickels_l986_98627


namespace NUMINAMATH_CALUDE_unique_nonnegative_solution_l986_98641

theorem unique_nonnegative_solution :
  ∃! (x : ℝ), x ≥ 0 ∧ x^2 = -4*x := by sorry

end NUMINAMATH_CALUDE_unique_nonnegative_solution_l986_98641


namespace NUMINAMATH_CALUDE_corgi_price_calculation_l986_98610

theorem corgi_price_calculation (x : ℝ) : 
  (2 * (x + 0.3 * x) = 2600) → x = 1000 := by
  sorry

end NUMINAMATH_CALUDE_corgi_price_calculation_l986_98610


namespace NUMINAMATH_CALUDE_triangle_ABC_properties_l986_98609

theorem triangle_ABC_properties (a b c : ℝ) (A B C : ℝ) :
  (0 < A) ∧ (A < π) ∧ (0 < B) ∧ (B < π) ∧ (0 < C) ∧ (C < π) →
  a * Real.cos B + b * Real.cos A = 2 * c * Real.sin C →
  b = 2 * Real.sqrt 3 →
  c = Real.sqrt 19 →
  ((C = π / 6) ∨ (C = 5 * π / 6)) ∧
  (∃ (S : ℝ), (S = (7 * Real.sqrt 3) / 2) ∨ (S = Real.sqrt 3 / 2)) :=
by sorry

end NUMINAMATH_CALUDE_triangle_ABC_properties_l986_98609


namespace NUMINAMATH_CALUDE_book_price_problem_l986_98611

theorem book_price_problem (original_price : ℝ) : 
  original_price * (1 - 0.25) + original_price * (1 - 0.40) = 66 → 
  original_price = 48.89 := by
sorry

end NUMINAMATH_CALUDE_book_price_problem_l986_98611


namespace NUMINAMATH_CALUDE_system_solution_l986_98623

theorem system_solution :
  ∃ (x y : ℚ), 
    (2 * x - 3 * y = 1) ∧
    (5 * x + 4 * y = 6) ∧
    (x + 2 * y = 2) ∧
    (x = 2/3) ∧
    (y = 2/3) := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l986_98623


namespace NUMINAMATH_CALUDE_derivative_of_f_l986_98648

noncomputable def f (x : ℝ) : ℝ := x^2 * Real.cos x

theorem derivative_of_f (x : ℝ) :
  deriv f x = 2 * x * Real.cos x - x^2 * Real.sin x :=
by sorry

end NUMINAMATH_CALUDE_derivative_of_f_l986_98648


namespace NUMINAMATH_CALUDE_remainder_8457_mod_9_l986_98672

/-- The sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- A number is congruent to the sum of its digits modulo 9 -/
axiom sum_of_digits_mod_9 (n : ℕ) : n ≡ sum_of_digits n [MOD 9]

/-- The remainder when 8457 is divided by 9 is 6 -/
theorem remainder_8457_mod_9 : 8457 % 9 = 6 := by sorry

end NUMINAMATH_CALUDE_remainder_8457_mod_9_l986_98672


namespace NUMINAMATH_CALUDE_annes_cats_weight_l986_98617

/-- Given Anne's cats' weights, prove the total weight she carries -/
theorem annes_cats_weight (female_weight : ℝ) (male_weight_ratio : ℝ) : 
  female_weight = 2 → 
  male_weight_ratio = 2 → 
  female_weight + female_weight * male_weight_ratio = 6 := by
  sorry

end NUMINAMATH_CALUDE_annes_cats_weight_l986_98617


namespace NUMINAMATH_CALUDE_sum_of_repeating_decimals_l986_98662

/-- Represents a repeating decimal where the digit repeats infinitely after the decimal point. -/
def repeating_decimal (d : ℕ) := (d : ℚ) / 9

/-- The sum of the repeating decimals 0.333... and 0.222... is equal to 5/9. -/
theorem sum_of_repeating_decimals : 
  repeating_decimal 3 + repeating_decimal 2 = 5 / 9 := by sorry

end NUMINAMATH_CALUDE_sum_of_repeating_decimals_l986_98662


namespace NUMINAMATH_CALUDE_sequence_bound_l986_98691

def is_valid_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n, a n > 0 ∧ (a n)^2 ≤ a n - a (n + 1)

theorem sequence_bound (a : ℕ → ℝ) (h : is_valid_sequence a) :
  ∀ n : ℕ, a n < 1 / n :=
sorry

end NUMINAMATH_CALUDE_sequence_bound_l986_98691


namespace NUMINAMATH_CALUDE_students_speaking_neither_language_l986_98634

theorem students_speaking_neither_language (total : ℕ) (english : ℕ) (telugu : ℕ) (both : ℕ) :
  total = 150 →
  english = 55 →
  telugu = 85 →
  both = 20 →
  total - (english + telugu - both) = 30 :=
by sorry

end NUMINAMATH_CALUDE_students_speaking_neither_language_l986_98634


namespace NUMINAMATH_CALUDE_work_completion_l986_98654

/-- Given that 8 men complete a work in 80 days, prove that 20 men will complete the same work in 32 days. -/
theorem work_completion (work : ℕ) : 
  (8 * 80 = work) → (20 * 32 = work) := by
  sorry

end NUMINAMATH_CALUDE_work_completion_l986_98654


namespace NUMINAMATH_CALUDE_golden_retriever_age_l986_98661

-- Define the weight gain per year
def weight_gain_per_year : ℕ := 11

-- Define the current weight
def current_weight : ℕ := 88

-- Define the age of the golden retriever
def age : ℕ := current_weight / weight_gain_per_year

-- Theorem to prove
theorem golden_retriever_age :
  age = 8 :=
by sorry

end NUMINAMATH_CALUDE_golden_retriever_age_l986_98661


namespace NUMINAMATH_CALUDE_naoh_equals_nano3_l986_98664

/-- Represents the number of moles of a chemical substance -/
def Moles : Type := ℝ

/-- Represents the chemical reaction between NH4NO3 and NaOH to produce NaNO3 -/
structure Reaction where
  nh4no3_initial : Moles
  naoh_combined : Moles
  nano3_formed : Moles

/-- The reaction has a 1:1 molar ratio between NH4NO3 and NaOH to produce NaNO3 -/
axiom molar_ratio (r : Reaction) : r.nh4no3_initial = r.naoh_combined

/-- The number of moles of NH4NO3 initially present equals the number of moles of NaNO3 formed -/
axiom conservation (r : Reaction) : r.nh4no3_initial = r.nano3_formed

/-- The number of moles of NaOH combined equals the number of moles of NaNO3 formed -/
theorem naoh_equals_nano3 (r : Reaction) : r.naoh_combined = r.nano3_formed := by
  sorry

end NUMINAMATH_CALUDE_naoh_equals_nano3_l986_98664


namespace NUMINAMATH_CALUDE_factor_of_x4_plus_12_l986_98608

theorem factor_of_x4_plus_12 (x : ℝ) : ∃ (y : ℝ), x^4 + 12 = (x^2 - 3*x + 3) * y := by
  sorry

end NUMINAMATH_CALUDE_factor_of_x4_plus_12_l986_98608


namespace NUMINAMATH_CALUDE_people_made_happy_l986_98668

/-- The number of institutions made happy -/
def institutions : ℕ := 6

/-- The number of people in each institution -/
def people_per_institution : ℕ := 80

/-- The total number of people made happy -/
def total_people_happy : ℕ := institutions * people_per_institution

theorem people_made_happy : total_people_happy = 480 := by
  sorry

end NUMINAMATH_CALUDE_people_made_happy_l986_98668


namespace NUMINAMATH_CALUDE_solve_burger_problem_l986_98650

/-- Represents the problem of calculating the number of double burgers bought. -/
def BurgerProblem (total_cost : ℚ) (total_burgers : ℕ) (single_cost : ℚ) (double_cost : ℚ) : Prop :=
  ∃ (single_burgers double_burgers : ℕ),
    single_burgers + double_burgers = total_burgers ∧
    single_cost * single_burgers + double_cost * double_burgers = total_cost ∧
    double_burgers = 29

/-- Theorem stating the solution to the burger problem. -/
theorem solve_burger_problem :
  BurgerProblem 64.5 50 1 1.5 :=
sorry

end NUMINAMATH_CALUDE_solve_burger_problem_l986_98650


namespace NUMINAMATH_CALUDE_compound_interest_rate_l986_98647

/-- Given a principal amount, final amount, and time period, 
    calculate the compound interest rate. -/
theorem compound_interest_rate 
  (P : ℝ) (A : ℝ) (n : ℕ) 
  (h_P : P = 453.51473922902494)
  (h_A : A = 500)
  (h_n : n = 2) :
  ∃ r : ℝ, A = P * (1 + r)^n := by
sorry

end NUMINAMATH_CALUDE_compound_interest_rate_l986_98647


namespace NUMINAMATH_CALUDE_friendship_ratio_theorem_l986_98631

/-- Represents a boy in the school -/
structure Boy where
  id : Nat

/-- Represents a girl in the school -/
structure Girl where
  id : Nat

/-- The number of girls who know a given boy -/
def d_Boy (b : Boy) : ℕ := sorry

/-- The number of boys who know a given girl -/
def d_Girl (g : Girl) : ℕ := sorry

/-- Represents that a boy and a girl know each other -/
def knows (b : Boy) (g : Girl) : Prop := sorry

theorem friendship_ratio_theorem 
  (n m : ℕ) 
  (boys : Finset Boy) 
  (girls : Finset Girl) 
  (h_boys : boys.card = n) 
  (h_girls : girls.card = m) 
  (h_girls_know_boy : ∀ g : Girl, ∃ b : Boy, knows b g) :
  ∃ (b : Boy) (g : Girl), 
    knows b g ∧ (d_Boy b : ℚ) / (d_Girl g : ℚ) ≥ (m : ℚ) / (n : ℚ) := by
  sorry

end NUMINAMATH_CALUDE_friendship_ratio_theorem_l986_98631


namespace NUMINAMATH_CALUDE_sum_of_A_and_C_is_five_l986_98602

-- Define the multiplication problem
def multiplication_problem (A B C D : Nat) : Prop :=
  A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D ∧
  A < 10 ∧ B < 10 ∧ C < 10 ∧ D < 10 ∧
  (100 * A + 10 * B + A) * D = 1000 * C + 100 * B + 10 * A + D

-- State the theorem
theorem sum_of_A_and_C_is_five :
  ∀ A B C D : Nat, multiplication_problem A B C D → A + C = 5 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_A_and_C_is_five_l986_98602


namespace NUMINAMATH_CALUDE_books_from_first_shop_l986_98615

theorem books_from_first_shop 
  (total_first : ℕ) 
  (books_second : ℕ) 
  (total_second : ℕ) 
  (avg_price : ℕ) :
  total_first = 1500 →
  books_second = 60 →
  total_second = 340 →
  avg_price = 16 →
  ∃ (books_first : ℕ), 
    (total_first + total_second) / (books_first + books_second) = avg_price ∧
    books_first = 55 :=
by sorry

end NUMINAMATH_CALUDE_books_from_first_shop_l986_98615


namespace NUMINAMATH_CALUDE_power_multiplication_l986_98694

theorem power_multiplication (a : ℝ) : a^2 * a = a^3 := by
  sorry

end NUMINAMATH_CALUDE_power_multiplication_l986_98694


namespace NUMINAMATH_CALUDE_paint_cans_problem_l986_98686

theorem paint_cans_problem (original_rooms : ℕ) (lost_cans : ℕ) (remaining_rooms : ℕ) :
  original_rooms = 50 →
  lost_cans = 5 →
  remaining_rooms = 35 →
  (∃ (cans_per_room : ℚ), 
    cans_per_room * (original_rooms - remaining_rooms) = lost_cans ∧
    cans_per_room * remaining_rooms = 12) :=
by sorry

end NUMINAMATH_CALUDE_paint_cans_problem_l986_98686


namespace NUMINAMATH_CALUDE_abs_five_minus_e_l986_98684

-- Define e as a real number approximately equal to 2.718
def e : ℝ := 2.718

-- State the theorem
theorem abs_five_minus_e : |5 - e| = 2.282 := by
  sorry

end NUMINAMATH_CALUDE_abs_five_minus_e_l986_98684


namespace NUMINAMATH_CALUDE_turn_duration_is_one_hour_l986_98642

/-- Represents the time taken to complete the work individually -/
structure WorkTime where
  a : ℝ
  b : ℝ

/-- Represents the amount of work done per hour -/
structure WorkRate where
  a : ℝ
  b : ℝ

/-- The duration of each turn when working alternately -/
def turn_duration (wt : WorkTime) (wr : WorkRate) : ℝ :=
  sorry

/-- The theorem stating that the turn duration is 1 hour -/
theorem turn_duration_is_one_hour (wt : WorkTime) (wr : WorkRate) :
  wt.a = 4 →
  wt.b = 12 →
  wr.a = 1 / wt.a →
  wr.b = 1 / wt.b →
  (3 * wr.a * turn_duration wt wr + 3 * wr.b * turn_duration wt wr = 1) →
  turn_duration wt wr = 1 :=
sorry

end NUMINAMATH_CALUDE_turn_duration_is_one_hour_l986_98642


namespace NUMINAMATH_CALUDE_systematic_sampling_theorem_l986_98680

def systematic_sampling (total_members : ℕ) (num_groups : ℕ) (group_number : ℕ) (number_in_group : ℕ) : ℕ :=
  number_in_group - (group_number - 1) * (total_members / num_groups)

theorem systematic_sampling_theorem (total_members num_groups group_5 group_3 : ℕ) 
  (h1 : total_members = 200)
  (h2 : num_groups = 40)
  (h3 : group_5 = 5)
  (h4 : group_3 = 3)
  (h5 : systematic_sampling total_members num_groups group_5 22 = 22) :
  systematic_sampling total_members num_groups group_3 22 = 12 :=
by
  sorry

end NUMINAMATH_CALUDE_systematic_sampling_theorem_l986_98680


namespace NUMINAMATH_CALUDE_student_rank_l986_98646

theorem student_rank (total : ℕ) (left_rank : ℕ) (right_rank : ℕ) :
  total = 31 → left_rank = 11 → right_rank = total - left_rank + 1 → right_rank = 21 := by
  sorry

end NUMINAMATH_CALUDE_student_rank_l986_98646


namespace NUMINAMATH_CALUDE_condition_relationship_l986_98655

theorem condition_relationship (a b : ℝ) :
  (∀ a b, a > 2 ∧ b > 2 → a + b > 4) ∧
  (∃ a b, a + b > 4 ∧ ¬(a > 2 ∧ b > 2)) :=
by sorry

end NUMINAMATH_CALUDE_condition_relationship_l986_98655


namespace NUMINAMATH_CALUDE_positive_real_inequality_l986_98630

theorem positive_real_inequality (a : ℝ) (h : a > 0) : a + 1/a ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_positive_real_inequality_l986_98630


namespace NUMINAMATH_CALUDE_quadratic_roots_after_modification_l986_98601

theorem quadratic_roots_after_modification (a b t l : ℝ) :
  -1 < t → t < 0 →
  (∀ x, x^2 + a*x + b = 0 ↔ x = t ∨ x = l) →
  ∃ r₁ r₂, r₁ ≠ r₂ ∧ ∀ x, x^2 + (a+t)*x + (b+t) = 0 ↔ x = r₁ ∨ x = r₂ :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_after_modification_l986_98601


namespace NUMINAMATH_CALUDE_complex_power_sum_l986_98618

theorem complex_power_sum (i : ℂ) : i^2 = -1 → i^123 + i^223 + i^323 = -3*i := by
  sorry

end NUMINAMATH_CALUDE_complex_power_sum_l986_98618


namespace NUMINAMATH_CALUDE_powers_of_four_unit_digits_l986_98607

theorem powers_of_four_unit_digits (a b c : ℕ+) : 
  ¬(∃ (x y z : ℕ), x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ 
    x = (4^a.val) % 10 ∧ 
    y = (4^b.val) % 10 ∧ 
    z = (4^c.val) % 10) := by
  sorry

end NUMINAMATH_CALUDE_powers_of_four_unit_digits_l986_98607


namespace NUMINAMATH_CALUDE_quadratic_functions_intersect_l986_98643

/-- A quadratic function of the form f(x) = x^2 + px + q where p + q = 2002 -/
structure QuadraticFunction where
  p : ℝ
  q : ℝ
  h : p + q = 2002

/-- The theorem stating that all quadratic functions satisfying the condition
    intersect at the point (1, 2003) -/
theorem quadratic_functions_intersect (f : QuadraticFunction) :
  f.p + f.q^2 + f.p + f.q = 2003 := by
  sorry

#check quadratic_functions_intersect

end NUMINAMATH_CALUDE_quadratic_functions_intersect_l986_98643


namespace NUMINAMATH_CALUDE_seating_theorem_l986_98659

def seating_arrangements (total_people : ℕ) (rows : ℕ) (people_per_row : ℕ) 
  (specific_front : ℕ) (specific_back : ℕ) : ℕ :=
  let front_arrangements := Nat.descFactorial people_per_row specific_front
  let back_arrangements := Nat.descFactorial people_per_row specific_back
  let remaining_people := total_people - specific_front - specific_back
  let remaining_arrangements := Nat.factorial remaining_people
  front_arrangements * back_arrangements * remaining_arrangements

theorem seating_theorem : 
  seating_arrangements 8 2 4 2 1 = 5760 := by
  sorry

end NUMINAMATH_CALUDE_seating_theorem_l986_98659


namespace NUMINAMATH_CALUDE_circle_circumference_irrational_l986_98677

/-- A circle with rational diameter has irrational circumference -/
theorem circle_circumference_irrational (d : ℚ) :
  ∃ (C : ℝ), C = π * (d : ℝ) ∧ Irrational C := by
  sorry

end NUMINAMATH_CALUDE_circle_circumference_irrational_l986_98677


namespace NUMINAMATH_CALUDE_geometric_sequence_minimum_value_l986_98689

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_minimum_value
  (a : ℕ → ℝ)
  (h_geo : is_geometric_sequence a)
  (h_pos : ∀ n, a n > 0)
  (h_a7 : a 7 = Real.sqrt 2 / 2) :
  (1 / a 3 + 2 / a 11) ≥ 4 ∧
  ∃ x : ℝ, (1 / a 3 + 2 / a 11) = x → x ≥ 4 :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_minimum_value_l986_98689


namespace NUMINAMATH_CALUDE_edwards_summer_earnings_l986_98695

/-- Given Edward's lawn mowing business earnings and expenses, prove the amount he made in the summer. -/
theorem edwards_summer_earnings (spring_earnings : ℕ) (supplies_cost : ℕ) (final_amount : ℕ) :
  spring_earnings = 2 →
  supplies_cost = 5 →
  final_amount = 24 →
  ∃ summer_earnings : ℕ, spring_earnings + summer_earnings - supplies_cost = final_amount ∧ summer_earnings = 27 :=
by sorry

end NUMINAMATH_CALUDE_edwards_summer_earnings_l986_98695
