import Mathlib

namespace NUMINAMATH_CALUDE_negative_x_to_negative_k_is_positive_l2219_221900

theorem negative_x_to_negative_k_is_positive
  (x : ℝ) (k : ℤ) (hx : x < 0) (hk : k > 0) :
  -x^(-k) > 0 :=
by sorry

end NUMINAMATH_CALUDE_negative_x_to_negative_k_is_positive_l2219_221900


namespace NUMINAMATH_CALUDE_abs_z_squared_value_l2219_221920

theorem abs_z_squared_value (z : ℂ) (h : z^2 + Complex.abs z^2 = 7 + 6*I) : 
  Complex.abs z^2 = 85/14 := by
  sorry

end NUMINAMATH_CALUDE_abs_z_squared_value_l2219_221920


namespace NUMINAMATH_CALUDE_polar_coordinates_of_point_l2219_221916

theorem polar_coordinates_of_point (x y : ℝ) (ρ θ : ℝ) :
  x = 1/2 ∧ y = -Real.sqrt 3 / 2 →
  ρ = Real.sqrt (x^2 + y^2) ∧
  θ = Real.arccos (x / ρ) + (if y < 0 then 2 * Real.pi else 0) →
  ρ = 1 ∧ θ = 5 * Real.pi / 3 := by
  sorry

end NUMINAMATH_CALUDE_polar_coordinates_of_point_l2219_221916


namespace NUMINAMATH_CALUDE_tens_digit_of_9_pow_2023_l2219_221940

theorem tens_digit_of_9_pow_2023 : ∃ n : ℕ, 9^2023 ≡ 80 + n [ZMOD 100] ∧ 0 ≤ n ∧ n < 10 :=
sorry

end NUMINAMATH_CALUDE_tens_digit_of_9_pow_2023_l2219_221940


namespace NUMINAMATH_CALUDE_soup_feeding_theorem_l2219_221918

/-- Represents the number of people a can of soup can feed -/
structure SoupCan where
  adults : Nat
  children : Nat

/-- Calculates the number of adults that can be fed with the remaining soup -/
def remainingAdults (totalCans : Nat) (canCapacity : SoupCan) (childrenFed : Nat) : Nat :=
  let cansForChildren := childrenFed / canCapacity.children
  let remainingCans := totalCans - cansForChildren
  remainingCans * canCapacity.adults

/-- Theorem stating that given the conditions, 20 adults can be fed with the remaining soup -/
theorem soup_feeding_theorem (totalCans : Nat) (canCapacity : SoupCan) (childrenFed : Nat) :
  totalCans = 10 →
  canCapacity.adults = 4 →
  canCapacity.children = 8 →
  childrenFed = 40 →
  remainingAdults totalCans canCapacity childrenFed = 20 := by
  sorry

end NUMINAMATH_CALUDE_soup_feeding_theorem_l2219_221918


namespace NUMINAMATH_CALUDE_solution_when_a_eq_one_two_solutions_range_max_value_F_l2219_221994

-- Define the functions
def f (a x : ℝ) := |x - a|
def g (a x : ℝ) := a * x
def F (a x : ℝ) := g a x * f a x

-- Theorem 1
theorem solution_when_a_eq_one :
  ∃ x : ℝ, f 1 x = g 1 x ∧ x = 1/2 := by sorry

-- Theorem 2
theorem two_solutions_range :
  ∀ a : ℝ, (∃ x y : ℝ, x ≠ y ∧ f a x = g a x ∧ f a y = g a y) ↔ 
  (a > -1 ∧ a < 0) ∨ (a > 0 ∧ a < 1) := by sorry

-- Theorem 3
theorem max_value_F :
  ∀ a : ℝ, a > 0 → 
  (∃ max : ℝ, ∀ x : ℝ, x ∈ Set.Icc 1 2 → F a x ≤ max) ∧
  (let max := if a < 5/3 then 4*a - 2*a^2
              else if a ≤ 2 then a^2 - a
              else if a < 4 then a^3/4
              else 2*a^2 - 4*a;
   ∀ x : ℝ, x ∈ Set.Icc 1 2 → F a x ≤ max) := by sorry

end NUMINAMATH_CALUDE_solution_when_a_eq_one_two_solutions_range_max_value_F_l2219_221994


namespace NUMINAMATH_CALUDE_log_sum_equality_l2219_221951

theorem log_sum_equality : 
  Real.log 8 / Real.log 2 + 3 * (Real.log 4 / Real.log 2) + 
  4 * (Real.log 16 / Real.log 4) + 2 * (Real.log 32 / Real.log 8) = 61 / 3 := by
  sorry

end NUMINAMATH_CALUDE_log_sum_equality_l2219_221951


namespace NUMINAMATH_CALUDE_sara_movie_tickets_l2219_221942

-- Define the constants
def ticket_cost : ℚ := 10.62
def rental_cost : ℚ := 1.59
def purchase_cost : ℚ := 13.95
def total_spent : ℚ := 36.78

-- Define the theorem
theorem sara_movie_tickets :
  ∃ (n : ℕ), n * ticket_cost + rental_cost + purchase_cost = total_spent ∧ n = 2 :=
sorry

end NUMINAMATH_CALUDE_sara_movie_tickets_l2219_221942


namespace NUMINAMATH_CALUDE_total_leaves_eq_696_l2219_221988

def basil_pots : ℕ := 3
def rosemary_pots : ℕ := 9
def thyme_pots : ℕ := 6
def cilantro_pots : ℕ := 7
def lavender_pots : ℕ := 4

def basil_leaves_per_plant : ℕ := 4
def rosemary_leaves_per_plant : ℕ := 18
def thyme_leaves_per_plant : ℕ := 30
def cilantro_leaves_per_plant : ℕ := 42
def lavender_leaves_per_plant : ℕ := 12

def total_leaves : ℕ := 
  basil_pots * basil_leaves_per_plant +
  rosemary_pots * rosemary_leaves_per_plant +
  thyme_pots * thyme_leaves_per_plant +
  cilantro_pots * cilantro_leaves_per_plant +
  lavender_pots * lavender_leaves_per_plant

theorem total_leaves_eq_696 : total_leaves = 696 := by
  sorry

end NUMINAMATH_CALUDE_total_leaves_eq_696_l2219_221988


namespace NUMINAMATH_CALUDE_trajectory_of_right_angle_vertex_l2219_221938

/-- Given points M(-2,0) and N(2,0), prove that any point P(x,y) forming a right-angled triangle
    with MN as the hypotenuse satisfies the equation x^2 + y^2 = 4, where x ≠ ±2. -/
theorem trajectory_of_right_angle_vertex (x y : ℝ) :
  x ≠ -2 → x ≠ 2 →
  (x + 2)^2 + y^2 + (x - 2)^2 + y^2 = 16 →
  x^2 + y^2 = 4 :=
by sorry

end NUMINAMATH_CALUDE_trajectory_of_right_angle_vertex_l2219_221938


namespace NUMINAMATH_CALUDE_count_numbers_with_seven_is_152_l2219_221925

/-- A function that checks if a natural number contains the digit 7 -/
def contains_seven (n : ℕ) : Bool :=
  sorry

/-- The count of natural numbers from 1 to 800 containing the digit 7 -/
def count_numbers_with_seven : ℕ :=
  (List.range 800).filter (λ n => contains_seven (n + 1)) |>.length

/-- Theorem stating that the count of numbers with seven is 152 -/
theorem count_numbers_with_seven_is_152 :
  count_numbers_with_seven = 152 :=
by sorry

end NUMINAMATH_CALUDE_count_numbers_with_seven_is_152_l2219_221925


namespace NUMINAMATH_CALUDE_total_cases_after_three_days_l2219_221929

-- Define the parameters
def initial_cases : ℕ := 2000
def increase_rate : ℚ := 20 / 100
def recovery_rate : ℚ := 2 / 100
def days : ℕ := 3

-- Function to calculate the cases for the next day
def next_day_cases (current_cases : ℚ) : ℚ :=
  current_cases + current_cases * increase_rate - current_cases * recovery_rate

-- Function to calculate cases after n days
def cases_after_days (n : ℕ) : ℚ :=
  match n with
  | 0 => initial_cases
  | n + 1 => next_day_cases (cases_after_days n)

-- Theorem statement
theorem total_cases_after_three_days :
  ⌊cases_after_days days⌋ = 3286 :=
sorry

end NUMINAMATH_CALUDE_total_cases_after_three_days_l2219_221929


namespace NUMINAMATH_CALUDE_pens_left_in_jar_l2219_221984

/-- The number of pens left in a jar after removing some pens -/
def pens_left (initial_blue initial_black initial_red removed_blue removed_black : ℕ) : ℕ :=
  (initial_blue - removed_blue) + (initial_black - removed_black) + initial_red

/-- Theorem stating the number of pens left in the jar -/
theorem pens_left_in_jar : pens_left 9 21 6 4 7 = 25 := by
  sorry

end NUMINAMATH_CALUDE_pens_left_in_jar_l2219_221984


namespace NUMINAMATH_CALUDE_hypotenuse_value_l2219_221927

-- Define a right triangle with sides 3, 5, and x (hypotenuse)
def right_triangle (x : ℝ) : Prop :=
  x > 0 ∧ x^2 = 3^2 + 5^2

-- Theorem statement
theorem hypotenuse_value :
  ∃ x : ℝ, right_triangle x ∧ x = Real.sqrt 34 :=
by sorry

end NUMINAMATH_CALUDE_hypotenuse_value_l2219_221927


namespace NUMINAMATH_CALUDE_monotonic_function_property_l2219_221950

/-- A monotonic function f: ℝ → ℝ satisfying f[f(x) - 3^x] = 4 for all x ∈ ℝ has f(2) = 10 -/
theorem monotonic_function_property (f : ℝ → ℝ) 
  (h_monotonic : Monotone f)
  (h_property : ∀ x : ℝ, f (f x - 3^x) = 4) :
  f 2 = 10 := by sorry

end NUMINAMATH_CALUDE_monotonic_function_property_l2219_221950


namespace NUMINAMATH_CALUDE_right_triangles_on_circle_l2219_221923

theorem right_triangles_on_circle (n : ℕ) (h : n = 100) :
  ¬ (∃ (t : ℕ), t = 1000 ∧ t = (n / 2) * (n - 2)) :=
by
  sorry

end NUMINAMATH_CALUDE_right_triangles_on_circle_l2219_221923


namespace NUMINAMATH_CALUDE_geometric_sequence_a10_l2219_221999

def geometric_sequence (a : ℕ → ℝ) (q : ℝ) := ∀ n, a (n + 1) = a n * q

theorem geometric_sequence_a10 (a : ℕ → ℝ) (q : ℝ) :
  geometric_sequence a q → a 6 = 2/3 → q = Real.sqrt 3 → a 10 = 6 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_a10_l2219_221999


namespace NUMINAMATH_CALUDE_laborer_wage_calculation_l2219_221958

/-- The daily wage of a general laborer -/
def laborer_wage : ℕ :=
  -- Define the wage here
  sorry

/-- The number of people hired -/
def total_hired : ℕ := 31

/-- The total payroll in dollars -/
def total_payroll : ℕ := 3952

/-- The daily wage of a heavy operator -/
def operator_wage : ℕ := 129

/-- The number of laborers employed -/
def laborers_employed : ℕ := 1

theorem laborer_wage_calculation : 
  laborer_wage = 82 ∧
  total_hired * operator_wage - (total_hired - laborers_employed) * operator_wage + laborer_wage = total_payroll :=
by sorry

end NUMINAMATH_CALUDE_laborer_wage_calculation_l2219_221958


namespace NUMINAMATH_CALUDE_geometric_sequence_arithmetic_mean_l2219_221972

/-- The arithmetic mean of the first three terms of a geometric sequence 
    with first term 4 and common ratio 3 is 52/3. -/
theorem geometric_sequence_arithmetic_mean : 
  let a : ℝ := 4  -- First term
  let r : ℝ := 3  -- Common ratio
  let term1 : ℝ := a
  let term2 : ℝ := a * r
  let term3 : ℝ := a * r^2
  (term1 + term2 + term3) / 3 = 52 / 3 := by
  sorry


end NUMINAMATH_CALUDE_geometric_sequence_arithmetic_mean_l2219_221972


namespace NUMINAMATH_CALUDE_even_function_implies_a_eq_neg_one_l2219_221971

/-- A function f is even if f(-x) = f(x) for all x in its domain -/
def IsEven (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

/-- The function f(x) = (x+1)(x+a) -/
def f (a : ℝ) : ℝ → ℝ := fun x ↦ (x + 1) * (x + a)

theorem even_function_implies_a_eq_neg_one :
  IsEven (f a) → a = -1 := by sorry

end NUMINAMATH_CALUDE_even_function_implies_a_eq_neg_one_l2219_221971


namespace NUMINAMATH_CALUDE_human_family_members_l2219_221906

/-- Represents the number of feet for each type of animal and the alien pet. -/
structure AnimalFeet where
  birds : ℕ
  dogs : ℕ
  cats : ℕ
  alien : ℕ

/-- Represents the number of heads for each type of animal and the alien pet. -/
structure AnimalHeads where
  birds : ℕ
  dogs : ℕ
  cats : ℕ
  alien : ℕ

/-- Calculates the total number of feet for all animals and the alien pet. -/
def totalAnimalFeet (af : AnimalFeet) : ℕ :=
  af.birds + af.dogs + af.cats + af.alien

/-- Calculates the total number of heads for all animals and the alien pet. -/
def totalAnimalHeads (ah : AnimalHeads) : ℕ :=
  ah.birds + ah.dogs + ah.cats + ah.alien

/-- Theorem stating the number of human family members. -/
theorem human_family_members :
  ∃ (h : ℕ),
    let af : AnimalFeet := ⟨7, 13, 74, 6⟩
    let ah : AnimalHeads := ⟨4, 3, 18, 1⟩
    totalAnimalFeet af + 2 * h = totalAnimalHeads ah + h + 108 ∧ h = 34 := by
  sorry

end NUMINAMATH_CALUDE_human_family_members_l2219_221906


namespace NUMINAMATH_CALUDE_max_value_expression_l2219_221941

theorem max_value_expression (a b c d : ℕ) : 
  a ∈ ({1, 3, 5, 7} : Set ℕ) → 
  b ∈ ({1, 3, 5, 7} : Set ℕ) → 
  c ∈ ({1, 3, 5, 7} : Set ℕ) → 
  d ∈ ({1, 3, 5, 7} : Set ℕ) → 
  a ≠ b → a ≠ c → a ≠ d → b ≠ c → b ≠ d → c ≠ d →
  (a + b) * (c + d) + (a + 1) * (d + 1) ≤ 112 :=
by sorry

end NUMINAMATH_CALUDE_max_value_expression_l2219_221941


namespace NUMINAMATH_CALUDE_min_distance_squared_l2219_221909

/-- The minimum squared distance between a curve and a line -/
theorem min_distance_squared (a b m n : ℝ) : 
  a > 0 → 
  b = -1/2 * a^2 + 3 * Real.log a → 
  n = 2 * m + 1/2 → 
  ∃ (min_dist : ℝ), 
    (∀ (x y : ℝ), y = -1/2 * x^2 + 3 * Real.log x → 
      (x - m)^2 + (y - n)^2 ≥ min_dist) ∧
    min_dist = 9/5 := by
  sorry

end NUMINAMATH_CALUDE_min_distance_squared_l2219_221909


namespace NUMINAMATH_CALUDE_garrett_cat_count_l2219_221992

/-- The number of cats Mrs. Sheridan has -/
def sheridan_cats : ℕ := 11

/-- The difference between Mrs. Garrett's and Mrs. Sheridan's cats -/
def cat_difference : ℕ := 13

/-- Mrs. Garrett's cats -/
def garrett_cats : ℕ := sheridan_cats + cat_difference

theorem garrett_cat_count : garrett_cats = 24 := by
  sorry

end NUMINAMATH_CALUDE_garrett_cat_count_l2219_221992


namespace NUMINAMATH_CALUDE_at_least_one_leq_neg_four_l2219_221996

theorem at_least_one_leq_neg_four (a b c : ℝ) 
  (ha : a < 0) (hb : b < 0) (hc : c < 0) : 
  (a + 4 / b ≤ -4) ∨ (b + 4 / c ≤ -4) ∨ (c + 4 / a ≤ -4) := by
sorry

end NUMINAMATH_CALUDE_at_least_one_leq_neg_four_l2219_221996


namespace NUMINAMATH_CALUDE_completing_square_equivalence_l2219_221978

theorem completing_square_equivalence :
  ∀ x : ℝ, x^2 - 6*x + 2 = 0 ↔ (x - 3)^2 = 7 := by
  sorry

end NUMINAMATH_CALUDE_completing_square_equivalence_l2219_221978


namespace NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l2219_221919

/-- 
Given a geometric sequence {a_n} where a₁ = -1 and a₄ = 8,
prove that the common ratio is -2.
-/
theorem geometric_sequence_common_ratio :
  ∀ (a : ℕ → ℝ),
  (∀ n, a (n + 1) = a n * (a 2 / a 1)) →  -- Definition of geometric sequence
  a 1 = -1 →
  a 4 = 8 →
  a 2 / a 1 = -2 :=
by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l2219_221919


namespace NUMINAMATH_CALUDE_root_difference_ratio_l2219_221928

theorem root_difference_ratio (a b : ℝ) : 
  a > b ∧ b > 0 ∧ 
  a^2 - 6*a + 4 = 0 ∧ 
  b^2 - 6*b + 4 = 0 → 
  (Real.sqrt a - Real.sqrt b) / (Real.sqrt a + Real.sqrt b) = Real.sqrt 5 / 5 := by
  sorry

end NUMINAMATH_CALUDE_root_difference_ratio_l2219_221928


namespace NUMINAMATH_CALUDE_a_10_value_a_satisfies_conditions_l2219_221977

def sequence_a (n : ℕ+) : ℚ :=
  1 / (3 * n - 2)

theorem a_10_value :
  sequence_a 10 = 1 / 28 :=
by sorry

theorem a_satisfies_conditions :
  sequence_a 1 = 1 ∧
  ∀ n : ℕ+, 1 / sequence_a (n + 1) = 1 / sequence_a n + 3 :=
by sorry

end NUMINAMATH_CALUDE_a_10_value_a_satisfies_conditions_l2219_221977


namespace NUMINAMATH_CALUDE_max_y_value_l2219_221936

theorem max_y_value (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : x * Real.log (y / x) - y * Real.exp x + x * (x + 1) ≥ 0) : 
  y ≤ 1 / Real.exp 1 := by
sorry

end NUMINAMATH_CALUDE_max_y_value_l2219_221936


namespace NUMINAMATH_CALUDE_smallest_blocks_needed_l2219_221983

/-- Represents the dimensions of a block -/
structure Block where
  height : ℕ
  length : ℕ

/-- Represents the dimensions of the wall -/
structure Wall where
  length : ℕ
  height : ℕ

/-- Calculates the number of blocks needed for the wall -/
def blocksNeeded (wall : Wall) (block3 : Block) (block1 : Block) : ℕ :=
  let rowsCount := wall.height / block3.height
  let oddRowBlocks := wall.length / block3.length
  let evenRowBlocks := 2 + (wall.length - 2 * block1.length) / block3.length
  (rowsCount / 2) * oddRowBlocks + ((rowsCount + 1) / 2) * evenRowBlocks

/-- The theorem stating the smallest number of blocks needed -/
theorem smallest_blocks_needed (wall : Wall) (block3 : Block) (block1 : Block) :
  wall.length = 120 ∧ wall.height = 8 ∧
  block3.height = 1 ∧ block3.length = 3 ∧
  block1.height = 1 ∧ block1.length = 1 →
  blocksNeeded wall block3 block1 = 324 := by
  sorry

#eval blocksNeeded ⟨120, 8⟩ ⟨1, 3⟩ ⟨1, 1⟩

end NUMINAMATH_CALUDE_smallest_blocks_needed_l2219_221983


namespace NUMINAMATH_CALUDE_num_triangles_in_dodecagon_l2219_221998

/-- A regular dodecagon has 12 vertices -/
def regular_dodecagon_vertices : ℕ := 12

/-- The number of triangles formed by choosing 3 vertices from a regular dodecagon -/
def num_triangles : ℕ := Nat.choose regular_dodecagon_vertices 3

/-- Theorem: The number of triangles formed by choosing 3 vertices from a regular dodecagon is 220 -/
theorem num_triangles_in_dodecagon : num_triangles = 220 := by sorry

end NUMINAMATH_CALUDE_num_triangles_in_dodecagon_l2219_221998


namespace NUMINAMATH_CALUDE_bus_passengers_l2219_221993

theorem bus_passengers (initial_passengers : ℕ) : 
  initial_passengers + 16 - 22 + 5 = 49 → initial_passengers = 50 := by
  sorry

end NUMINAMATH_CALUDE_bus_passengers_l2219_221993


namespace NUMINAMATH_CALUDE_arithmetic_equality_l2219_221959

theorem arithmetic_equality : 4 * 7 + 5 * 12 + 12 * 4 + 4 * 9 = 172 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_equality_l2219_221959


namespace NUMINAMATH_CALUDE_friends_team_assignment_l2219_221964

theorem friends_team_assignment (n : ℕ) (k : ℕ) (h1 : n = 8) (h2 : k = 4) :
  k ^ n = 65536 := by
  sorry

end NUMINAMATH_CALUDE_friends_team_assignment_l2219_221964


namespace NUMINAMATH_CALUDE_linear_functions_through_point_l2219_221907

theorem linear_functions_through_point :
  ∃ (x₀ y₀ : ℝ) (k b : Fin 10 → ℕ),
    (∀ i : Fin 10, 1 ≤ k i ∧ k i ≤ 20 ∧ 1 ≤ b i ∧ b i ≤ 20) ∧
    (∀ i j : Fin 10, i ≠ j → k i ≠ k j ∧ b i ≠ b j) ∧
    (∀ i : Fin 10, y₀ = k i * x₀ + b i) := by
  sorry

end NUMINAMATH_CALUDE_linear_functions_through_point_l2219_221907


namespace NUMINAMATH_CALUDE_condition_relationship_l2219_221912

theorem condition_relationship : ¬(∀ x y : ℝ, (x > 1 ∧ y > 1) ↔ x + y > 3) :=
by
  sorry

end NUMINAMATH_CALUDE_condition_relationship_l2219_221912


namespace NUMINAMATH_CALUDE_hexagon_congruent_angles_l2219_221975

/-- In a hexagon with three congruent angles and two pairs of supplementary angles,
    each of the congruent angles measures 120 degrees. -/
theorem hexagon_congruent_angles (F I G U R E : Real) : 
  F = I ∧ I = U ∧  -- Three angles are congruent
  G + E = 180 ∧    -- One pair of supplementary angles
  R + U = 180 ∧    -- Another pair of supplementary angles
  F + I + G + U + R + E = 720  -- Sum of angles in a hexagon
  → U = 120 := by sorry

end NUMINAMATH_CALUDE_hexagon_congruent_angles_l2219_221975


namespace NUMINAMATH_CALUDE_square_area_from_perimeter_l2219_221957

theorem square_area_from_perimeter (p : ℝ) (p_pos : p > 0) : 
  let perimeter := 12 * p
  let side_length := perimeter / 4
  let area := side_length ^ 2
  area = 9 * p ^ 2 := by
sorry

end NUMINAMATH_CALUDE_square_area_from_perimeter_l2219_221957


namespace NUMINAMATH_CALUDE_function_range_theorem_l2219_221979

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x + 2 * x - a

theorem function_range_theorem (a : ℝ) :
  (∃ x₀ y₀ : ℝ, y₀ = Real.sin x₀ ∧ f a (f a y₀) = y₀) →
  a ∈ Set.Icc (Real.exp (-1) - 1) (Real.exp 1 + 1) :=
by sorry

end NUMINAMATH_CALUDE_function_range_theorem_l2219_221979


namespace NUMINAMATH_CALUDE_quadratic_inequality_l2219_221931

def f (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

theorem quadratic_inequality (a b c : ℝ) 
  (h1 : ∀ x, a * x^2 + b * x + c > 0 ↔ x < -2 ∨ x > 4) :
  f a b c 2 < f a b c (-1) ∧ f a b c (-1) < f a b c 5 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_l2219_221931


namespace NUMINAMATH_CALUDE_set_operations_l2219_221924

-- Define the sets A and B
def A : Set ℝ := {x | -2 < x ∧ x < 3}
def B : Set ℝ := {x | -3 < x ∧ x ≤ 3}

-- State the theorem
theorem set_operations :
  (Set.univ \ A = {x | x ≥ 3 ∨ x ≤ -2}) ∧
  (A ∩ B = {x | -2 < x ∧ x < 3}) ∧
  (Set.univ \ (A ∩ B) = {x | x ≥ 3 ∨ x ≤ -2}) ∧
  ((Set.univ \ A) ∩ B = {x | (-3 < x ∧ x ≤ -2) ∨ x = 3}) := by
  sorry

end NUMINAMATH_CALUDE_set_operations_l2219_221924


namespace NUMINAMATH_CALUDE_right_triangle_shorter_leg_l2219_221981

theorem right_triangle_shorter_leg (a b c m : ℝ) : 
  a > 0 → b > 0 → c > 0 → m > 0 →
  a^2 + b^2 = c^2 →  -- Right triangle
  m = c / 2 →        -- Median to hypotenuse
  m = 15 →           -- Median length
  b = a + 9 →        -- One leg 9 units longer
  a = (-9 + Real.sqrt 1719) / 2 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_shorter_leg_l2219_221981


namespace NUMINAMATH_CALUDE_black_and_white_films_count_l2219_221954

theorem black_and_white_films_count 
  (B : ℕ) -- number of black-and-white films
  (x y : ℚ) -- parameters for selection percentage and color films
  (h1 : y / x > 0) -- ensure y/x is positive
  (h2 : y > 0) -- ensure y is positive
  (h3 : (4 * y) / ((y / x * B / 100) + 4 * y) = 10 / 11) -- fraction of selected color films
  : B = 40 * x := by
sorry

end NUMINAMATH_CALUDE_black_and_white_films_count_l2219_221954


namespace NUMINAMATH_CALUDE_x_plus_p_equals_2p_plus_3_l2219_221932

theorem x_plus_p_equals_2p_plus_3 (x p : ℝ) (h1 : |x - 3| = p) (h2 : x > 3) : x + p = 2*p + 3 := by
  sorry

end NUMINAMATH_CALUDE_x_plus_p_equals_2p_plus_3_l2219_221932


namespace NUMINAMATH_CALUDE_extreme_values_and_interval_extrema_l2219_221914

-- Define the function f(x) = x^3 - 3x
def f (x : ℝ) : ℝ := x^3 - 3*x

-- Define the interval [-3, 3/2]
def interval : Set ℝ := {x | -3 ≤ x ∧ x ≤ 3/2}

theorem extreme_values_and_interval_extrema :
  -- Global maximum
  (∃ (x : ℝ), f x = 2 ∧ ∀ (y : ℝ), f y ≤ f x) ∧
  -- Global minimum
  (∃ (x : ℝ), f x = -2 ∧ ∀ (y : ℝ), f y ≥ f x) ∧
  -- Maximum on the interval
  (∃ (x : ℝ), x ∈ interval ∧ f x = 2 ∧ ∀ (y : ℝ), y ∈ interval → f y ≤ f x) ∧
  -- Minimum on the interval
  (∃ (x : ℝ), x ∈ interval ∧ f x = -18 ∧ ∀ (y : ℝ), y ∈ interval → f y ≥ f x) :=
by sorry


end NUMINAMATH_CALUDE_extreme_values_and_interval_extrema_l2219_221914


namespace NUMINAMATH_CALUDE_squared_sum_bound_l2219_221946

theorem squared_sum_bound (a b : ℝ) (x₁ x₂ : ℝ) : 
  (3 * x₁^2 + 3*(a+b)*x₁ + 4*a*b = 0) →
  (3 * x₂^2 + 3*(a+b)*x₂ + 4*a*b = 0) →
  (x₁ * (x₁ + 1) + x₂ * (x₂ + 1) = (x₁ + 1) * (x₂ + 1)) →
  (a + b)^2 ≤ 4 := by
sorry

end NUMINAMATH_CALUDE_squared_sum_bound_l2219_221946


namespace NUMINAMATH_CALUDE_valid_pairs_count_l2219_221952

/-- Represents the number of books in each category -/
def num_books_per_category : ℕ := 4

/-- Represents the total number of books -/
def total_books : ℕ := 3 * num_books_per_category

/-- Represents the number of novels -/
def num_novels : ℕ := 2 * num_books_per_category

/-- Calculates the number of ways to choose 2 books such that each pair includes at least one novel -/
def count_valid_pairs : ℕ :=
  let total_choices := num_novels * (total_books - num_books_per_category)
  let overcounted_pairs := num_novels * num_books_per_category
  (total_choices - overcounted_pairs) / 2

theorem valid_pairs_count : count_valid_pairs = 28 := by
  sorry

end NUMINAMATH_CALUDE_valid_pairs_count_l2219_221952


namespace NUMINAMATH_CALUDE_complex_absolute_value_product_l2219_221913

theorem complex_absolute_value_product : 
  Complex.abs ((3 * Real.sqrt 5 - 5 * Complex.I) * (2 * Real.sqrt 2 + 4 * Complex.I)) = 12 * Real.sqrt 35 := by
  sorry

end NUMINAMATH_CALUDE_complex_absolute_value_product_l2219_221913


namespace NUMINAMATH_CALUDE_collinear_points_b_value_l2219_221949

/-- A point in 2D space --/
structure Point where
  x : ℝ
  y : ℝ

/-- Check if three points are collinear --/
def collinear (p1 p2 p3 : Point) : Prop :=
  (p2.y - p1.y) * (p3.x - p1.x) = (p3.y - p1.y) * (p2.x - p1.x)

theorem collinear_points_b_value :
  ∀ b : ℝ,
  let A : Point := ⟨3, 1⟩
  let B : Point := ⟨-2, b⟩
  let C : Point := ⟨8, 11⟩
  collinear A B C → b = -9 := by
  sorry

end NUMINAMATH_CALUDE_collinear_points_b_value_l2219_221949


namespace NUMINAMATH_CALUDE_expression_equivalence_l2219_221986

theorem expression_equivalence :
  let original := -1/2 + Real.sqrt 3 / 2
  let a := -(1 + Real.sqrt 3) / 2
  let b := (Real.sqrt 3 - 1) / 2
  let c := -(1 - Real.sqrt 3) / 2
  let d := (-1 + Real.sqrt 3) / 2
  (a ≠ original) ∧ (b = original) ∧ (c = original) ∧ (d = original) := by
sorry

end NUMINAMATH_CALUDE_expression_equivalence_l2219_221986


namespace NUMINAMATH_CALUDE_sum_of_digits_equality_l2219_221968

def num1 : ℕ := (10^100 - 1) / 9
def num2 : ℕ := 4 * ((10^50 - 1) / 9)

def sumOfDigits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sumOfDigits (n / 10)

theorem sum_of_digits_equality :
  sumOfDigits (num1 * num2) = sumOfDigits (4 * (10^150 - 10^100 - 10^50 + 1) / 81) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_digits_equality_l2219_221968


namespace NUMINAMATH_CALUDE_probability_white_or_red_ball_l2219_221974

theorem probability_white_or_red_ball (white black red : ℕ) 
  (h_white : white = 8)
  (h_black : black = 7)
  (h_red : red = 4) :
  (white + red : ℚ) / (white + black + red) = 12 / 19 :=
by sorry

end NUMINAMATH_CALUDE_probability_white_or_red_ball_l2219_221974


namespace NUMINAMATH_CALUDE_rectangle_length_l2219_221937

/-- Given a rectangle with width 4 inches and area 8 square inches, prove its length is 2 inches. -/
theorem rectangle_length (width : ℝ) (area : ℝ) (h1 : width = 4) (h2 : area = 8) :
  area / width = 2 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_length_l2219_221937


namespace NUMINAMATH_CALUDE_january_text_messages_l2219_221921

-- Define the sequence
def text_message_sequence : ℕ → ℕ
| 0 => 1  -- November (first month)
| n + 1 => 2 * text_message_sequence n  -- Each subsequent month

-- Theorem statement
theorem january_text_messages : text_message_sequence 2 = 4 := by
  sorry

end NUMINAMATH_CALUDE_january_text_messages_l2219_221921


namespace NUMINAMATH_CALUDE_bacteria_growth_rate_l2219_221944

/-- Represents the growth rate of bacteria in a dish -/
def growth_rate (r : ℝ) : Prop :=
  ∀ (t : ℕ), t ≥ 0 → (1 / 16 : ℝ) * r^30 = r^26 ∧ r^30 = r^30

theorem bacteria_growth_rate :
  ∃ (r : ℝ), r > 0 ∧ growth_rate r ∧ r = 2 :=
sorry

end NUMINAMATH_CALUDE_bacteria_growth_rate_l2219_221944


namespace NUMINAMATH_CALUDE_robotic_octopus_dressing_orders_l2219_221963

/-- Represents the number of legs on the robotic octopus -/
def num_legs : ℕ := 4

/-- Represents the number of tentacles on the robotic octopus -/
def num_tentacles : ℕ := 2

/-- Represents the number of items per leg (glove and boot) -/
def items_per_leg : ℕ := 2

/-- Represents the number of items per tentacle (bracelet) -/
def items_per_tentacle : ℕ := 1

/-- Calculates the total number of items to be worn -/
def total_items : ℕ := num_legs * items_per_leg + num_tentacles * items_per_tentacle

/-- Theorem stating the number of different dressing orders for the robotic octopus -/
theorem robotic_octopus_dressing_orders : 
  (Nat.factorial num_tentacles) * (2 ^ num_legs) * (Nat.factorial (num_legs * items_per_leg)) = 1286400 :=
sorry

end NUMINAMATH_CALUDE_robotic_octopus_dressing_orders_l2219_221963


namespace NUMINAMATH_CALUDE_total_amount_is_120_l2219_221945

def amount_from_grandpa : ℕ := 30

def amount_from_grandma : ℕ := 3 * amount_from_grandpa

def total_amount : ℕ := amount_from_grandpa + amount_from_grandma

theorem total_amount_is_120 : total_amount = 120 := by
  sorry

end NUMINAMATH_CALUDE_total_amount_is_120_l2219_221945


namespace NUMINAMATH_CALUDE_popped_kernels_in_first_bag_l2219_221955

/-- Represents the number of kernels in a bag -/
structure BagOfKernels where
  total : ℕ
  popped : ℕ

/-- Given information about three bags of popcorn kernels, proves that
    the number of popped kernels in the first bag is 61. -/
theorem popped_kernels_in_first_bag
  (bag1 : BagOfKernels)
  (bag2 : BagOfKernels)
  (bag3 : BagOfKernels)
  (h1 : bag1.total = 75)
  (h2 : bag2.total = 50 ∧ bag2.popped = 42)
  (h3 : bag3.total = 100 ∧ bag3.popped = 82)
  (h_avg : (bag1.popped + bag2.popped + bag3.popped) / (bag1.total + bag2.total + bag3.total) = 82 / 100) :
  bag1.popped = 61 := by
  sorry

#check popped_kernels_in_first_bag

end NUMINAMATH_CALUDE_popped_kernels_in_first_bag_l2219_221955


namespace NUMINAMATH_CALUDE_unique_odd_divisors_pair_l2219_221962

/-- A number has an odd number of divisors if and only if it is a perfect square -/
def has_odd_divisors (n : ℕ) : Prop := ∃ k : ℕ, n = k^2

/-- The theorem states that 576 is the only positive integer n such that
    both n and n + 100 have an odd number of divisors -/
theorem unique_odd_divisors_pair :
  ∀ n : ℕ, n > 0 ∧ has_odd_divisors n ∧ has_odd_divisors (n + 100) → n = 576 :=
sorry

end NUMINAMATH_CALUDE_unique_odd_divisors_pair_l2219_221962


namespace NUMINAMATH_CALUDE_equation_solution_l2219_221953

theorem equation_solution : 
  ∃ (x₁ x₂ : ℝ), 
    x₁ > 0 ∧ x₂ > 0 ∧
    (∀ (x : ℝ), x > 0 → 
      ((1/3) * (4*x^2 - 3) = (x^2 - 75*x - 15) * (x^2 + 40*x + 8)) ↔ 
      (x = x₁ ∨ x = x₂)) ∧
    x₁ = (75 + Real.sqrt 5677) / 2 ∧
    x₂ = (-40 + Real.sqrt 1572) / 2 :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l2219_221953


namespace NUMINAMATH_CALUDE_final_amoeba_type_l2219_221933

/-- Represents the type of a Martian amoeba -/
inductive AmoebaTy
  | A
  | B
  | C

/-- Represents the state of the amoeba population -/
structure AmoebaPop where
  a : Nat
  b : Nat
  c : Nat

/-- Merges two amoebas of different types into the third type -/
def merge (pop : AmoebaPop) : AmoebaPop :=
  sorry

/-- Checks if a number is odd -/
def isOdd (n : Nat) : Prop :=
  n % 2 = 1

/-- The initial population of amoebas -/
def initialPop : AmoebaPop :=
  { a := 20, b := 21, c := 22 }

theorem final_amoeba_type (finalPop : AmoebaPop)
    (h : ∃ n : Nat, finalPop = (merge^[n] initialPop))
    (hTotal : finalPop.a + finalPop.b + finalPop.c = 1) :
    isOdd finalPop.b ∧ ¬isOdd finalPop.a ∧ ¬isOdd finalPop.c :=
  sorry

end NUMINAMATH_CALUDE_final_amoeba_type_l2219_221933


namespace NUMINAMATH_CALUDE_ones_digit_of_8_to_47_l2219_221948

theorem ones_digit_of_8_to_47 : 8^47 % 10 = 2 := by sorry

end NUMINAMATH_CALUDE_ones_digit_of_8_to_47_l2219_221948


namespace NUMINAMATH_CALUDE_max_product_sum_2000_l2219_221956

theorem max_product_sum_2000 : 
  ∃ (x : ℤ), ∀ (y : ℤ), y * (2000 - y) ≤ x * (2000 - x) ∧ x * (2000 - x) = 1000000 :=
by sorry

end NUMINAMATH_CALUDE_max_product_sum_2000_l2219_221956


namespace NUMINAMATH_CALUDE_welcoming_and_planning_committees_l2219_221991

theorem welcoming_and_planning_committees 
  (n : ℕ) -- Number of students
  (h1 : Nat.choose n 2 = 10) -- There are 10 ways to choose 2 from n
  : Nat.choose n 3 = 10 := by
  sorry

end NUMINAMATH_CALUDE_welcoming_and_planning_committees_l2219_221991


namespace NUMINAMATH_CALUDE_salary_decrease_equivalence_l2219_221982

-- Define the pay cuts
def first_cut : ℝ := 0.05
def second_cut : ℝ := 0.10
def third_cut : ℝ := 0.15

-- Define the function to calculate the equivalent single percentage decrease
def equivalent_decrease (c1 c2 c3 : ℝ) : ℝ :=
  (1 - (1 - c1) * (1 - c2) * (1 - c3)) * 100

-- State the theorem
theorem salary_decrease_equivalence :
  equivalent_decrease first_cut second_cut third_cut = 27.325 := by
  sorry

end NUMINAMATH_CALUDE_salary_decrease_equivalence_l2219_221982


namespace NUMINAMATH_CALUDE_log_inequality_l2219_221995

/-- The function f as defined in the problem -/
def f (m : ℝ) (x : ℝ) : ℝ := x - |x + 2| - |x - 3| - m

/-- The theorem statement -/
theorem log_inequality (m : ℝ) 
  (h1 : ∀ x : ℝ, (1 / m) - 4 ≥ f m x) 
  (h2 : m > 0) : 
  Real.log (m + 2) / Real.log (m + 1) > Real.log (m + 3) / Real.log (m + 2) := by
  sorry

end NUMINAMATH_CALUDE_log_inequality_l2219_221995


namespace NUMINAMATH_CALUDE_valid_grid_has_twelve_red_cells_l2219_221990

/-- Represents the color of a cell -/
inductive Color
| Red
| Blue

/-- Represents a 4x4 grid of colored cells -/
def Grid := Fin 4 → Fin 4 → Color

/-- Returns the list of neighboring cells for a given position -/
def neighbors (i j : Fin 4) : List (Fin 4 × Fin 4) :=
  sorry

/-- Counts the number of neighbors of a given color -/
def countNeighbors (g : Grid) (i j : Fin 4) (c : Color) : Nat :=
  sorry

/-- Checks if the grid satisfies the conditions for red cells -/
def validRedCells (g : Grid) : Prop :=
  ∀ i j, g i j = Color.Red →
    countNeighbors g i j Color.Red > countNeighbors g i j Color.Blue

/-- Checks if the grid satisfies the conditions for blue cells -/
def validBlueCells (g : Grid) : Prop :=
  ∀ i j, g i j = Color.Blue →
    countNeighbors g i j Color.Red = countNeighbors g i j Color.Blue

/-- Counts the total number of red cells in the grid -/
def countRedCells (g : Grid) : Nat :=
  sorry

/-- The main theorem stating that a valid grid has exactly 12 red cells -/
theorem valid_grid_has_twelve_red_cells (g : Grid)
  (h_red : validRedCells g)
  (h_blue : validBlueCells g)
  (h_both_colors : ∃ i j, g i j = Color.Red ∧ ∃ i' j', g i' j' = Color.Blue) :
  countRedCells g = 12 :=
sorry

end NUMINAMATH_CALUDE_valid_grid_has_twelve_red_cells_l2219_221990


namespace NUMINAMATH_CALUDE_strip_sum_unique_l2219_221903

def strip_sum (T : ℕ) : List ℕ → Prop
  | [a₁, a₂, a₃, a₄, a₅, a₆, a₇, a₈] =>
    a₁ = 2021 ∧ a₈ = 2021 ∧
    (∀ i ∈ [1, 2, 3, 4, 5, 6, 7], 
      (List.get! [a₁, a₂, a₃, a₄, a₅, a₆, a₇, a₈] i + 
       List.get! [a₁, a₂, a₃, a₄, a₅, a₆, a₇, a₈] (i+1) = T ∨
       List.get! [a₁, a₂, a₃, a₄, a₅, a₆, a₇, a₈] i + 
       List.get! [a₁, a₂, a₃, a₄, a₅, a₆, a₇, a₈] (i+1) = T+1)) ∧
    a₇ + a₈ = T
  | _ => False

theorem strip_sum_unique : 
  ∃! T, ∃ (l : List ℕ), strip_sum T l ∧ T = 4045 := by sorry

end NUMINAMATH_CALUDE_strip_sum_unique_l2219_221903


namespace NUMINAMATH_CALUDE_complex_equation_solution_l2219_221910

theorem complex_equation_solution (a b : ℝ) (i : ℂ) :
  i * i = -1 →
  (a + i) * i = b + i →
  a = 1 ∧ b = -1 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l2219_221910


namespace NUMINAMATH_CALUDE_combination_count_l2219_221930

theorem combination_count (n k m : ℕ) :
  (∃ (s : Finset (Finset ℕ)),
    (∀ t ∈ s, t.card = k ∧
      (∀ j ∈ t, 1 ≤ j ∧ j ≤ n) ∧
      (∀ (i j : ℕ), i ∈ t → j ∈ t → i < j → m ≤ j - i) ∧
      (∀ (i j : ℕ), i ∈ t → j ∈ t → i ≠ j → i < j)) ∧
    s.card = Nat.choose (n - (k - 1) * (m - 1)) k) :=
by sorry

end NUMINAMATH_CALUDE_combination_count_l2219_221930


namespace NUMINAMATH_CALUDE_negative_three_squared_plus_negative_two_cubed_l2219_221905

theorem negative_three_squared_plus_negative_two_cubed : -3^2 + (-2)^3 = -17 := by
  sorry

end NUMINAMATH_CALUDE_negative_three_squared_plus_negative_two_cubed_l2219_221905


namespace NUMINAMATH_CALUDE_crossing_point_distance_less_than_one_l2219_221969

/-- Represents a ladder in the ditch -/
structure Ladder :=
  (length : ℝ)
  (base_point : ℝ × ℝ)
  (top_point : ℝ × ℝ)

/-- Represents the ditch setup -/
structure DitchSetup :=
  (width : ℝ)
  (height : ℝ)
  (ladder1 : Ladder)
  (ladder2 : Ladder)

/-- The crossing point of two ladders -/
def crossing_point (l1 l2 : Ladder) : ℝ × ℝ := sorry

/-- Distance from a point to the left wall of the ditch -/
def distance_to_left_wall (p : ℝ × ℝ) : ℝ := p.1

/-- Main theorem: The crossing point is less than 1m from the left wall -/
theorem crossing_point_distance_less_than_one (setup : DitchSetup) :
  setup.ladder1.length = 3 →
  setup.ladder2.length = 2 →
  setup.ladder1.base_point.1 = 0 →
  setup.ladder2.base_point.1 = setup.width →
  setup.ladder1.top_point.2 = setup.height →
  setup.ladder2.top_point.2 = setup.height →
  distance_to_left_wall (crossing_point setup.ladder1 setup.ladder2) < 1 := by
  sorry

end NUMINAMATH_CALUDE_crossing_point_distance_less_than_one_l2219_221969


namespace NUMINAMATH_CALUDE_circle_tangent_sum_radii_l2219_221915

theorem circle_tangent_sum_radii : ∀ r : ℝ,
  (r > 0) →
  ((r - 5)^2 + r^2 = (r + 2)^2) →
  (∃ r₁ r₂ : ℝ, (r = r₁ ∨ r = r₂) ∧ r₁ + r₂ = 14) :=
by sorry

end NUMINAMATH_CALUDE_circle_tangent_sum_radii_l2219_221915


namespace NUMINAMATH_CALUDE_linear_function_max_value_l2219_221980

theorem linear_function_max_value (m : ℝ) :
  (∀ x : ℝ, -1 ≤ x ∧ x ≤ 3 → m * x - 2 * m ≤ 6) ∧
  (∃ x : ℝ, -1 ≤ x ∧ x ≤ 3 ∧ m * x - 2 * m = 6) →
  m = -2 ∨ m = 6 := by
sorry

end NUMINAMATH_CALUDE_linear_function_max_value_l2219_221980


namespace NUMINAMATH_CALUDE_num_triangles_eq_choose_l2219_221976

/-- The number of triangles formed by n lines in general position on a plane -/
def num_triangles (n : ℕ) : ℕ :=
  Nat.choose n 3

/-- 
Theorem: The number of triangles formed by n lines in general position on a plane
is equal to (n choose 3).
-/
theorem num_triangles_eq_choose (n : ℕ) : 
  num_triangles n = Nat.choose n 3 := by
  sorry

end NUMINAMATH_CALUDE_num_triangles_eq_choose_l2219_221976


namespace NUMINAMATH_CALUDE_father_son_speed_ratio_l2219_221908

/-- 
Given a hallway of length 16 meters where a father and son start walking from opposite ends 
at the same time and meet at a point 12 meters from the father's end, 
the ratio of the father's walking speed to the son's walking speed is 3:1.
-/
theorem father_son_speed_ratio 
  (hallway_length : ℝ) 
  (meeting_point : ℝ) 
  (father_speed : ℝ) 
  (son_speed : ℝ) 
  (h1 : hallway_length = 16)
  (h2 : meeting_point = 12)
  (h3 : father_speed > 0)
  (h4 : son_speed > 0)
  (h5 : meeting_point / father_speed = (hallway_length - meeting_point) / son_speed) :
  father_speed / son_speed = 3 := by
  sorry

end NUMINAMATH_CALUDE_father_son_speed_ratio_l2219_221908


namespace NUMINAMATH_CALUDE_exists_special_subset_l2219_221967

theorem exists_special_subset : ∃ (X : Set ℤ), ∀ (n : ℤ), ∃! (p : ℤ × ℤ), p.1 ∈ X ∧ p.2 ∈ X ∧ p.1 + 2 * p.2 = n := by
  sorry

end NUMINAMATH_CALUDE_exists_special_subset_l2219_221967


namespace NUMINAMATH_CALUDE_reflection_line_equation_l2219_221947

/-- The line of reflection for a triangle given its original and reflected coordinates -/
def line_of_reflection (D E F D' E' F' : ℝ × ℝ) : ℝ → Prop :=
  fun x ↦ x = -7

/-- Theorem: The equation of the line of reflection for the given triangle and its image -/
theorem reflection_line_equation :
  let D := (-3, 2)
  let E := (1, 4)
  let F := (-5, -1)
  let D' := (-11, 2)
  let E' := (-9, 4)
  let F' := (-15, -1)
  line_of_reflection D E F D' E' F' = fun x ↦ x = -7 := by
  sorry

end NUMINAMATH_CALUDE_reflection_line_equation_l2219_221947


namespace NUMINAMATH_CALUDE_group_size_l2219_221966

theorem group_size (B S B_intersect_S : ℕ) 
  (hB : B = 50)
  (hS : S = 70)
  (hIntersect : B_intersect_S = 20) :
  B + S - B_intersect_S = 100 := by
  sorry

end NUMINAMATH_CALUDE_group_size_l2219_221966


namespace NUMINAMATH_CALUDE_problem_statement_l2219_221973

def additive_inverse (a b : ℚ) : Prop := a + b = 0

def multiplicative_inverse (b c : ℚ) : Prop := b * c = 1

def cubic_identity (m : ℚ) : Prop := m^3 = m

theorem problem_statement 
  (a b c m : ℚ) 
  (h1 : additive_inverse a b) 
  (h2 : multiplicative_inverse b c) 
  (h3 : cubic_identity m) :
  (∃ S : ℚ, 
    (2*a + 2*b) / (m + 2) + a*c = -1 ∧
    (a > 1 → m < 0 → 
      S = |2*a - 3*b| - 2*|b - m| - |b + 1/2| →
      4*(2*a - S) + 2*(2*a - S) - (2*a - S) = -25/2) ∧
    (m ≠ 0 → ∃ (max_val : ℚ), 
      (∀ (x : ℚ), |x + m| - |x - m| ≤ max_val) ∧
      (∃ (x : ℚ), |x + m| - |x - m| = max_val) ∧
      max_val = 2)) :=
by sorry

end NUMINAMATH_CALUDE_problem_statement_l2219_221973


namespace NUMINAMATH_CALUDE_complex_power_sum_l2219_221935

theorem complex_power_sum (i : ℂ) (h : i^2 = -1) :
  i^14 + i^19 + i^24 + i^29 + 3*i^34 + 2*i^39 = -3 - 2*i := by
  sorry

end NUMINAMATH_CALUDE_complex_power_sum_l2219_221935


namespace NUMINAMATH_CALUDE_division_multiplication_result_l2219_221902

theorem division_multiplication_result : 
  let number := 5
  let intermediate := number / 6
  let result := intermediate * 12
  result = 10 := by sorry

end NUMINAMATH_CALUDE_division_multiplication_result_l2219_221902


namespace NUMINAMATH_CALUDE_bills_piggy_bank_l2219_221922

theorem bills_piggy_bank (x : ℕ) : 
  (∀ week : ℕ, week ≥ 1 ∧ week ≤ 8 → x + 2 * week = 3 * x) →
  x + 2 * 8 = 24 :=
by sorry

end NUMINAMATH_CALUDE_bills_piggy_bank_l2219_221922


namespace NUMINAMATH_CALUDE_four_n_plus_two_not_in_M_l2219_221917

/-- The set M of differences of squares of integers -/
def M : Set ℤ := {a | ∃ x y : ℤ, a = x^2 - y^2}

/-- Theorem stating that 4n+2 is not in M for any integer n -/
theorem four_n_plus_two_not_in_M (n : ℤ) : (4*n + 2) ∉ M := by
  sorry

end NUMINAMATH_CALUDE_four_n_plus_two_not_in_M_l2219_221917


namespace NUMINAMATH_CALUDE_fifth_element_row_21_l2219_221961

/-- Pascal's triangle element -/
def pascal_triangle_element (n : ℕ) (k : ℕ) : ℕ := Nat.choose n (k - 1)

/-- The fifth element in Row 21 of Pascal's triangle is 1995 -/
theorem fifth_element_row_21 : pascal_triangle_element 21 5 = 1995 := by
  sorry

end NUMINAMATH_CALUDE_fifth_element_row_21_l2219_221961


namespace NUMINAMATH_CALUDE_white_ball_count_l2219_221943

/-- Given a bag of 100 glass balls with red, black, and white colors,
    prove that if the frequency of drawing red balls is 15% and black balls is 40%,
    then the number of white balls is 45. -/
theorem white_ball_count (total : ℕ) (red_freq black_freq : ℚ) :
  total = 100 →
  red_freq = 15 / 100 →
  black_freq = 40 / 100 →
  ∃ (white_count : ℕ), white_count = 45 ∧ white_count = total * (1 - red_freq - black_freq) :=
sorry

end NUMINAMATH_CALUDE_white_ball_count_l2219_221943


namespace NUMINAMATH_CALUDE_consecutive_draw_probability_l2219_221989

/-- The probability of drawing one red marble and then one blue marble consecutively from a bag of marbles. -/
theorem consecutive_draw_probability
  (red : ℕ) (blue : ℕ) (green : ℕ)
  (h_red : red = 5)
  (h_blue : blue = 4)
  (h_green : green = 6)
  : (red : ℚ) / (red + blue + green) * (blue : ℚ) / (red + blue + green - 1) = 2 / 21 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_draw_probability_l2219_221989


namespace NUMINAMATH_CALUDE_smallest_integer_with_remainders_l2219_221997

theorem smallest_integer_with_remainders : ∃ n : ℕ,
  n > 0 ∧
  n % 10 = 9 ∧
  n % 9 = 8 ∧
  n % 8 = 7 ∧
  n % 7 = 6 ∧
  n % 6 = 5 ∧
  n % 5 = 4 ∧
  n % 4 = 3 ∧
  n % 3 = 2 ∧
  n % 2 = 1 ∧
  (∀ m : ℕ, m > 0 →
    m % 10 = 9 →
    m % 9 = 8 →
    m % 8 = 7 →
    m % 7 = 6 →
    m % 6 = 5 →
    m % 5 = 4 →
    m % 4 = 3 →
    m % 3 = 2 →
    m % 2 = 1 →
    n ≤ m) ∧
  n = 2519 :=
by sorry

end NUMINAMATH_CALUDE_smallest_integer_with_remainders_l2219_221997


namespace NUMINAMATH_CALUDE_power_of_power_l2219_221987

theorem power_of_power (a : ℝ) : (a^2)^3 = a^6 := by
  sorry

end NUMINAMATH_CALUDE_power_of_power_l2219_221987


namespace NUMINAMATH_CALUDE_square_root_extraction_scheme_l2219_221970

theorem square_root_extraction_scheme (n : Nat) (root : Nat) : 
  n = 418089 ∧ root = 647 → root * root = n := by
  sorry

end NUMINAMATH_CALUDE_square_root_extraction_scheme_l2219_221970


namespace NUMINAMATH_CALUDE_down_payment_calculation_l2219_221911

/-- Given a loan with the following conditions:
  * The loan has 0% interest
  * The loan is to be paid back in 5 years
  * Monthly payments are $600.00
  * The total loan amount (including down payment) is $46,000
  This theorem proves that the down payment is $10,000 -/
theorem down_payment_calculation (loan_amount : ℝ) (years : ℕ) (monthly_payment : ℝ) :
  loan_amount = 46000 ∧ 
  years = 5 ∧ 
  monthly_payment = 600 →
  loan_amount - (years * 12 : ℝ) * monthly_payment = 10000 :=
by sorry

end NUMINAMATH_CALUDE_down_payment_calculation_l2219_221911


namespace NUMINAMATH_CALUDE_burger_problem_l2219_221985

theorem burger_problem (total_burgers : ℕ) (total_cost : ℚ) (single_cost : ℚ) (double_cost : ℚ) 
  (h1 : total_burgers = 50)
  (h2 : total_cost = 64.5)
  (h3 : single_cost = 1)
  (h4 : double_cost = 1.5) :
  ∃ (single_count double_count : ℕ),
    single_count + double_count = total_burgers ∧
    single_cost * single_count + double_cost * double_count = total_cost ∧
    double_count = 29 := by
  sorry

end NUMINAMATH_CALUDE_burger_problem_l2219_221985


namespace NUMINAMATH_CALUDE_remaining_money_l2219_221965

def octal_to_decimal (n : ℕ) : ℕ := sorry

def john_savings : ℕ := 5273
def rental_car_cost : ℕ := 1500

theorem remaining_money :
  octal_to_decimal john_savings - rental_car_cost = 1247 := by sorry

end NUMINAMATH_CALUDE_remaining_money_l2219_221965


namespace NUMINAMATH_CALUDE_birds_flying_away_l2219_221926

theorem birds_flying_away (total : ℕ) (remaining : ℕ) : 
  total = 60 → remaining = 8 → 
  ∃ (F : ℚ), F = 1/3 ∧ 
  (1 - 2/3) * (1 - 2/5) * (1 - F) * total = remaining :=
by sorry

end NUMINAMATH_CALUDE_birds_flying_away_l2219_221926


namespace NUMINAMATH_CALUDE_complex_magnitude_equals_five_l2219_221901

theorem complex_magnitude_equals_five (t : ℝ) :
  Complex.abs (1 + 2 * t * Complex.I) = 5 ↔ t = Real.sqrt 6 ∨ t = -Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_equals_five_l2219_221901


namespace NUMINAMATH_CALUDE_bamboo_break_height_l2219_221960

theorem bamboo_break_height (total_height : ℝ) (fall_distance : ℝ) (break_height : ℝ) : 
  total_height = 9 → 
  fall_distance = 3 → 
  break_height^2 + fall_distance^2 = (total_height - break_height)^2 →
  break_height = 4 := by
sorry

end NUMINAMATH_CALUDE_bamboo_break_height_l2219_221960


namespace NUMINAMATH_CALUDE_marble_problem_l2219_221934

/-- The total number of marbles given the conditions of the problem -/
def total_marbles : ℕ := 36

/-- Mario's share of marbles before Manny gives away 2 marbles -/
def mario_marbles : ℕ := 16

/-- Manny's share of marbles before giving away 2 marbles -/
def manny_marbles : ℕ := 20

/-- The ratio of Mario's marbles to Manny's marbles -/
def marble_ratio : Rat := 4 / 5

theorem marble_problem :
  (mario_marbles : ℚ) / (manny_marbles : ℚ) = marble_ratio ∧
  manny_marbles - 2 = 18 ∧
  total_marbles = mario_marbles + manny_marbles :=
by sorry

end NUMINAMATH_CALUDE_marble_problem_l2219_221934


namespace NUMINAMATH_CALUDE_problem_solution_l2219_221904

theorem problem_solution (x y : ℝ) (h1 : x + y = 8) (h2 : x^2 - y^2 = 24) : x - y = 3 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2219_221904


namespace NUMINAMATH_CALUDE_problem_solution_l2219_221939

theorem problem_solution (a b c d : ℕ+) 
  (h1 : a < b) (h2 : b < c) (h3 : c < d)
  (h4 : a * b + b * c + a * c = a * b * c)
  (h5 : a * b * c = d) : d = 36 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2219_221939
