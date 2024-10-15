import Mathlib

namespace NUMINAMATH_CALUDE_unique_point_in_S_l429_42921

def S : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 > 0 ∧ p.2 > 0 ∧ Real.log (p.1^3 + (1/3) * p.2^3 + 1/9) = Real.log p.1 + Real.log p.2}

theorem unique_point_in_S : ∃! p : ℝ × ℝ, p ∈ S := by
  sorry

end NUMINAMATH_CALUDE_unique_point_in_S_l429_42921


namespace NUMINAMATH_CALUDE_m_range_l429_42950

theorem m_range (m : ℝ) : 
  (∃ x y : ℝ, x < 0 ∧ y < 0 ∧ x ≠ y ∧ x^2 + m*x + 1 = 0 ∧ y^2 + m*y + 1 = 0) ∨ 
  (∀ x : ℝ, x^2 + (m-2)*x + 1 ≠ 0) ∧
  ¬((∃ x y : ℝ, x < 0 ∧ y < 0 ∧ x ≠ y ∧ x^2 + m*x + 1 = 0 ∧ y^2 + m*y + 1 = 0) ∧ 
    (∀ x : ℝ, x^2 + (m-2)*x + 1 ≠ 0)) →
  m ∈ Set.Ioo 0 2 ∪ Set.Ici 4 :=
by sorry

end NUMINAMATH_CALUDE_m_range_l429_42950


namespace NUMINAMATH_CALUDE_square_of_complex_is_real_implies_m_is_plus_minus_one_l429_42923

theorem square_of_complex_is_real_implies_m_is_plus_minus_one (m : ℝ) :
  (∃ (r : ℝ), (m + Complex.I)^2 = r) → (m = 1 ∨ m = -1) := by
  sorry

end NUMINAMATH_CALUDE_square_of_complex_is_real_implies_m_is_plus_minus_one_l429_42923


namespace NUMINAMATH_CALUDE_first_job_wages_proof_l429_42974

/-- Calculates the amount received from the first job given total wages and second job details. -/
def first_job_wages (total_wages : ℕ) (second_job_hours : ℕ) (second_job_rate : ℕ) : ℕ :=
  total_wages - second_job_hours * second_job_rate

/-- Proves that given the specified conditions, the amount received from the first job is $52. -/
theorem first_job_wages_proof :
  first_job_wages 160 12 9 = 52 := by
  sorry

end NUMINAMATH_CALUDE_first_job_wages_proof_l429_42974


namespace NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l429_42915

theorem geometric_sequence_common_ratio 
  (a : ℕ → ℝ) 
  (h_geometric : ∀ n, a (n + 1) = a n * (a 2 / a 1)) 
  (h_a3 : a 3 = 2) 
  (h_a7 : a 7 = 32) : 
  (a 2 / a 1 = 2) ∨ (a 2 / a 1 = -2) :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l429_42915


namespace NUMINAMATH_CALUDE_lines_parallel_to_same_line_are_parallel_l429_42933

-- Define a type for lines
def Line := Type

-- Define a relation for parallel lines
def Parallel (l1 l2 : Line) : Prop := sorry

-- Theorem statement
theorem lines_parallel_to_same_line_are_parallel 
  (l1 l2 l3 : Line) : 
  Parallel l1 l3 → Parallel l2 l3 → Parallel l1 l2 := by sorry

end NUMINAMATH_CALUDE_lines_parallel_to_same_line_are_parallel_l429_42933


namespace NUMINAMATH_CALUDE_min_l_trominos_count_l429_42994

/-- Represents a tile type -/
inductive TileType
| LTromino
| STetromino

/-- Represents the grid -/
def Grid := Fin 2020 × Fin 2021

/-- A tiling is a function that assigns a tile type to each grid position -/
def Tiling := Grid → Option TileType

/-- Checks if a tiling is valid (covers the entire grid without overlaps) -/
def is_valid_tiling (t : Tiling) : Prop := sorry

/-- Counts the number of L-Trominos in a tiling -/
def count_l_trominos (t : Tiling) : Nat := sorry

/-- Theorem: The minimum number of L-Trominos in a valid tiling is 1010 -/
theorem min_l_trominos_count :
  ∃ (t : Tiling), is_valid_tiling t ∧
    ∀ (t' : Tiling), is_valid_tiling t' →
      count_l_trominos t ≤ count_l_trominos t' ∧
      count_l_trominos t = 1010 :=
sorry

end NUMINAMATH_CALUDE_min_l_trominos_count_l429_42994


namespace NUMINAMATH_CALUDE_daycare_toddlers_l429_42927

/-- Given a day care center with toddlers and infants, prove that under certain conditions, 
    the number of toddlers is 42. -/
theorem daycare_toddlers (T I : ℕ) : 
  T / I = 7 / 3 →  -- Initial ratio of toddlers to infants
  T / (I + 12) = 7 / 5 →  -- New ratio after 12 infants join
  T = 42 := by
  sorry

end NUMINAMATH_CALUDE_daycare_toddlers_l429_42927


namespace NUMINAMATH_CALUDE_brownie_pieces_count_l429_42978

/-- Represents the dimensions of a rectangular object -/
structure Dimensions where
  length : ℕ
  width : ℕ

/-- Calculates the area of a rectangular object given its dimensions -/
def area (d : Dimensions) : ℕ := d.length * d.width

/-- Represents a pan of brownies -/
structure BrowniePan where
  panDimensions : Dimensions
  pieceDimensions : Dimensions

/-- Calculates the number of brownie pieces that can be cut from a pan -/
def numberOfPieces (pan : BrowniePan) : ℕ :=
  (area pan.panDimensions) / (area pan.pieceDimensions)

theorem brownie_pieces_count :
  let pan : BrowniePan := {
    panDimensions := { length := 24, width := 15 },
    pieceDimensions := { length := 3, width := 2 }
  }
  numberOfPieces pan = 60 := by sorry

end NUMINAMATH_CALUDE_brownie_pieces_count_l429_42978


namespace NUMINAMATH_CALUDE_quadratic_two_distinct_roots_l429_42956

/-- The quadratic equation (a+1)x^2 - 4x + 1 = 0 has two distinct real roots if and only if a < 3 and a ≠ -1 -/
theorem quadratic_two_distinct_roots (a : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ (a + 1) * x^2 - 4 * x + 1 = 0 ∧ (a + 1) * y^2 - 4 * y + 1 = 0) ↔ 
  (a < 3 ∧ a ≠ -1) := by
sorry

end NUMINAMATH_CALUDE_quadratic_two_distinct_roots_l429_42956


namespace NUMINAMATH_CALUDE_f_is_even_l429_42944

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f x = f (-x)

def f (x : ℝ) : ℝ := x^4

theorem f_is_even : is_even_function f := by
  sorry

end NUMINAMATH_CALUDE_f_is_even_l429_42944


namespace NUMINAMATH_CALUDE_min_value_expression_min_value_attainable_l429_42979

theorem min_value_expression (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  4 * x^4 + 16 * y^4 + 36 * z^4 + 9 / (x * y * z) ≥ 24 :=
by
  sorry

theorem min_value_attainable :
  ∃ x y z : ℝ, x > 0 ∧ y > 0 ∧ z > 0 ∧
  4 * x^4 + 16 * y^4 + 36 * z^4 + 9 / (x * y * z) = 24 :=
by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_min_value_attainable_l429_42979


namespace NUMINAMATH_CALUDE_binary_linear_equation_problem_l429_42943

theorem binary_linear_equation_problem (m n : ℤ) : 
  (3 * m - 2 * n = -2) → 
  (3 * (m + 405) - 2 * (n - 405) = 2023) := by
  sorry

end NUMINAMATH_CALUDE_binary_linear_equation_problem_l429_42943


namespace NUMINAMATH_CALUDE_ship_distance_theorem_l429_42989

/-- Represents the ship's position relative to Island X -/
structure ShipPosition where
  angle : ℝ  -- angle in radians for circular motion
  distance : ℝ -- distance from Island X

/-- Represents the ship's path -/
inductive ShipPath
  | Circle (t : ℝ) -- t represents time spent on circular path
  | StraightLine (t : ℝ) -- t represents time spent on straight line

/-- Function to calculate the ship's distance from Island X -/
def shipDistance (r : ℝ) (path : ShipPath) : ℝ :=
  match path with
  | ShipPath.Circle _ => r
  | ShipPath.StraightLine t => r + t

theorem ship_distance_theorem (r : ℝ) (h : r > 0) :
  ∃ (t₁ t₂ : ℝ), t₁ > 0 ∧ t₂ > 0 ∧
    (∀ t, 0 ≤ t ∧ t ≤ t₁ → shipDistance r (ShipPath.Circle t) = r) ∧
    (∀ t, t > t₁ ∧ t ≤ t₁ + t₂ → shipDistance r (ShipPath.StraightLine (t - t₁)) > r ∧
      (shipDistance r (ShipPath.StraightLine (t - t₁)) - r) = t - t₁) :=
  sorry

end NUMINAMATH_CALUDE_ship_distance_theorem_l429_42989


namespace NUMINAMATH_CALUDE_greatest_divisible_by_seven_ninety_five_six_five_nine_is_valid_l429_42919

def is_valid_number (n : ℕ) : Prop :=
  10000 ≤ n ∧ n < 100000 ∧
  ∃ (a b c : ℕ),
    a < 10 ∧ b < 10 ∧ c < 10 ∧
    a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
    n = a * 10000 + b * 1000 + c * 100 + b * 10 + a

theorem greatest_divisible_by_seven :
  ∀ n : ℕ, is_valid_number n ∧ n % 7 = 0 → n ≤ 95659 :=
by sorry

theorem ninety_five_six_five_nine_is_valid :
  is_valid_number 95659 ∧ 95659 % 7 = 0 :=
by sorry

end NUMINAMATH_CALUDE_greatest_divisible_by_seven_ninety_five_six_five_nine_is_valid_l429_42919


namespace NUMINAMATH_CALUDE_meaningful_expression_range_l429_42966

theorem meaningful_expression_range (x : ℝ) :
  (∃ y : ℝ, y = x / Real.sqrt (x + 2)) ↔ x > -2 := by sorry

end NUMINAMATH_CALUDE_meaningful_expression_range_l429_42966


namespace NUMINAMATH_CALUDE_arithmetic_geometric_mean_inequality_l429_42938

theorem arithmetic_geometric_mean_inequality (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  a * b ≤ ((a + b) / 2) * Real.sqrt (a * b) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_mean_inequality_l429_42938


namespace NUMINAMATH_CALUDE_square_triangle_area_equality_l429_42953

theorem square_triangle_area_equality (x : ℝ) (h : x > 0) :
  let square_area := x^2
  let triangle_base := x
  let triangle_altitude := 2 * x
  let triangle_area := (1 / 2) * triangle_base * triangle_altitude
  square_area = triangle_area := by
  sorry

end NUMINAMATH_CALUDE_square_triangle_area_equality_l429_42953


namespace NUMINAMATH_CALUDE_tan_alpha_3_implies_fraction_eq_5_div_7_l429_42987

theorem tan_alpha_3_implies_fraction_eq_5_div_7 (α : Real) (h : Real.tan α = 3) :
  (4 * Real.sin α - 2 * Real.cos α) / (5 * Real.cos α + 3 * Real.sin α) = 5 / 7 := by
  sorry

end NUMINAMATH_CALUDE_tan_alpha_3_implies_fraction_eq_5_div_7_l429_42987


namespace NUMINAMATH_CALUDE_inverse_of_five_mod_221_l429_42995

theorem inverse_of_five_mod_221 : ∃! x : ℕ, x ∈ Finset.range 221 ∧ (5 * x) % 221 = 1 :=
by
  use 177
  sorry

end NUMINAMATH_CALUDE_inverse_of_five_mod_221_l429_42995


namespace NUMINAMATH_CALUDE_town_x_employment_l429_42906

structure TownPopulation where
  total_employed : Real
  employed_20_35 : Real
  employed_36_50 : Real
  employed_51_65 : Real
  employed_males : Real
  males_high_school : Real
  males_college : Real
  males_postgrad : Real

def employed_females (pop : TownPopulation) : Real :=
  pop.total_employed - pop.employed_males

theorem town_x_employment (pop : TownPopulation)
  (h1 : pop.total_employed = 0.96)
  (h2 : pop.employed_20_35 = 0.40 * pop.total_employed)
  (h3 : pop.employed_36_50 = 0.50 * pop.total_employed)
  (h4 : pop.employed_51_65 = 0.10 * pop.total_employed)
  (h5 : pop.employed_males = 0.24)
  (h6 : pop.males_high_school = 0.45 * pop.employed_males)
  (h7 : pop.males_college = 0.35 * pop.employed_males)
  (h8 : pop.males_postgrad = 0.20 * pop.employed_males) :
  let females := employed_females pop
  ∃ (f_20_35 f_36_50 f_51_65 f_high_school f_college f_postgrad : Real),
    f_20_35 = 0.288 ∧
    f_36_50 = 0.36 ∧
    f_51_65 = 0.072 ∧
    f_high_school = 0.324 ∧
    f_college = 0.252 ∧
    f_postgrad = 0.144 ∧
    f_20_35 = 0.40 * females ∧
    f_36_50 = 0.50 * females ∧
    f_51_65 = 0.10 * females ∧
    f_high_school = 0.45 * females ∧
    f_college = 0.35 * females ∧
    f_postgrad = 0.20 * females :=
by sorry

end NUMINAMATH_CALUDE_town_x_employment_l429_42906


namespace NUMINAMATH_CALUDE_expression_value_l429_42999

theorem expression_value (a : ℝ) (h : a ≠ 0) : (20 * a^5) * (8 * a^4) * (1 / (4 * a^3)^3) = 2.5 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l429_42999


namespace NUMINAMATH_CALUDE_weight_of_b_l429_42996

/-- Given the average weights of different combinations of people a, b, c, and d,
    prove that the weight of b is 31 kg. -/
theorem weight_of_b (a b c d : ℝ) : 
  (a + b + c + d) / 4 = 48 →
  (a + b + c) / 3 = 45 →
  (a + b) / 2 = 40 →
  (b + c) / 2 = 43 →
  (c + d) / 2 = 46 →
  b = 31 := by
  sorry

end NUMINAMATH_CALUDE_weight_of_b_l429_42996


namespace NUMINAMATH_CALUDE_quadratic_equation_solutions_l429_42908

theorem quadratic_equation_solutions :
  let equation := fun y : ℝ => 3 * y * (y - 1) = 2 * (y - 1)
  (equation (2/3) ∧ equation 1) ∧
  ∀ y : ℝ, equation y → (y = 2/3 ∨ y = 1) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_solutions_l429_42908


namespace NUMINAMATH_CALUDE_function_composition_difference_l429_42920

/-- Given functions f and g, prove that f(g(x)) - g(f(x)) = 5/2 for all x. -/
theorem function_composition_difference (x : ℝ) : 
  let f : ℝ → ℝ := λ x ↦ 5 * x - 3
  let g : ℝ → ℝ := λ x ↦ x / 2 + 1
  f (g x) - g (f x) = 5 / 2 := by
  sorry

end NUMINAMATH_CALUDE_function_composition_difference_l429_42920


namespace NUMINAMATH_CALUDE_square_area_doubling_l429_42909

theorem square_area_doubling (s : ℝ) (h : s > 0) :
  (2 * s) ^ 2 = 4 * s ^ 2 := by sorry

end NUMINAMATH_CALUDE_square_area_doubling_l429_42909


namespace NUMINAMATH_CALUDE_remainder_four_eleven_mod_five_l429_42922

theorem remainder_four_eleven_mod_five : 4^11 % 5 = 4 := by
  sorry

end NUMINAMATH_CALUDE_remainder_four_eleven_mod_five_l429_42922


namespace NUMINAMATH_CALUDE_share_distribution_l429_42901

theorem share_distribution (total : ℝ) (maya annie saiji : ℝ) : 
  total = 900 →
  maya = (1/2) * annie →
  annie = (1/2) * saiji →
  total = maya + annie + saiji →
  saiji = 900 * (4/7) :=
by sorry

end NUMINAMATH_CALUDE_share_distribution_l429_42901


namespace NUMINAMATH_CALUDE_coefficient_x_squared_in_expansion_l429_42962

theorem coefficient_x_squared_in_expansion :
  let n : ℕ := 6
  let a : ℤ := 1
  let b : ℤ := -3
  (Finset.sum (Finset.range (n + 1)) (fun k => (n.choose k) * a^(n - k) * b^k * (if k = 2 then 1 else 0))) = 135 :=
by sorry

end NUMINAMATH_CALUDE_coefficient_x_squared_in_expansion_l429_42962


namespace NUMINAMATH_CALUDE_arithmetic_progression_y_range_l429_42967

theorem arithmetic_progression_y_range (x y : ℝ) : 
  (∃ r : ℝ, r > 0 ∧ 
    Real.log r = Real.log 2 - Real.log (Real.sin x - 1/3) ∧ 
    Real.log (Real.sin x - 1/3) = Real.log 2 - Real.log (1 - y)) →
  (∃ y_min : ℝ, y_min = 7/9 ∧ y ≥ y_min) ∧ 
  (∀ y_max : ℝ, y < y_max) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_progression_y_range_l429_42967


namespace NUMINAMATH_CALUDE_rental_ratio_l429_42975

def comedies_rented : ℕ := 15
def action_movies_rented : ℕ := 5

theorem rental_ratio : 
  (comedies_rented : ℚ) / (action_movies_rented : ℚ) = 3 / 1 := by
  sorry

end NUMINAMATH_CALUDE_rental_ratio_l429_42975


namespace NUMINAMATH_CALUDE_inverse_variation_solution_l429_42929

/-- Inverse variation relation between three quantities -/
def inverse_variation (r s t : ℝ) : Prop :=
  ∃ (k₁ k₂ : ℝ), k₁ ≠ 0 ∧ k₂ ≠ 0 ∧ r * s = k₁ ∧ r * t = k₂

theorem inverse_variation_solution (r₁ s₁ t₁ r₂ s₂ t₂ : ℝ) :
  inverse_variation r₁ s₁ t₁ →
  inverse_variation r₂ s₂ t₂ →
  r₁ = 1500 →
  s₁ = 0.25 →
  t₁ = 0.5 →
  r₂ = 3000 →
  s₂ = 0.125 ∧ t₂ = 0.25 := by
  sorry


end NUMINAMATH_CALUDE_inverse_variation_solution_l429_42929


namespace NUMINAMATH_CALUDE_stadium_seats_count_l429_42914

/-- The number of seats in a stadium is equal to the sum of occupied and empty seats -/
theorem stadium_seats_count
  (children : ℕ)
  (adults : ℕ)
  (empty_seats : ℕ)
  (h1 : children = 52)
  (h2 : adults = 29)
  (h3 : empty_seats = 14) :
  children + adults + empty_seats = 95 := by
  sorry

#check stadium_seats_count

end NUMINAMATH_CALUDE_stadium_seats_count_l429_42914


namespace NUMINAMATH_CALUDE_equal_squares_exist_l429_42947

/-- Represents a cell in the grid -/
structure Cell where
  row : Fin 10
  col : Fin 10

/-- Represents a square in the grid -/
structure Square where
  cell : Cell
  size : ℕ

/-- The theorem to be proved -/
theorem equal_squares_exist (squares : Finset Square) 
  (h1 : squares.card = 9)
  (h2 : ∀ s ∈ squares, s.cell.row < 10 ∧ s.cell.col < 10) :
  ∃ s1 s2 : Square, s1 ∈ squares ∧ s2 ∈ squares ∧ s1 ≠ s2 ∧ s1.size = s2.size :=
sorry

end NUMINAMATH_CALUDE_equal_squares_exist_l429_42947


namespace NUMINAMATH_CALUDE_modulus_of_9_minus_40i_l429_42949

theorem modulus_of_9_minus_40i : Complex.abs (9 - 40*I) = 41 := by sorry

end NUMINAMATH_CALUDE_modulus_of_9_minus_40i_l429_42949


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l429_42981

theorem quadratic_inequality_range (a : ℝ) : 
  (∀ x : ℝ, 2 * x^2 + (a - 1) * x + 1/2 > 0) ↔ -1 < a ∧ a < 3 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l429_42981


namespace NUMINAMATH_CALUDE_cost_to_selling_price_ratio_l429_42980

/-- Given a 25% profit, prove that the ratio of cost price to selling price is 4 : 5 -/
theorem cost_to_selling_price_ratio (cost_price selling_price : ℝ) 
  (h_positive : cost_price > 0)
  (h_profit : selling_price = cost_price * (1 + 0.25)) :
  cost_price / selling_price = 4 / 5 := by
  sorry

end NUMINAMATH_CALUDE_cost_to_selling_price_ratio_l429_42980


namespace NUMINAMATH_CALUDE_sequence_theorem_l429_42985

/-- A sequence whose reciprocal forms an arithmetic sequence -/
def IsReciprocalArithmetic (a : ℕ → ℝ) : Prop :=
  ∀ n, 2 / a (n + 1) = 1 / a n + 1 / a (n + 2)

/-- The main theorem -/
theorem sequence_theorem (x : ℕ → ℝ) (a : ℕ → ℝ) 
    (h_pos : ∀ n, x n > 0)
    (h_recip_arith : IsReciprocalArithmetic a)
    (h_x1 : x 1 = 3)
    (h_sum : x 1 + x 2 + x 3 = 39)
    (h_power : ∀ n, (x n) ^ (a n) = (x (n + 1)) ^ (a (n + 1)) ∧ 
                    (x n) ^ (a n) = (x (n + 2)) ^ (a (n + 2))) : 
  ∀ n, x n = 3^n := by
  sorry

end NUMINAMATH_CALUDE_sequence_theorem_l429_42985


namespace NUMINAMATH_CALUDE_max_pencils_is_seven_l429_42924

/-- The maximum number of pencils Alice can purchase given the conditions --/
def max_pencils : ℕ :=
  let pin_cost : ℕ := 3
  let pen_cost : ℕ := 4
  let pencil_cost : ℕ := 9
  let total_budget : ℕ := 72
  let min_purchase : ℕ := pin_cost + pen_cost
  let remaining_budget : ℕ := total_budget - min_purchase
  remaining_budget / pencil_cost

/-- Theorem stating that the maximum number of pencils Alice can purchase is 7 --/
theorem max_pencils_is_seven : max_pencils = 7 := by
  sorry

#eval max_pencils -- This will evaluate to 7

end NUMINAMATH_CALUDE_max_pencils_is_seven_l429_42924


namespace NUMINAMATH_CALUDE_apples_distribution_l429_42913

/-- The number of people who received apples -/
def num_people (total_apples : ℕ) (apples_per_person : ℚ) : ℚ :=
  total_apples / apples_per_person

/-- Proof that 3 people received apples -/
theorem apples_distribution (total_apples : ℕ) (apples_per_person : ℚ) 
  (h1 : total_apples = 45)
  (h2 : apples_per_person = 15.0) : 
  num_people total_apples apples_per_person = 3 := by
  sorry


end NUMINAMATH_CALUDE_apples_distribution_l429_42913


namespace NUMINAMATH_CALUDE_drum_oil_capacity_l429_42961

theorem drum_oil_capacity (c : ℝ) (h1 : c > 0) : 
  let drum_x_capacity := c
  let drum_x_oil := (1 / 2 : ℝ) * drum_x_capacity
  let drum_y_capacity := 2 * drum_x_capacity
  let drum_y_oil := (1 / 3 : ℝ) * drum_y_capacity
  let final_oil := drum_y_oil + drum_x_oil
  final_oil / drum_y_capacity = 7 / 12
  := by sorry

end NUMINAMATH_CALUDE_drum_oil_capacity_l429_42961


namespace NUMINAMATH_CALUDE_athena_total_spent_l429_42912

def sandwich_price : ℝ := 3
def fruit_drink_price : ℝ := 2.5
def num_sandwiches : ℕ := 3
def num_fruit_drinks : ℕ := 2

theorem athena_total_spent :
  (num_sandwiches : ℝ) * sandwich_price + (num_fruit_drinks : ℝ) * fruit_drink_price = 14 := by
  sorry

end NUMINAMATH_CALUDE_athena_total_spent_l429_42912


namespace NUMINAMATH_CALUDE_a_and_b_know_own_results_a_and_b_dont_know_each_others_results_l429_42911

-- Define the possible results
inductive Result
| Excellent
| Good

-- Define the students
inductive Student
| A
| B
| C
| D

-- Define the function that assigns results to students
def result : Student → Result := sorry

-- Define the knowledge state of each student
structure Knowledge where
  knows_b : Bool
  knows_c : Bool
  knows_d : Bool

-- Define the initial knowledge state
def initial_knowledge : Student → Knowledge
| Student.A => { knows_b := false, knows_c := false, knows_d := true }
| Student.B => { knows_b := false, knows_c := true,  knows_d := false }
| Student.C => { knows_b := false, knows_c := false, knows_d := false }
| Student.D => { knows_b := true,  knows_c := true,  knows_d := false }

-- Theorem stating that A and B can know their own results
theorem a_and_b_know_own_results :
  (∃ (s₁ s₂ : Student), result s₁ = Result.Excellent ∧ result s₂ = Result.Excellent) ∧
  (∃ (s₃ s₄ : Student), result s₃ = Result.Good ∧ result s₄ = Result.Good) ∧
  (result Student.B ≠ result Student.C) ∧
  (¬ (result Student.D = Result.Excellent ∧ result Student.B = Result.Excellent ∧ result Student.C = Result.Good)) ∧
  (¬ (result Student.D = Result.Excellent ∧ result Student.B = Result.Good ∧ result Student.C = Result.Excellent)) ∧
  (¬ (result Student.D = Result.Good ∧ result Student.B = Result.Good ∧ result Student.C = Result.Excellent)) ∧
  (¬ (result Student.D = Result.Good ∧ result Student.B = Result.Excellent ∧ result Student.C = Result.Good)) →
  (∃ (f : Student → Result),
    (f Student.A = result Student.A) ∧
    (f Student.B = result Student.B) ∧
    (f Student.C ≠ result Student.C ∨ f Student.D ≠ result Student.D)) :=
sorry

-- Theorem stating that A and B cannot know each other's results
theorem a_and_b_dont_know_each_others_results :
  (∃ (s₁ s₂ : Student), result s₁ = Result.Excellent ∧ result s₂ = Result.Excellent) ∧
  (∃ (s₃ s₄ : Student), result s₃ = Result.Good ∧ result s₄ = Result.Good) ∧
  (result Student.B ≠ result Student.C) ∧
  (¬ (result Student.D = Result.Excellent ∧ result Student.B = Result.Excellent ∧ result Student.C = Result.Good)) ∧
  (¬ (result Student.D = Result.Excellent ∧ result Student.B = Result.Good ∧ result Student.C = Result.Excellent)) ∧
  (¬ (result Student.D = Result.Good ∧ result Student.B = Result.Good ∧ result Student.C = Result.Excellent)) ∧
  (¬ (result Student.D = Result.Good ∧ result Student.B = Result.Excellent ∧ result Student.C = Result.Good)) →
  ¬(∃ (f : Student → Result),
    (f Student.A = result Student.A) ∧
    (f Student.B = result Student.B) ∧
    (f Student.C = result Student.C) ∧
    (f Student.D = result Student.D)) :=
sorry

end NUMINAMATH_CALUDE_a_and_b_know_own_results_a_and_b_dont_know_each_others_results_l429_42911


namespace NUMINAMATH_CALUDE_fitted_bowling_ball_volume_l429_42997

/-- The volume of a fitted bowling ball -/
theorem fitted_bowling_ball_volume :
  let ball_diameter : ℝ := 40
  let ball_radius : ℝ := ball_diameter / 2
  let hole1_diameter : ℝ := 4
  let hole1_radius : ℝ := hole1_diameter / 2
  let hole2_diameter : ℝ := 2.5
  let hole2_radius : ℝ := hole2_diameter / 2
  let hole_depth : ℝ := 8
  let ball_volume : ℝ := (4 / 3) * π * (ball_radius ^ 3)
  let hole1_volume : ℝ := π * (hole1_radius ^ 2) * hole_depth
  let hole2_volume : ℝ := π * (hole2_radius ^ 2) * hole_depth
  ball_volume - hole1_volume - 2 * hole2_volume = 10609.67 * π :=
by sorry

end NUMINAMATH_CALUDE_fitted_bowling_ball_volume_l429_42997


namespace NUMINAMATH_CALUDE_negation_of_existence_l429_42983

theorem negation_of_existence (a : ℝ) :
  (¬ ∃ x : ℝ, x^2 - a*x + 1 < 0) ↔ (∀ x : ℝ, x^2 - a*x + 1 ≥ 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_existence_l429_42983


namespace NUMINAMATH_CALUDE_harvest_duration_l429_42959

def total_earnings : ℕ := 1216
def weekly_earnings : ℕ := 16

theorem harvest_duration :
  total_earnings / weekly_earnings = 76 :=
sorry

end NUMINAMATH_CALUDE_harvest_duration_l429_42959


namespace NUMINAMATH_CALUDE_pool_water_increase_l429_42970

theorem pool_water_increase (total_capacity : ℝ) (additional_water : ℝ) 
  (h1 : total_capacity = 1312.5)
  (h2 : additional_water = 300)
  (h3 : (0.8 : ℝ) * total_capacity = additional_water + (total_capacity - additional_water)) :
  let current_water := total_capacity - additional_water
  let new_water := current_water + additional_water
  (new_water - current_water) / current_water * 100 = 40 := by
  sorry

end NUMINAMATH_CALUDE_pool_water_increase_l429_42970


namespace NUMINAMATH_CALUDE_number_145_column_l429_42960

/-- Represents the columns in the arrangement --/
inductive Column
| A | B | C | D | E | F

/-- The function that determines the column for a given position in the sequence --/
def column_for_position (n : ℕ) : Column :=
  match n % 11 with
  | 1 => Column.A
  | 2 => Column.B
  | 3 => Column.C
  | 4 => Column.D
  | 5 => Column.E
  | 6 => Column.F
  | 7 => Column.E
  | 8 => Column.D
  | 9 => Column.C
  | 10 => Column.B
  | 0 => Column.A
  | _ => Column.A  -- This case should never occur, but Lean requires it for completeness

theorem number_145_column :
  column_for_position 143 = Column.A :=
sorry

end NUMINAMATH_CALUDE_number_145_column_l429_42960


namespace NUMINAMATH_CALUDE_smallest_b_for_inequality_l429_42957

theorem smallest_b_for_inequality (b : ℕ) : (∀ k : ℕ, 27^k > 3^24 → k ≥ b) ↔ b = 9 := by
  sorry

end NUMINAMATH_CALUDE_smallest_b_for_inequality_l429_42957


namespace NUMINAMATH_CALUDE_benedicts_house_size_l429_42940

theorem benedicts_house_size (kennedy_house : ℕ) (benedict_house : ℕ) : 
  kennedy_house = 10000 ∧ kennedy_house = 4 * benedict_house + 600 → benedict_house = 2350 := by
  sorry

end NUMINAMATH_CALUDE_benedicts_house_size_l429_42940


namespace NUMINAMATH_CALUDE_initial_amount_proof_l429_42951

/-- Proves that if an amount increases by 1/8 of itself every year and after two years
    it becomes 40500, then the initial amount was 32000. -/
theorem initial_amount_proof (A : ℚ) : 
  (A + A/8 + (A + A/8)/8 = 40500) → A = 32000 :=
by sorry

end NUMINAMATH_CALUDE_initial_amount_proof_l429_42951


namespace NUMINAMATH_CALUDE_michaels_crayons_value_l429_42958

/-- The value of crayons Michael will have after the purchase -/
def total_value (initial_packs : ℕ) (additional_packs : ℕ) (price_per_pack : ℚ) : ℚ :=
  (initial_packs + additional_packs : ℚ) * price_per_pack

/-- Proof that Michael's crayons will be worth $15 after the purchase -/
theorem michaels_crayons_value :
  total_value 4 2 (5/2) = 15 := by
  sorry

end NUMINAMATH_CALUDE_michaels_crayons_value_l429_42958


namespace NUMINAMATH_CALUDE_mandys_data_plan_charge_l429_42977

/-- The normal monthly charge for Mandy's data plan -/
def normal_charge : ℝ := 30

/-- The total amount Mandy paid for 6 months -/
def total_paid : ℝ := 175

/-- The extra fee charged in the fourth month -/
def extra_fee : ℝ := 15

theorem mandys_data_plan_charge :
  (normal_charge / 3) +  -- First month (promotional rate)
  (normal_charge + extra_fee) +  -- Fourth month (with extra fee)
  (4 * normal_charge) =  -- Other four months
  total_paid := by sorry

end NUMINAMATH_CALUDE_mandys_data_plan_charge_l429_42977


namespace NUMINAMATH_CALUDE_skateboard_price_after_discounts_l429_42941

/-- Calculates the final price of an item after two consecutive percentage discounts -/
def final_price (original_price : ℝ) (discount1 : ℝ) (discount2 : ℝ) : ℝ :=
  original_price * (1 - discount1) * (1 - discount2)

/-- Theorem: The final price of a $150 skateboard after 40% and 25% discounts is $67.50 -/
theorem skateboard_price_after_discounts :
  final_price 150 0.4 0.25 = 67.5 := by
  sorry

#eval final_price 150 0.4 0.25

end NUMINAMATH_CALUDE_skateboard_price_after_discounts_l429_42941


namespace NUMINAMATH_CALUDE_inequality_proof_l429_42971

theorem inequality_proof (x y z : ℝ) 
  (h_pos : x > 0 ∧ y > 0 ∧ z > 0) 
  (h_sum : x * y + y * z + z * x ≤ 1) : 
  (x + 1/x) * (y + 1/y) * (z + 1/z) ≥ 8 * (x + y) * (y + z) * (z + x) := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l429_42971


namespace NUMINAMATH_CALUDE_division_problem_l429_42910

theorem division_problem (n : ℕ) : n / 4 = 5 ∧ n % 4 = 3 → n = 23 := by
  sorry

end NUMINAMATH_CALUDE_division_problem_l429_42910


namespace NUMINAMATH_CALUDE_fiftieth_term_of_sequence_l429_42935

def arithmetic_sequence (a₁ : ℕ) (d : ℕ) (n : ℕ) : ℕ := a₁ + (n - 1) * d

theorem fiftieth_term_of_sequence : arithmetic_sequence 2 4 50 = 198 := by
  sorry

end NUMINAMATH_CALUDE_fiftieth_term_of_sequence_l429_42935


namespace NUMINAMATH_CALUDE_book_pages_calculation_l429_42998

/-- The number of pages Steve reads per day -/
def pages_per_day : ℕ := 100

/-- The number of days per week Steve reads -/
def reading_days_per_week : ℕ := 3

/-- The number of weeks Steve takes to read the book -/
def total_weeks : ℕ := 7

/-- The total number of pages in the book -/
def total_pages : ℕ := pages_per_day * reading_days_per_week * total_weeks

theorem book_pages_calculation :
  total_pages = 2100 :=
by sorry

end NUMINAMATH_CALUDE_book_pages_calculation_l429_42998


namespace NUMINAMATH_CALUDE_quadratic_roots_relation_l429_42954

theorem quadratic_roots_relation : 
  ∀ x₁ x₂ : ℝ, 
  (x₁^2 - 2*x₁ - 4 = 0) → 
  (x₂^2 - 2*x₂ - 4 = 0) → 
  x₁ ≠ x₂ →
  (x₁ + x₂) / (x₁ * x₂) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_relation_l429_42954


namespace NUMINAMATH_CALUDE_total_purchase_ways_l429_42942

/-- The number of oreo flavors --/
def oreo_flavors : ℕ := 5

/-- The number of milk flavors --/
def milk_flavors : ℕ := 3

/-- The total number of product types --/
def total_products : ℕ := oreo_flavors + milk_flavors

/-- The number of products they must purchase collectively --/
def purchase_count : ℕ := 3

/-- Represents the ways Alpha can choose items without repetition --/
def alpha_choices (k : ℕ) : ℕ := Nat.choose total_products k

/-- Represents the ways Beta can choose oreos with possible repetition --/
def beta_choices (k : ℕ) : ℕ :=
  if k = 0 then 1
  else if k = 1 then oreo_flavors
  else if k = 2 then Nat.choose oreo_flavors 2 + oreo_flavors
  else Nat.choose oreo_flavors 3 + oreo_flavors * (oreo_flavors - 1) + oreo_flavors

/-- The total number of ways Alpha and Beta can purchase 3 products collectively --/
def total_ways : ℕ := 
  alpha_choices 3 +
  alpha_choices 2 * beta_choices 1 +
  alpha_choices 1 * beta_choices 2 +
  beta_choices 3

theorem total_purchase_ways : total_ways = 351 := by sorry

end NUMINAMATH_CALUDE_total_purchase_ways_l429_42942


namespace NUMINAMATH_CALUDE_complement_intersection_equals_set_l429_42934

def U : Set Nat := {0, 1, 2, 3}
def M : Set Nat := {0, 1, 2}
def N : Set Nat := {1, 2, 3}

theorem complement_intersection_equals_set : (U \ (M ∩ N)) = {0, 3} := by
  sorry

end NUMINAMATH_CALUDE_complement_intersection_equals_set_l429_42934


namespace NUMINAMATH_CALUDE_min_value_exponential_sum_l429_42900

theorem min_value_exponential_sum (a b : ℝ) (h : 2 * a + b = 6) :
  ∃ (min : ℝ), min = 4 * Real.sqrt 2 ∧ ∀ (x y : ℝ), 2 * x + y = 6 → 2^x + Real.sqrt 2^y ≥ min := by
  sorry

end NUMINAMATH_CALUDE_min_value_exponential_sum_l429_42900


namespace NUMINAMATH_CALUDE_tom_total_money_l429_42991

/-- Tom's initial amount of money in dollars -/
def initial_amount : ℕ := 74

/-- Amount Tom earned from washing cars in dollars -/
def car_wash_earnings : ℕ := 86

/-- Tom's total money after washing cars -/
def total_money : ℕ := initial_amount + car_wash_earnings

theorem tom_total_money :
  total_money = 160 := by sorry

end NUMINAMATH_CALUDE_tom_total_money_l429_42991


namespace NUMINAMATH_CALUDE_smallest_n_with_1981_zeros_l429_42952

def count_trailing_zeros (n : ℕ) : ℕ :=
  (n / 5) + (n / 25) + (n / 125) + (n / 625) + (n / 3125)

theorem smallest_n_with_1981_zeros :
  ∃ (n : ℕ), count_trailing_zeros n = 1981 ∧
    ∀ (m : ℕ), m < n → count_trailing_zeros m < 1981 :=
by
  use 7935
  sorry

end NUMINAMATH_CALUDE_smallest_n_with_1981_zeros_l429_42952


namespace NUMINAMATH_CALUDE_firecracker_explosion_speed_l429_42917

/-- The speed of a fragment after an explosion, given initial conditions of a firecracker. -/
theorem firecracker_explosion_speed 
  (v₀ : ℝ)           -- Initial upward speed of firecracker
  (t : ℝ)            -- Time of explosion
  (m₁ m₂ : ℝ)        -- Masses of fragments
  (v_small : ℝ)      -- Horizontal speed of smaller fragment after explosion
  (g : ℝ)            -- Acceleration due to gravity
  (h : v₀ = 20)      -- Initial speed is 20 m/s
  (h_t : t = 3)      -- Explosion occurs at 3 seconds
  (h_m : m₂ = 2 * m₁) -- Mass ratio is 1:2
  (h_v : v_small = 16) -- Smaller fragment's horizontal speed is 16 m/s
  (h_g : g = 10)     -- Acceleration due to gravity is 10 m/s^2
  : ∃ v : ℝ, v = 17 ∧ v = 
    Real.sqrt ((2 * m₁ * v_small / (m₁ + m₂))^2 + (v₀ - g * t)^2) :=
by sorry

end NUMINAMATH_CALUDE_firecracker_explosion_speed_l429_42917


namespace NUMINAMATH_CALUDE_geometric_sequence_a4_l429_42969

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_a4 (a : ℕ → ℝ) :
  geometric_sequence a → a 2 = 4 → a 6 = 16 → a 4 = 8 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_a4_l429_42969


namespace NUMINAMATH_CALUDE_tangent_slope_at_A_l429_42916

-- Define the curve
def f (x : ℝ) : ℝ := x^2 + x

-- Define the point A
def point_A : ℝ × ℝ := (2, 6)

-- Theorem statement
theorem tangent_slope_at_A :
  (deriv f) point_A.1 = 5 := by sorry

end NUMINAMATH_CALUDE_tangent_slope_at_A_l429_42916


namespace NUMINAMATH_CALUDE_factorization_identities_l429_42918

theorem factorization_identities (x y : ℝ) : 
  (x^3 + 6*x^2 + 9*x = x*(x + 3)^2) ∧ 
  (16*x^2 - 9*y^2 = (4*x - 3*y)*(4*x + 3*y)) ∧ 
  ((3*x+y)^2 - (x-3*y)*(3*x+y) = 2*(3*x+y)*(x+2*y)) := by
  sorry

end NUMINAMATH_CALUDE_factorization_identities_l429_42918


namespace NUMINAMATH_CALUDE_inverse_proportional_problem_l429_42948

/-- Given that a and b are inversely proportional, their sum is 24, and their difference is 6,
    prove that when a = 5, b = 27. -/
theorem inverse_proportional_problem (a b : ℝ) (h1 : ∃ k : ℝ, a * b = k) 
  (h2 : a + b = 24) (h3 : a - b = 6) : a = 5 → b = 27 := by
  sorry

end NUMINAMATH_CALUDE_inverse_proportional_problem_l429_42948


namespace NUMINAMATH_CALUDE_solution_set1_solution_set2_l429_42930

-- Part 1
def system1 (x : ℝ) : Prop :=
  3 * x - (x - 2) ≥ 6 ∧ x + 1 > (4 * x - 1) / 3

theorem solution_set1 : 
  ∀ x : ℝ, system1 x ↔ 1 ≤ x ∧ x < 4 := by sorry

-- Part 2
def system2 (x : ℝ) : Prop :=
  2 * x + 1 > 0 ∧ x > 2 * x - 5

def is_positive_integer (x : ℝ) : Prop :=
  ∃ n : ℕ, x = n ∧ n > 0

theorem solution_set2 :
  {x : ℝ | system2 x ∧ is_positive_integer x} = {1, 2, 3, 4} := by sorry

end NUMINAMATH_CALUDE_solution_set1_solution_set2_l429_42930


namespace NUMINAMATH_CALUDE_base_nine_calculation_l429_42907

/-- Represents a number in base 9 --/
def BaseNine : Type := Nat

/-- Addition operation for base 9 numbers --/
def add_base_nine : BaseNine → BaseNine → BaseNine
| a, b => sorry

/-- Multiplication operation for base 9 numbers --/
def mul_base_nine : BaseNine → BaseNine → BaseNine
| a, b => sorry

/-- Converts a natural number to its base 9 representation --/
def to_base_nine : Nat → BaseNine
| n => sorry

theorem base_nine_calculation :
  let a : BaseNine := to_base_nine 35
  let b : BaseNine := to_base_nine 273
  let c : BaseNine := to_base_nine 2
  let result : BaseNine := to_base_nine 620
  mul_base_nine (add_base_nine a b) c = result := by sorry

end NUMINAMATH_CALUDE_base_nine_calculation_l429_42907


namespace NUMINAMATH_CALUDE_problem_solution_l429_42945

open Real

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x^2 + 4*a*x + 1

noncomputable def g (a b : ℝ) (x : ℝ) : ℝ := 6*a^2 * log x + 2*b + 1

noncomputable def h (a b : ℝ) (x : ℝ) : ℝ := f a x + g a b x

theorem problem_solution (a : ℝ) (ha : a > 0) :
  ∃ b : ℝ,
    (∃ x : ℝ, x > 0 ∧ f a x = g a b x ∧ (deriv (f a)) x = (deriv (g a b)) x) ∧
    b = (5/2)*a^2 - 3*a^2 * log a ∧
    ∀ b' : ℝ, b' ≤ (3/2) * Real.exp ((2:ℝ)/3) ∧
    (a ≥ Real.sqrt 3 - 1 →
      ∀ x₁ x₂ : ℝ, x₁ > 0 → x₂ > 0 → x₁ ≠ x₂ →
        (h a b x₂ - h a b x₁) / (x₂ - x₁) > 8) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l429_42945


namespace NUMINAMATH_CALUDE_custom_deck_combination_l429_42965

-- Define the number of suits
def num_suits : ℕ := 4

-- Define the number of cards per suit
def cards_per_suit : ℕ := 12

-- Define the number of face cards per suit
def face_cards_per_suit : ℕ := 3

-- Define the total number of cards in the deck
def total_cards : ℕ := num_suits * cards_per_suit

-- Theorem statement
theorem custom_deck_combination : 
  (Nat.choose num_suits 3) * 3 * face_cards_per_suit * cards_per_suit * cards_per_suit = 5184 := by
  sorry

end NUMINAMATH_CALUDE_custom_deck_combination_l429_42965


namespace NUMINAMATH_CALUDE_sphere_cube_volume_ratio_l429_42905

/-- Given a cube with its vertices on a spherical surface, 
    the ratio of the sphere's volume to the cube's volume is √3π/2 -/
theorem sphere_cube_volume_ratio : 
  ∀ (cube_edge : ℝ) (sphere_radius : ℝ),
  cube_edge > 0 →
  sphere_radius > 0 →
  sphere_radius = cube_edge * (Real.sqrt 3) / 2 →
  (4 / 3 * Real.pi * sphere_radius^3) / cube_edge^3 = Real.sqrt 3 * Real.pi / 2 :=
by sorry


end NUMINAMATH_CALUDE_sphere_cube_volume_ratio_l429_42905


namespace NUMINAMATH_CALUDE_sequence_and_sum_properties_l429_42903

def sequence_a (n : ℕ) : ℤ :=
  4 * n - 25

def sum_S (n : ℕ) : ℤ :=
  n * (sequence_a 1 + sequence_a n) / 2

theorem sequence_and_sum_properties :
  (sequence_a 3 = -13) ∧
  (∀ n > 1, sequence_a n = sequence_a (n - 1) + 4) ∧
  (sequence_a 1 = -21) ∧
  (sequence_a 2 = -17) ∧
  (∀ n, sequence_a n = 4 * n - 25) ∧
  (∀ k, sum_S 6 ≤ sum_S k) ∧
  (sum_S 6 = -66) := by
  sorry

end NUMINAMATH_CALUDE_sequence_and_sum_properties_l429_42903


namespace NUMINAMATH_CALUDE_winner_is_C_l429_42973

structure Singer :=
  (name : String)

def Singers : List Singer := [⟨"A"⟩, ⟨"B"⟩, ⟨"C"⟩, ⟨"D"⟩]

def Statement : Singer → Prop
| ⟨"A"⟩ => ∃ s : Singer, (s.name = "B" ∨ s.name = "C") ∧ s ∈ Singers
| ⟨"B"⟩ => ∀ s : Singer, (s.name = "A" ∨ s.name = "C") → s ∉ Singers
| ⟨"C"⟩ => ⟨"C"⟩ ∈ Singers
| ⟨"D"⟩ => ⟨"B"⟩ ∈ Singers
| _ => False

def Winner (s : Singer) : Prop :=
  s ∈ Singers ∧
  (∀ t : Singer, t ∈ Singers ∧ t ≠ s → t ∉ Singers) ∧
  (∃ (s1 s2 : Singer), s1 ≠ s2 ∧ Statement s1 ∧ Statement s2 ∧
    (∀ s3 : Singer, s3 ≠ s1 ∧ s3 ≠ s2 → ¬Statement s3))

theorem winner_is_C :
  Winner ⟨"C"⟩ ∧ (∀ s : Singer, s ≠ ⟨"C"⟩ → ¬Winner s) :=
sorry

end NUMINAMATH_CALUDE_winner_is_C_l429_42973


namespace NUMINAMATH_CALUDE_unique_solution_system_l429_42926

/-- The system of equations:
    1. 2(x-1) - 3(y+1) = 12
    2. x/2 + y/3 = 1
    has a unique solution (x, y) = (4, -3) -/
theorem unique_solution_system :
  ∃! (x y : ℝ), (2*(x-1) - 3*(y+1) = 12) ∧ (x/2 + y/3 = 1) ∧ x = 4 ∧ y = -3 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_system_l429_42926


namespace NUMINAMATH_CALUDE_negative_x_power_seven_divided_by_negative_x_l429_42993

theorem negative_x_power_seven_divided_by_negative_x (x : ℝ) :
  ((-x)^7) / (-x) = x^6 := by sorry

end NUMINAMATH_CALUDE_negative_x_power_seven_divided_by_negative_x_l429_42993


namespace NUMINAMATH_CALUDE_max_volume_container_frame_l429_42946

/-- Represents a rectangular container frame constructed from a steel bar -/
structure ContainerFrame where
  total_length : ℝ
  width : ℝ
  length : ℝ
  height : ℝ

/-- Calculates the volume of the container frame -/
def volume (c : ContainerFrame) : ℝ :=
  c.length * c.width * c.height

/-- Checks if the container frame satisfies the given conditions -/
def is_valid_frame (c : ContainerFrame) : Prop :=
  c.total_length = 14.8 ∧
  c.length = c.width + 0.5 ∧
  2 * (c.length + c.width) + 4 * c.height = c.total_length

/-- Theorem stating the maximum volume and corresponding height -/
theorem max_volume_container_frame :
  ∃ (c : ContainerFrame),
    is_valid_frame c ∧
    c.height = 1.8 ∧
    volume c = 1.512 ∧
    ∀ (c' : ContainerFrame), is_valid_frame c' → volume c' ≤ volume c :=
sorry

end NUMINAMATH_CALUDE_max_volume_container_frame_l429_42946


namespace NUMINAMATH_CALUDE_synthetic_analytic_properties_l429_42976

/-- Represents a reasoning approach in mathematics or logic -/
inductive ReasoningApproach
| Synthetic
| Analytic

/-- Represents the direction of reasoning -/
inductive ReasoningDirection
| Forward
| Backward

/-- Represents the relationship between cause and effect in reasoning -/
inductive CauseEffectRelation
| CauseToEffect
| EffectToCause

/-- Properties of a reasoning approach -/
structure ApproachProperties where
  direction : ReasoningDirection
  causeEffect : CauseEffectRelation

/-- Define properties of synthetic and analytic approaches -/
def approachProperties : ReasoningApproach → ApproachProperties
| ReasoningApproach.Synthetic => ⟨ReasoningDirection.Forward, CauseEffectRelation.CauseToEffect⟩
| ReasoningApproach.Analytic => ⟨ReasoningDirection.Backward, CauseEffectRelation.EffectToCause⟩

theorem synthetic_analytic_properties :
  (approachProperties ReasoningApproach.Synthetic).direction = ReasoningDirection.Forward ∧
  (approachProperties ReasoningApproach.Synthetic).causeEffect = CauseEffectRelation.CauseToEffect ∧
  (approachProperties ReasoningApproach.Analytic).direction = ReasoningDirection.Backward ∧
  (approachProperties ReasoningApproach.Analytic).causeEffect = CauseEffectRelation.EffectToCause :=
by sorry

end NUMINAMATH_CALUDE_synthetic_analytic_properties_l429_42976


namespace NUMINAMATH_CALUDE_inverse_proportion_k_condition_l429_42928

/-- Theorem: For an inverse proportion function y = (k-1)/x, given two points
    A(x₁, y₁) and B(x₂, y₂) on its graph where 0 < x₁ < x₂ and y₁ < y₂, 
    the value of k must be less than 1. -/
theorem inverse_proportion_k_condition 
  (k : ℝ) (x₁ x₂ y₁ y₂ : ℝ) 
  (h1 : 0 < x₁) (h2 : x₁ < x₂) (h3 : y₁ < y₂)
  (h4 : y₁ = (k - 1) / x₁) (h5 : y₂ = (k - 1) / x₂) : 
  k < 1 := by
  sorry

end NUMINAMATH_CALUDE_inverse_proportion_k_condition_l429_42928


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l429_42902

theorem imaginary_part_of_z (z : ℂ) (h : z - Complex.I = (4 - 2 * Complex.I) / (1 + 2 * Complex.I)) : 
  z.im = -1 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l429_42902


namespace NUMINAMATH_CALUDE_min_value_expression_min_value_achievable_l429_42982

theorem min_value_expression (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (x^2 + 4*x + 2) * (y^2 + 4*y + 2) * (z^2 + 4*z + 2) / (x*y*z) ≥ 216 :=
by sorry

theorem min_value_achievable :
  ∃ x y z : ℝ, x > 0 ∧ y > 0 ∧ z > 0 ∧
  (x^2 + 4*x + 2) * (y^2 + 4*y + 2) * (z^2 + 4*z + 2) / (x*y*z) = 216 :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_min_value_achievable_l429_42982


namespace NUMINAMATH_CALUDE_skittles_shared_l429_42904

theorem skittles_shared (starting_amount ending_amount : ℕ) 
  (h1 : starting_amount = 76)
  (h2 : ending_amount = 4) :
  starting_amount - ending_amount = 72 := by
  sorry

end NUMINAMATH_CALUDE_skittles_shared_l429_42904


namespace NUMINAMATH_CALUDE_polynomial_symmetry_l429_42968

/-- A polynomial is symmetric with respect to a point if and only if it has a specific form. -/
theorem polynomial_symmetry (P : ℝ → ℝ) (a b : ℝ) :
  (∀ x, P (2*a - x) = 2*b - P x) ↔
  (∃ Q : ℝ → ℝ, ∀ x, P x = b + (x - a) * Q ((x - a)^2)) :=
by sorry

end NUMINAMATH_CALUDE_polynomial_symmetry_l429_42968


namespace NUMINAMATH_CALUDE_repeating_decimal_as_fraction_l429_42937

/-- The repeating decimal 0.5̄10 as a rational number -/
def repeating_decimal : ℚ := 0.5 + 0.01 / (1 - 1/100)

/-- The theorem stating that 0.5̄10 is equal to 101/198 -/
theorem repeating_decimal_as_fraction : repeating_decimal = 101 / 198 := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_as_fraction_l429_42937


namespace NUMINAMATH_CALUDE_sqrt_meaningful_range_l429_42932

theorem sqrt_meaningful_range (x : ℝ) : 
  (∃ y : ℝ, y^2 = x - 4) → x ≥ 4 := by sorry

end NUMINAMATH_CALUDE_sqrt_meaningful_range_l429_42932


namespace NUMINAMATH_CALUDE_square_plus_reciprocal_square_l429_42986

theorem square_plus_reciprocal_square (x : ℝ) (h : x + (1/x) = 3) : x^2 + (1/x^2) = 7 := by
  sorry

end NUMINAMATH_CALUDE_square_plus_reciprocal_square_l429_42986


namespace NUMINAMATH_CALUDE_eulers_partition_theorem_l429_42955

/-- The number of partitions of a natural number into distinct parts -/
def d (n : ℕ) : ℕ := sorry

/-- The number of partitions of a natural number into odd parts -/
def l (n : ℕ) : ℕ := sorry

/-- Euler's partition theorem: The number of partitions of a natural number
    into distinct parts is equal to the number of partitions into odd parts -/
theorem eulers_partition_theorem : ∀ n : ℕ, d n = l n := by sorry

end NUMINAMATH_CALUDE_eulers_partition_theorem_l429_42955


namespace NUMINAMATH_CALUDE_product_inequality_l429_42925

theorem product_inequality (x₁ x₂ x₃ x₄ y₁ y₂ : ℝ) 
  (h1 : y₂ ≥ y₁ ∧ y₁ ≥ x₁ ∧ x₁ ≥ x₃ ∧ x₃ ≥ x₂ ∧ x₂ ≥ x₁ ∧ x₁ ≥ 2)
  (h2 : x₁ + x₂ + x₃ + x₄ ≥ y₁ + y₂) :
  x₁ * x₂ * x₃ * x₄ ≥ y₁ * y₂ := by
sorry

end NUMINAMATH_CALUDE_product_inequality_l429_42925


namespace NUMINAMATH_CALUDE_negative_reals_inequality_l429_42963

theorem negative_reals_inequality (a b c : ℝ) (ha : a < 0) (hb : b < 0) (hc : c < 0) :
  (Real.sqrt (a / (b + c)) + 1 / Real.sqrt 2) ^ 2 +
  (Real.sqrt (b / (c + a)) + 1 / Real.sqrt 2) ^ 2 +
  (Real.sqrt (c / (a + b)) + 1 / Real.sqrt 2) ^ 2 ≥ 6 := by
  sorry

end NUMINAMATH_CALUDE_negative_reals_inequality_l429_42963


namespace NUMINAMATH_CALUDE_max_value_of_d_l429_42988

theorem max_value_of_d (a b c d : ℝ) 
  (sum_eq : a + b + c + d = 10)
  (prod_sum_eq : a*b + a*c + a*d + b*c + b*d + c*d = 20) :
  d ≤ 5 + (5 * Real.sqrt 34) / 3 := by
sorry

end NUMINAMATH_CALUDE_max_value_of_d_l429_42988


namespace NUMINAMATH_CALUDE_oil_bill_ratio_change_l429_42939

theorem oil_bill_ratio_change (january_bill : ℚ) (february_bill : ℚ) : 
  january_bill = 120 →
  february_bill / january_bill = 5 / 4 →
  (february_bill + 30) / january_bill = 3 / 2 := by
sorry

end NUMINAMATH_CALUDE_oil_bill_ratio_change_l429_42939


namespace NUMINAMATH_CALUDE_subtract_negative_real_l429_42992

theorem subtract_negative_real : 3.7 - (-1.45) = 5.15 := by
  sorry

end NUMINAMATH_CALUDE_subtract_negative_real_l429_42992


namespace NUMINAMATH_CALUDE_cucumber_equivalent_to_16_apples_l429_42936

/-- The cost of fruits in an arbitrary unit -/
structure FruitCost where
  apple : ℕ → ℚ
  banana : ℕ → ℚ
  cucumber : ℕ → ℚ

/-- The given conditions about fruit costs -/
def fruit_cost_conditions (c : FruitCost) : Prop :=
  c.apple 8 = c.banana 4 ∧ c.banana 2 = c.cucumber 3

/-- The theorem to prove -/
theorem cucumber_equivalent_to_16_apples (c : FruitCost) 
  (h : fruit_cost_conditions c) : 
  ∃ n : ℕ, c.apple 16 = c.cucumber n ∧ n = 12 := by
  sorry

end NUMINAMATH_CALUDE_cucumber_equivalent_to_16_apples_l429_42936


namespace NUMINAMATH_CALUDE_x_range_l429_42990

theorem x_range (x y : ℝ) (h1 : 4 * x + y = 3) (h2 : -2 < y ∧ y ≤ 7) : 
  -1 ≤ x ∧ x < 5/4 := by
  sorry

end NUMINAMATH_CALUDE_x_range_l429_42990


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_l429_42931

theorem right_triangle_hypotenuse : 
  ∀ (a b c : ℝ), 
  a = 15 → b = 36 → c^2 = a^2 + b^2 → c = 39 :=
by
  sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_l429_42931


namespace NUMINAMATH_CALUDE_unique_positive_solution_l429_42984

theorem unique_positive_solution :
  ∃! (x : ℝ), x > 0 ∧ (x - 5) / 10 = 5 / (x - 10) :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_unique_positive_solution_l429_42984


namespace NUMINAMATH_CALUDE_complex_arithmetic_expression_equals_seven_l429_42972

theorem complex_arithmetic_expression_equals_seven :
  (2 + 3/5 - (17/2 - 8/3) / (7/2)) * (15/2) = 7 := by sorry

end NUMINAMATH_CALUDE_complex_arithmetic_expression_equals_seven_l429_42972


namespace NUMINAMATH_CALUDE_inequality_solution_l429_42964

theorem inequality_solution (x y : ℝ) : 
  2^y - 2 * Real.cos x + Real.sqrt (y - x^2 - 1) ≤ 0 ↔ x = 0 ∧ y = 1 :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l429_42964
