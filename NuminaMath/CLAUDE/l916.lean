import Mathlib

namespace problem_29_AHSME_1978_l916_91698

theorem problem_29_AHSME_1978 (a b c x : ℝ) 
  (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
  (h1 : (a + b - c) / c = (a - b + c) / b)
  (h2 : (a + b - c) / c = (-a + b + c) / a)
  (h3 : x = ((a + b) * (b + c) * (c + a)) / (a * b * c))
  (h4 : x < 0) :
  x = -1 := by sorry

end problem_29_AHSME_1978_l916_91698


namespace factor_sum_l916_91658

theorem factor_sum (P Q : ℝ) : 
  (∃ b c : ℝ, (X^2 + 3*X + 7) * (X^2 + b*X + c) = X^4 + P*X^2 + Q) →
  P + Q = 54 := by
sorry

end factor_sum_l916_91658


namespace no_integer_solutions_l916_91605

theorem no_integer_solutions : 
  ¬∃ (y z : ℤ), (2*y^2 - 2*y*z - z^2 = 15) ∧ 
                (6*y*z + 2*z^2 = 60) ∧ 
                (y^2 + 8*z^2 = 90) := by
  sorry

end no_integer_solutions_l916_91605


namespace kolya_is_wrong_l916_91692

/-- Represents the statements made by each boy -/
structure Statements where
  vasya : ℕ → Prop
  kolya : ℕ → Prop
  petya : ℕ → ℕ → Prop
  misha : ℕ → ℕ → Prop

/-- The actual statements made by the boys -/
def boys_statements : Statements where
  vasya := λ b => b ≥ 4
  kolya := λ g => g ≥ 5
  petya := λ b g => b ≥ 3 ∧ g ≥ 4
  misha := λ b g => b ≥ 4 ∧ g ≥ 4

/-- Theorem stating that Kolya's statement is the only one that can be false -/
theorem kolya_is_wrong (s : Statements) (b g : ℕ) :
  s = boys_statements →
  (s.vasya b ∧ s.petya b g ∧ s.misha b g ∧ ¬s.kolya g) ↔
  (b ≥ 4 ∧ g = 4) :=
sorry

end kolya_is_wrong_l916_91692


namespace equal_distances_exist_l916_91624

/-- Represents a position on an 8x8 grid -/
structure Position where
  row : Fin 8
  col : Fin 8

/-- Calculates the squared Euclidean distance between two positions -/
def squaredDistance (p1 p2 : Position) : ℕ :=
  (p1.row - p2.row).val ^ 2 + (p1.col - p2.col).val ^ 2

/-- Represents a configuration of 8 rooks on a chessboard -/
structure RookConfiguration where
  positions : Fin 8 → Position
  no_attack : ∀ i j, i ≠ j → positions i ≠ positions j

theorem equal_distances_exist (config : RookConfiguration) :
  ∃ i j k l : Fin 8, i < j ∧ k < l ∧ (i, j) ≠ (k, l) ∧
    squaredDistance (config.positions i) (config.positions j) =
    squaredDistance (config.positions k) (config.positions l) :=
sorry

end equal_distances_exist_l916_91624


namespace geometric_sum_first_eight_l916_91678

def geometric_sequence (a r : ℚ) (n : ℕ) : ℚ := a * r ^ (n - 1)

def geometric_sum (a r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

theorem geometric_sum_first_eight :
  let a : ℚ := 1/3
  let r : ℚ := 1/3
  let n : ℕ := 8
  geometric_sum a r n = 3280/6561 := by
sorry

end geometric_sum_first_eight_l916_91678


namespace gcd_5616_11609_l916_91627

theorem gcd_5616_11609 : Nat.gcd 5616 11609 = 13 := by
  sorry

end gcd_5616_11609_l916_91627


namespace line_passes_through_fixed_point_l916_91620

/-- A line in the form kx + y + k = 0 passes through the point (-1, 0) for all real k. -/
theorem line_passes_through_fixed_point :
  ∀ (k : ℝ), (k * (-1) + 0 + k = 0) := by
  sorry

end line_passes_through_fixed_point_l916_91620


namespace x_plus_y_equals_two_l916_91631

theorem x_plus_y_equals_two (x y : ℝ) 
  (hx : (x - 1)^3 + 1997*(x - 1) = -1)
  (hy : (y - 1)^3 + 1997*(y - 1) = 1) : 
  x + y = 2 := by
sorry

end x_plus_y_equals_two_l916_91631


namespace probability_two_boys_l916_91680

theorem probability_two_boys (total : ℕ) (boys : ℕ) (girls : ℕ) :
  total = 15 →
  boys = 8 →
  girls = 7 →
  boys + girls = total →
  (Nat.choose boys 2 : ℚ) / (Nat.choose total 2 : ℚ) = 4 / 15 := by
sorry

end probability_two_boys_l916_91680


namespace sin_RPS_equals_sin_RPQ_l916_91638

-- Define the angles
variable (RPQ RPS : Real)

-- Define the supplementary angle relationship
axiom supplementary_angles : RPQ + RPS = Real.pi

-- Define the given sine value
axiom sin_RPQ : Real.sin RPQ = 7/25

-- Theorem to prove
theorem sin_RPS_equals_sin_RPQ : Real.sin RPS = 7/25 := by
  sorry

end sin_RPS_equals_sin_RPQ_l916_91638


namespace curve_transformation_l916_91666

theorem curve_transformation (x : ℝ) : 2 * Real.cos (2 * (x - π/3)) = Real.sqrt 3 * Real.sin (2 * x) - Real.cos (2 * x) := by
  sorry

end curve_transformation_l916_91666


namespace table_length_is_77_l916_91622

/-- Represents the dimensions of a rectangular object -/
structure Dimensions where
  length : ℕ
  width : ℕ

/-- Represents the placement of a sheet on the table -/
structure SheetPlacement where
  horizontal_offset : ℕ
  vertical_offset : ℕ

/-- Calculates the final dimensions of a table covered by sheets -/
def calculateTableDimensions (sheet_size : Dimensions) (table_width : ℕ) : Dimensions :=
  let total_sheets := table_width - sheet_size.width
  { length := sheet_size.length + total_sheets,
    width := table_width }

theorem table_length_is_77 (sheet_size : Dimensions) (table_width : ℕ) :
  sheet_size.length = 5 →
  sheet_size.width = 8 →
  table_width = 80 →
  (calculateTableDimensions sheet_size table_width).length = 77 := by
  sorry

#eval (calculateTableDimensions { length := 5, width := 8 } 80).length

end table_length_is_77_l916_91622


namespace opposite_of_negative_1009_opposite_of_negative_1009_proof_l916_91618

theorem opposite_of_negative_1009 : Int → Prop :=
  fun x => x + (-1009) = 0 → x = 1009

-- The proof is omitted
theorem opposite_of_negative_1009_proof : opposite_of_negative_1009 1009 := by sorry

end opposite_of_negative_1009_opposite_of_negative_1009_proof_l916_91618


namespace least_multiple_of_25_greater_than_450_l916_91626

theorem least_multiple_of_25_greater_than_450 :
  ∀ n : ℕ, n > 0 ∧ 25 ∣ n ∧ n > 450 → n ≥ 475 :=
by
  sorry

end least_multiple_of_25_greater_than_450_l916_91626


namespace max_value_fraction_l916_91651

theorem max_value_fraction (x y : ℝ) (hx : -3 ≤ x ∧ x ≤ -1) (hy : 1 ≤ y ∧ y ≤ 3) :
  (∀ a b : ℝ, -3 ≤ a ∧ a ≤ -1 → 1 ≤ b ∧ b ≤ 3 → (a + b) / (a - b) ≤ (x + y) / (x - y)) →
  (x + y) / (x - y) = 1/2 :=
by sorry

end max_value_fraction_l916_91651


namespace inequality_proof_l916_91614

theorem inequality_proof (x₁ x₂ : ℝ) (h₁ : 0 < x₁) (h₂ : x₁ < x₂) :
  Real.sqrt (x₁ * x₂) < (x₁ - x₂) / (Real.log x₁ - Real.log x₂) ∧
  (x₁ - x₂) / (Real.log x₁ - Real.log x₂) < (x₁ + x₂) / 2 := by
  sorry

end inequality_proof_l916_91614


namespace number_of_students_l916_91653

theorem number_of_students (S : ℕ) (N : ℕ) : 
  (4 * S + 3 = N) → (5 * S = N + 6) → S = 9 := by
sorry

end number_of_students_l916_91653


namespace calculation_proof_l916_91676

theorem calculation_proof :
  let four_million : ℝ := 4 * 10^6
  let four_hundred_thousand : ℝ := 4 * 10^5
  let four_billion : ℝ := 4 * 10^9
  (four_million * four_hundred_thousand + four_billion) = 1.604 * 10^12 := by
  sorry

end calculation_proof_l916_91676


namespace subset_implies_m_values_l916_91641

def A : Set ℝ := {2, 3}

def B (m : ℝ) : Set ℝ := {x | m * x - 6 = 0}

theorem subset_implies_m_values (m : ℝ) (h : B m ⊆ A) : m = 0 ∨ m = 2 ∨ m = 3 := by
  sorry

end subset_implies_m_values_l916_91641


namespace solve_candy_problem_l916_91695

def candy_problem (kit_kat : ℕ) (nerds : ℕ) (lollipops : ℕ) (baby_ruth : ℕ) (remaining : ℕ) : Prop :=
  let hershey := 3 * kit_kat
  let reese := baby_ruth / 2
  let total := kit_kat + hershey + nerds + lollipops + baby_ruth + reese
  let given_away := total - remaining
  given_away = 5

theorem solve_candy_problem :
  candy_problem 5 8 11 10 49 := by sorry

end solve_candy_problem_l916_91695


namespace population_growth_rate_exists_and_unique_l916_91664

theorem population_growth_rate_exists_and_unique :
  ∃! r : ℝ, 0 < r ∧ r < 1 ∧ 20000 * (1 + r)^3 = 26620 := by
  sorry

end population_growth_rate_exists_and_unique_l916_91664


namespace range_of_a_l916_91633

theorem range_of_a (a x : ℝ) : 
  (∀ x, (x^2 - 7*x + 10 ≤ 0 → a < x ∧ x < a + 1) ∧ 
        (a < x ∧ x < a + 1 → ¬(∀ y, a < y ∧ y < a + 1 → y^2 - 7*y + 10 ≤ 0))) →
  2 ≤ a ∧ a ≤ 4 :=
by sorry

end range_of_a_l916_91633


namespace sum_sub_fixed_points_ln_exp_zero_l916_91615

/-- A real number t is a sub-fixed point of function f if f(t) = -t -/
def IsSubFixedPoint (f : ℝ → ℝ) (t : ℝ) : Prop := f t = -t

/-- The sum of sub-fixed points of ln and exp -/
def SumSubFixedPoints : ℝ := sorry

theorem sum_sub_fixed_points_ln_exp_zero : SumSubFixedPoints = 0 := by
  sorry

end sum_sub_fixed_points_ln_exp_zero_l916_91615


namespace count_solutions_power_diff_l916_91663

/-- The number of solutions to x^n - y^n = 2^100 where x, y, n are positive integers and n > 1 -/
theorem count_solutions_power_diff : 
  (Finset.filter 
    (fun t : ℕ × ℕ × ℕ => 
      let (x, y, n) := t
      x > 0 ∧ y > 0 ∧ n > 1 ∧ x^n - y^n = 2^100)
    (Finset.product (Finset.range (2^100 + 1)) 
      (Finset.product (Finset.range (2^100 + 1)) (Finset.range 101)))).card = 49 := by
  sorry

end count_solutions_power_diff_l916_91663


namespace postage_calculation_l916_91657

/-- Calculates the postage cost for a letter given its weight and rate structure -/
def calculate_postage (weight : ℚ) (base_rate : ℕ) (additional_rate : ℕ) : ℚ :=
  base_rate + additional_rate * (⌈weight - 1⌉ : ℚ)

/-- The postage for a 5.75 ounce letter is $1.00 given the specified rates -/
theorem postage_calculation :
  let weight : ℚ := 5.75
  let base_rate : ℕ := 25
  let additional_rate : ℕ := 15
  calculate_postage weight base_rate additional_rate = 100 := by
sorry

#eval calculate_postage (5.75 : ℚ) 25 15

end postage_calculation_l916_91657


namespace horner_operations_count_l916_91612

/-- Represents a univariate polynomial --/
structure UnivariatePoly (α : Type*) where
  coeffs : List α

/-- Horner's method for polynomial evaluation --/
def hornerMethod (p : UnivariatePoly ℤ) : ℕ × ℕ :=
  (p.coeffs.length - 1, p.coeffs.length - 1)

/-- The given polynomial f(x) = 5x^5 + 4x^4 + 3x^3 + 2x^2 + x + 1 --/
def f : UnivariatePoly ℤ :=
  ⟨[1, 1, 2, 3, 4, 5]⟩

theorem horner_operations_count :
  hornerMethod f = (5, 5) := by sorry

end horner_operations_count_l916_91612


namespace shadow_length_change_l916_91668

/-- Represents the length of a shadow -/
inductive ShadowLength
  | Long
  | Short

/-- Represents a time of day -/
inductive TimeOfDay
  | Morning
  | Noon
  | Afternoon

/-- Represents the direction of a shadow -/
inductive ShadowDirection
  | West
  | North
  | East

/-- Function to determine shadow length based on time of day -/
def shadowLengthAtTime (time : TimeOfDay) : ShadowLength :=
  match time with
  | TimeOfDay.Morning => ShadowLength.Long
  | TimeOfDay.Noon => ShadowLength.Short
  | TimeOfDay.Afternoon => ShadowLength.Long

/-- Function to determine shadow direction based on time of day -/
def shadowDirectionAtTime (time : TimeOfDay) : ShadowDirection :=
  match time with
  | TimeOfDay.Morning => ShadowDirection.West
  | TimeOfDay.Noon => ShadowDirection.North
  | TimeOfDay.Afternoon => ShadowDirection.East

/-- Theorem stating the change in shadow length throughout the day -/
theorem shadow_length_change :
  ∀ (t1 t2 t3 : TimeOfDay),
    t1 = TimeOfDay.Morning →
    t2 = TimeOfDay.Noon →
    t3 = TimeOfDay.Afternoon →
    (shadowLengthAtTime t1 = ShadowLength.Long ∧
     shadowLengthAtTime t2 = ShadowLength.Short ∧
     shadowLengthAtTime t3 = ShadowLength.Long) :=
by
  sorry

#check shadow_length_change

end shadow_length_change_l916_91668


namespace half_vector_AB_l916_91613

/-- Given two vectors OA and OB in ℝ², prove that half of vector AB equals (1/2, 5/2) -/
theorem half_vector_AB (OA OB : ℝ × ℝ) (h1 : OA = (3, 2)) (h2 : OB = (4, 7)) :
  (1 / 2 : ℝ) • (OB - OA) = (1/2, 5/2) := by sorry

end half_vector_AB_l916_91613


namespace min_value_4a_plus_b_l916_91681

theorem min_value_4a_plus_b (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a^2 + a*b - 3 = 0) :
  4*a + b ≥ 6 ∧ ∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ a₀^2 + a₀*b₀ - 3 = 0 ∧ 4*a₀ + b₀ = 6 :=
by sorry

end min_value_4a_plus_b_l916_91681


namespace difference_of_squares_81_49_l916_91619

theorem difference_of_squares_81_49 : 81^2 - 49^2 = 4160 := by
  sorry

end difference_of_squares_81_49_l916_91619


namespace sqrt_two_division_l916_91645

theorem sqrt_two_division : 2 * Real.sqrt 2 / Real.sqrt 2 = 2 := by
  sorry

end sqrt_two_division_l916_91645


namespace triangle_angle_theorem_l916_91688

theorem triangle_angle_theorem (a b c : ℝ) : 
  (a = 2 * b) →                 -- One angle is twice the second angle
  (c = b + 30) →                -- The third angle is 30° more than the second angle
  (a + b + c = 180) →           -- Sum of angles in a triangle is 180°
  (a = 75 ∧ b = 37.5 ∧ c = 67.5) -- The measures of the angles are 75°, 37.5°, and 67.5°
  := by sorry

end triangle_angle_theorem_l916_91688


namespace venny_car_cost_l916_91675

def original_price : ℝ := 37500

def discount_percentage : ℝ := 40

theorem venny_car_cost : ℝ := by
  -- Define the amount Venny spent as 40% of the original price
  let amount_spent := (discount_percentage / 100) * original_price
  
  -- Prove that this amount is equal to $15,000
  sorry

end venny_car_cost_l916_91675


namespace pizza_cost_is_twelve_l916_91693

/-- Calculates the cost of each pizza given the number of people, people per pizza, 
    earnings per night, and number of nights worked. -/
def pizza_cost (total_people : ℕ) (people_per_pizza : ℕ) (earnings_per_night : ℕ) (nights_worked : ℕ) : ℚ :=
  let total_pizzas := (total_people + people_per_pizza - 1) / people_per_pizza
  let total_earnings := earnings_per_night * nights_worked
  total_earnings / total_pizzas

/-- Proves that the cost of each pizza is $12 under the given conditions. -/
theorem pizza_cost_is_twelve :
  pizza_cost 15 3 4 15 = 12 := by
  sorry

end pizza_cost_is_twelve_l916_91693


namespace weighted_sum_inequality_l916_91629

theorem weighted_sum_inequality (a b c d : ℝ) 
  (pos_a : 0 < a) (pos_b : 0 < b) (pos_c : 0 < c) (pos_d : 0 < d)
  (order : a ≤ b ∧ b ≤ c ∧ c ≤ d)
  (sum_cond : a + b + c + d ≥ 1) :
  a^2 + 3*b^2 + 5*c^2 + 7*d^2 ≥ 1 := by
sorry

end weighted_sum_inequality_l916_91629


namespace perpendicular_to_same_line_relationships_l916_91672

-- Define a line in 3D space
structure Line3D where
  point : ℝ × ℝ × ℝ
  direction : ℝ × ℝ × ℝ

-- Define perpendicularity between a line and a plane
def perpendicular (l : Line3D) (p : Line3D) : Prop :=
  -- We assume this definition exists in the library
  sorry

-- Define the relationships between two lines
def parallel (l1 l2 : Line3D) : Prop :=
  -- We assume this definition exists in the library
  sorry

def intersect (l1 l2 : Line3D) : Prop :=
  -- We assume this definition exists in the library
  sorry

def skew (l1 l2 : Line3D) : Prop :=
  -- We assume this definition exists in the library
  sorry

-- Theorem statement
theorem perpendicular_to_same_line_relationships 
  (l1 l2 p : Line3D) (h1 : perpendicular l1 p) (h2 : perpendicular l2 p) :
  (parallel l1 l2 ∨ intersect l1 l2 ∨ skew l1 l2) :=
sorry

end perpendicular_to_same_line_relationships_l916_91672


namespace larger_number_proof_l916_91630

theorem larger_number_proof (x y : ℝ) (h1 : x + y = 45) (h2 : x - y = 5) (h3 : x ≥ y) : x = 25 := by
  sorry

end larger_number_proof_l916_91630


namespace complex_real_condition_l916_91637

theorem complex_real_condition (m : ℝ) :
  (∃ (x : ℝ), Complex.mk (m - 1) (m + 1) = x) ↔ m = -1 := by sorry

end complex_real_condition_l916_91637


namespace weight_of_b_l916_91684

theorem weight_of_b (a b c : ℝ) 
  (h1 : (a + b + c) / 3 = 45)
  (h2 : (a + b) / 2 = 40)
  (h3 : (b + c) / 2 = 41) :
  b = 27 := by
  sorry

end weight_of_b_l916_91684


namespace novel_pages_count_l916_91655

/-- Represents the number of pages in the novel -/
def total_pages : ℕ := 420

/-- Pages read on the first day -/
def pages_read_day1 (x : ℕ) : ℕ := x / 4 + 10

/-- Pages read on the second day -/
def pages_read_day2 (x : ℕ) : ℕ := (x - pages_read_day1 x) / 3 + 20

/-- Pages read on the third day -/
def pages_read_day3 (x : ℕ) : ℕ := (x - pages_read_day1 x - pages_read_day2 x) / 2 + 40

/-- Pages remaining after the third day -/
def pages_remaining (x : ℕ) : ℕ := x - pages_read_day1 x - pages_read_day2 x - pages_read_day3 x

theorem novel_pages_count : pages_remaining total_pages = 50 := by sorry

end novel_pages_count_l916_91655


namespace complex_point_on_line_l916_91677

theorem complex_point_on_line (a : ℝ) : 
  let z : ℂ := (1 + a * Complex.I) / (1 - Complex.I)
  (z.re - z.im = 1) → a = -1 := by
  sorry

end complex_point_on_line_l916_91677


namespace expansion_dissimilar_terms_l916_91616

/-- The number of dissimilar terms in the expansion of (a + b + c + d)^10 -/
def dissimilarTerms : ℕ := 286

/-- The number of variables in the expansion -/
def numVariables : ℕ := 4

/-- The exponent in the expansion -/
def exponent : ℕ := 10

/-- Theorem: The number of dissimilar terms in (a + b + c + d)^10 is 286 -/
theorem expansion_dissimilar_terms :
  dissimilarTerms = (numVariables + exponent - 1).choose (numVariables - 1) :=
sorry

end expansion_dissimilar_terms_l916_91616


namespace imaginary_part_of_complex_fraction_l916_91674

theorem imaginary_part_of_complex_fraction :
  let z : ℂ := 5 / (3 + 4 * I)
  Complex.im z = -(4 / 5) := by sorry

end imaginary_part_of_complex_fraction_l916_91674


namespace triangle_perimeter_l916_91609

theorem triangle_perimeter (a b x : ℝ) : 
  a = 1 → b = 2 → x^2 - 3*x + 2 = 0 → 
  (a + b > x ∧ a + x > b ∧ b + x > a) →
  a + b + x = 5 := by
sorry

end triangle_perimeter_l916_91609


namespace soda_cost_l916_91669

theorem soda_cost (bill : ℕ) (change : ℕ) (num_sodas : ℕ) (h1 : bill = 20) (h2 : change = 14) (h3 : num_sodas = 3) :
  (bill - change) / num_sodas = 2 := by
sorry

end soda_cost_l916_91669


namespace monochromatic_triangle_exists_l916_91647

-- Define a type for colors
inductive Color
| Red
| Blue
| Green

-- Define a type for the graph
def Graph := Fin 17 → Fin 17 → Color

-- Statement of the theorem
theorem monochromatic_triangle_exists (g : Graph) : 
  ∃ (a b c : Fin 17), a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ 
  g a b = g b c ∧ g b c = g a c :=
sorry

end monochromatic_triangle_exists_l916_91647


namespace division_remainder_proof_l916_91689

theorem division_remainder_proof (dividend : ℕ) (divisor : ℚ) (quotient : ℕ) (remainder : ℕ) : 
  dividend = 16698 →
  divisor = 187.46067415730337 →
  quotient = 89 →
  dividend = (divisor * quotient).floor + remainder →
  remainder = 14 := by
sorry

end division_remainder_proof_l916_91689


namespace hash_five_two_l916_91635

-- Define the # operation
def hash (a b : ℤ) : ℤ := (a + 2*b) * (a - 2*b)

-- Theorem statement
theorem hash_five_two : hash 5 2 = 9 := by
  sorry

end hash_five_two_l916_91635


namespace square_area_from_diagonal_l916_91648

theorem square_area_from_diagonal (d : ℝ) (h : d = 8 * Real.sqrt 2) :
  ∃ (s : ℝ), s * s = 64 ∧ d = s * Real.sqrt 2 := by
  sorry

end square_area_from_diagonal_l916_91648


namespace expression_change_l916_91683

theorem expression_change (x a : ℝ) (h : a > 0) :
  let f : ℝ → ℝ := λ t ↦ t^2 - 3
  (f (x + a) - f x = 2*a*x + a^2) ∧ (f (x - a) - f x = -2*a*x + a^2) :=
sorry

end expression_change_l916_91683


namespace oldest_babysat_age_l916_91673

-- Define constants
def jane_start_age : ℕ := 16
def jane_current_age : ℕ := 32
def years_since_stopped : ℕ := 10

-- Define the theorem
theorem oldest_babysat_age :
  ∀ (oldest_age : ℕ),
  (oldest_age = (jane_current_age - years_since_stopped) / 2 + years_since_stopped) →
  (oldest_age ≤ jane_current_age) →
  (∀ (jane_age : ℕ) (child_age : ℕ),
    jane_start_age ≤ jane_age →
    jane_age ≤ jane_current_age - years_since_stopped →
    child_age ≤ jane_age / 2 →
    child_age + (jane_current_age - jane_age) ≤ oldest_age) →
  oldest_age = 21 :=
by sorry

end oldest_babysat_age_l916_91673


namespace short_trees_planted_count_l916_91602

/-- The number of short trees planted in the park -/
def short_trees_planted (initial_short_trees final_short_trees : ℕ) : ℕ :=
  final_short_trees - initial_short_trees

/-- Theorem stating that the number of short trees planted is 64 -/
theorem short_trees_planted_count : short_trees_planted 31 95 = 64 := by
  sorry

end short_trees_planted_count_l916_91602


namespace quadratic_equation_solution_l916_91686

theorem quadratic_equation_solution (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) 
  (h : ∀ x : ℝ, x^2 + b*x + a = 0 ↔ x = a ∨ x = -b) : 
  a = 1 ∧ b = 1 := by
sorry

end quadratic_equation_solution_l916_91686


namespace tenth_term_is_18_l916_91600

/-- An arithmetic sequence with given conditions -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  a 2 = 2 ∧ a 5 = 8 ∧ ∀ n m : ℕ, a (n + 1) - a n = a (m + 1) - a m

/-- The 10th term of the arithmetic sequence is 18 -/
theorem tenth_term_is_18 (a : ℕ → ℝ) (h : arithmetic_sequence a) : a 10 = 18 := by
  sorry

end tenth_term_is_18_l916_91600


namespace yards_mowed_l916_91661

/-- The problem of calculating how many yards Christian mowed --/
theorem yards_mowed (perfume_price : ℕ) (christian_savings sue_savings : ℕ)
  (yard_price : ℕ) (dogs_walked dog_price : ℕ) (remaining : ℕ) :
  perfume_price = 50 →
  christian_savings = 5 →
  sue_savings = 7 →
  yard_price = 5 →
  dogs_walked = 6 →
  dog_price = 2 →
  remaining = 6 →
  (perfume_price - (christian_savings + sue_savings + dogs_walked * dog_price + remaining)) / yard_price = 4 :=
by
  sorry

end yards_mowed_l916_91661


namespace evaluate_expression_l916_91696

theorem evaluate_expression : (1 / ((-5^4)^2)) * (-5)^9 = -5 := by sorry

end evaluate_expression_l916_91696


namespace square_of_powers_l916_91650

theorem square_of_powers (n : ℕ) : 
  (∃ k : ℕ, 2^10 + 2^13 + 2^14 + 3 * 2^n = k^2) ↔ n = 13 ∨ n = 15 := by
sorry

end square_of_powers_l916_91650


namespace area_KLMQ_is_ten_l916_91601

structure Rectangle where
  width : ℝ
  height : ℝ

def area (r : Rectangle) : ℝ := r.width * r.height

theorem area_KLMQ_is_ten (JLMR JKQR : Rectangle) 
  (h1 : JLMR.width = 2)
  (h2 : JKQR.height = 3)
  (h3 : JLMR.height = 8) :
  ∃ KLMQ : Rectangle, area KLMQ = 10 :=
sorry

end area_KLMQ_is_ten_l916_91601


namespace skittles_distribution_l916_91634

theorem skittles_distribution (initial_skittles : ℕ) (additional_skittles : ℕ) (num_people : ℕ) :
  initial_skittles = 14 →
  additional_skittles = 22 →
  num_people = 7 →
  (initial_skittles + additional_skittles) / num_people = 5 :=
by sorry

end skittles_distribution_l916_91634


namespace valentines_count_l916_91660

theorem valentines_count (boys girls : ℕ) : 
  boys * girls = boys + girls + 16 → boys * girls = 36 := by
  sorry

end valentines_count_l916_91660


namespace chord_equation_l916_91628

/-- A circle in a 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- A line in a 2D plane represented by its equation ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Given circle, point P, prove that the line AB passing through P has the specified equation -/
theorem chord_equation (c : Circle) (p : ℝ × ℝ) : 
  c.center = (2, 0) → 
  c.radius = 4 → 
  p = (3, 1) → 
  ∃ (l : Line), l.a = 1 ∧ l.b = 1 ∧ l.c = -4 := by
  sorry

end chord_equation_l916_91628


namespace stating_tom_initial_investment_l916_91697

/-- Represents the profit sharing scenario between Tom and Jose -/
structure ProfitSharing where
  tom_investment : ℕ  -- Tom's initial investment
  jose_investment : ℕ := 45000  -- Jose's investment
  total_profit : ℕ := 72000  -- Total profit after one year
  jose_profit : ℕ := 40000  -- Jose's share of the profit
  tom_months : ℕ := 12  -- Months Tom was in business
  jose_months : ℕ := 10  -- Months Jose was in business

/-- 
Theorem stating that given the conditions of the profit sharing scenario, 
Tom's initial investment was 30000.
-/
theorem tom_initial_investment (ps : ProfitSharing) : ps.tom_investment = 30000 := by
  sorry

#check tom_initial_investment

end stating_tom_initial_investment_l916_91697


namespace difference_of_squares_l916_91652

theorem difference_of_squares (m : ℤ) : 
  (∃ x y : ℤ, m = x^2 - y^2) ↔ ¬(∃ k : ℤ, m = 4*k + 2) := by
  sorry

end difference_of_squares_l916_91652


namespace min_value_product_l916_91694

theorem min_value_product (x y z : ℝ) (h_pos : x > 0 ∧ y > 0 ∧ z > 0)
  (h_sum : x/y + y/z + z/x + y/x + z/y + x/z = 6) :
  (x/y + y/z + z/x) * (y/x + z/y + x/z) ≥ 15 :=
by sorry

end min_value_product_l916_91694


namespace al_sewing_time_l916_91623

/-- Represents the time it takes for Al to sew dresses individually -/
def al_time : ℝ := 12

/-- Represents the time it takes for Allison to sew dresses individually -/
def allison_time : ℝ := 9

/-- Represents the time Allison and Al work together -/
def together_time : ℝ := 3

/-- Represents the additional time Allison needs after Al leaves -/
def allison_additional_time : ℝ := 3.75

/-- Theorem stating that Al's individual sewing time is 12 hours -/
theorem al_sewing_time : 
  (together_time * (1 / allison_time + 1 / al_time)) + 
  (allison_additional_time * (1 / allison_time)) = 1 := by
sorry

end al_sewing_time_l916_91623


namespace odd_function_property_l916_91640

-- Define an odd function f
def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

-- Main theorem
theorem odd_function_property (f : ℝ → ℝ) 
  (h_odd : is_odd_function f)
  (h_f_1 : f 1 = 1/2)
  (h_f_shift : ∀ x, f (x + 2) = f x + f 2) :
  f 5 = 5/2 := by
  sorry

end odd_function_property_l916_91640


namespace average_flux_1_to_999_l916_91617

/-- The flux of a positive integer is the number of times the digits change from increasing to decreasing or vice versa, ignoring consecutive equal digits. -/
def flux (n : ℕ+) : ℕ := sorry

/-- The sum of fluxes for all positive integers from 1 to 999, inclusive. -/
def sum_of_fluxes : ℕ := sorry

theorem average_flux_1_to_999 :
  (sum_of_fluxes : ℚ) / 999 = 175 / 333 := by sorry

end average_flux_1_to_999_l916_91617


namespace polynomial_factorization_l916_91636

theorem polynomial_factorization (x y : ℝ) : x * y^2 - 16 * x = x * (y + 4) * (y - 4) := by
  sorry

end polynomial_factorization_l916_91636


namespace train_length_calculation_l916_91610

/-- Proves that given two trains of equal length running on parallel lines in the same direction,
    with speeds of 45 km/hr and 36 km/hr respectively, if the faster train passes the slower train
    in 36 seconds, then the length of each train is 45 meters. -/
theorem train_length_calculation (faster_speed slower_speed : ℝ) (passing_time : ℝ) 
  (h1 : faster_speed = 45) 
  (h2 : slower_speed = 36) 
  (h3 : passing_time = 36) : 
  let relative_speed := (faster_speed - slower_speed) * (5 / 18)
  let distance_covered := relative_speed * passing_time
  let train_length := distance_covered / 2
  train_length = 45 := by
sorry

end train_length_calculation_l916_91610


namespace money_value_difference_l916_91656

def euro_to_dollar : ℝ := 1.5
def diana_dollars : ℝ := 600
def etienne_euros : ℝ := 450

theorem money_value_difference : 
  let etienne_dollars := etienne_euros * euro_to_dollar
  let percentage_diff := (diana_dollars - etienne_dollars) / etienne_dollars * 100
  ∀ ε > 0, |percentage_diff + 11.11| < ε :=
sorry

end money_value_difference_l916_91656


namespace original_mean_calculation_l916_91643

theorem original_mean_calculation (n : ℕ) (decrease : ℝ) (updated_mean : ℝ) :
  n = 50 →
  decrease = 6 →
  updated_mean = 194 →
  (updated_mean + decrease : ℝ) = 200 := by
  sorry

end original_mean_calculation_l916_91643


namespace distribution_centers_count_l916_91667

/-- The number of unique representations using either a single color or a pair of different colors -/
def uniqueRepresentations (n : ℕ) : ℕ := n + n.choose 2

/-- Theorem stating that with 5 colors, there are 15 unique representations -/
theorem distribution_centers_count : uniqueRepresentations 5 = 15 := by
  sorry

end distribution_centers_count_l916_91667


namespace alice_additional_spend_l916_91671

/-- The amount Alice needs to spend for free delivery -/
def free_delivery_threshold : ℚ := 35

/-- The cost of chicken per pound -/
def chicken_price : ℚ := 6

/-- The amount of chicken in pounds -/
def chicken_amount : ℚ := 3/2

/-- The cost of lettuce -/
def lettuce_price : ℚ := 3

/-- The cost of cherry tomatoes -/
def tomatoes_price : ℚ := 5/2

/-- The cost of one sweet potato -/
def sweet_potato_price : ℚ := 3/4

/-- The number of sweet potatoes -/
def sweet_potato_count : ℕ := 4

/-- The cost of one head of broccoli -/
def broccoli_price : ℚ := 2

/-- The number of broccoli heads -/
def broccoli_count : ℕ := 2

/-- The cost of Brussel sprouts -/
def brussel_sprouts_price : ℚ := 5/2

/-- The total cost of items in Alice's cart -/
def cart_total : ℚ :=
  chicken_price * chicken_amount + lettuce_price + tomatoes_price +
  sweet_potato_price * sweet_potato_count + broccoli_price * broccoli_count +
  brussel_sprouts_price

/-- The additional amount Alice needs to spend for free delivery -/
def additional_spend : ℚ := free_delivery_threshold - cart_total

theorem alice_additional_spend :
  additional_spend = 11 := by sorry

end alice_additional_spend_l916_91671


namespace sufficient_but_not_necessary_l916_91611

theorem sufficient_but_not_necessary : 
  (∀ x : ℝ, x^2 - 2*x < 0 → abs (x - 2) < 2) ∧ 
  (∃ x : ℝ, abs (x - 2) < 2 ∧ ¬(x^2 - 2*x < 0)) := by
sorry

end sufficient_but_not_necessary_l916_91611


namespace right_triangle_inequality_l916_91679

theorem right_triangle_inequality (a b c : ℝ) (h : a > 0 ∧ b > 0 ∧ c > 0) 
  (pythagorean : a^2 + b^2 = c^2) : (a + b) / (a * b / c) ≤ 2 * Real.sqrt 2 := by
  sorry

end right_triangle_inequality_l916_91679


namespace students_liking_both_desserts_l916_91606

theorem students_liking_both_desserts 
  (total : Nat) 
  (like_apple : Nat) 
  (like_chocolate : Nat) 
  (like_neither : Nat) 
  (h1 : total = 40)
  (h2 : like_apple = 18)
  (h3 : like_chocolate = 15)
  (h4 : like_neither = 12) :
  like_apple + like_chocolate - (total - like_neither) = 5 := by
  sorry

end students_liking_both_desserts_l916_91606


namespace geometric_sequence_ninth_term_l916_91639

def geometric_sequence (a : ℕ → ℚ) : Prop :=
  ∃ r : ℚ, ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_ninth_term
  (a : ℕ → ℚ)
  (h_geom : geometric_sequence a)
  (h_first : a 1 = 1/2)
  (h_relation : a 2 * a 8 = 2 * a 5 + 3) :
  a 9 = 18 := by
  sorry

end geometric_sequence_ninth_term_l916_91639


namespace seeds_per_can_l916_91642

def total_seeds : ℕ := 54
def num_cans : ℕ := 9

theorem seeds_per_can :
  total_seeds / num_cans = 6 :=
by sorry

end seeds_per_can_l916_91642


namespace age_difference_l916_91625

theorem age_difference (x y z : ℕ) : 
  x + y = y + z + 18 → (x - z : ℚ) / 10 = 1.8 := by
  sorry

end age_difference_l916_91625


namespace lucas_investment_l916_91685

theorem lucas_investment (total_investment : ℝ) (alpha_rate beta_rate : ℝ) (final_amount : ℝ)
  (h1 : total_investment = 1500)
  (h2 : alpha_rate = 0.04)
  (h3 : beta_rate = 0.06)
  (h4 : final_amount = 1584.50) :
  ∃ (alpha_investment : ℝ),
    alpha_investment * (1 + alpha_rate) + (total_investment - alpha_investment) * (1 + beta_rate) = final_amount ∧
    alpha_investment = 275 :=
by sorry

end lucas_investment_l916_91685


namespace rosa_pages_called_l916_91687

/-- The number of pages Rosa called last week -/
def last_week_pages : ℝ := 10.2

/-- The total number of pages Rosa called -/
def total_pages : ℝ := 18.8

/-- The number of pages Rosa called this week -/
def this_week_pages : ℝ := total_pages - last_week_pages

theorem rosa_pages_called : this_week_pages = 8.6 := by
  sorry

end rosa_pages_called_l916_91687


namespace special_polynomial_value_l916_91632

theorem special_polynomial_value (x : ℝ) (h : x + 1/x = 3) : 
  x^10 - 5*x^6 + x^2 = 8436*x - 338 := by
sorry

end special_polynomial_value_l916_91632


namespace base6_addition_l916_91654

/-- Represents a number in base 6 as a list of digits (least significant first) -/
def Base6 := List Nat

/-- Addition of two base 6 numbers -/
def add_base6 (a b : Base6) : Base6 :=
  sorry

/-- Conversion of a natural number to base 6 -/
def to_base6 (n : Nat) : Base6 :=
  sorry

/-- Conversion of a base 6 number to a natural number -/
def from_base6 (b : Base6) : Nat :=
  sorry

theorem base6_addition :
  add_base6 [2, 3, 5, 4] [6, 4, 3, 5, 2] = [5, 2, 5, 2, 3] :=
sorry

end base6_addition_l916_91654


namespace cheryl_material_usage_l916_91644

theorem cheryl_material_usage 
  (material1 : ℚ) 
  (material2 : ℚ) 
  (leftover : ℚ) 
  (h1 : material1 = 2 / 9) 
  (h2 : material2 = 1 / 8) 
  (h3 : leftover = 4 / 18) : 
  material1 + material2 - leftover = 1 / 8 := by
  sorry

end cheryl_material_usage_l916_91644


namespace sum_of_y_values_l916_91603

theorem sum_of_y_values (x y : ℝ) : 
  x^2 + x^2*y^2 + x^2*y^4 = 525 ∧ x + x*y + x*y^2 = 35 →
  ∃ y₁ y₂ : ℝ, (x^2 + x^2*y₁^2 + x^2*y₁^4 = 525 ∧ x + x*y₁ + x*y₁^2 = 35) ∧
             (x^2 + x^2*y₂^2 + x^2*y₂^4 = 525 ∧ x + x*y₂ + x*y₂^2 = 35) ∧
             y₁ + y₂ = 5/2 :=
by sorry

end sum_of_y_values_l916_91603


namespace tangent_equation_solution_l916_91607

theorem tangent_equation_solution (x : Real) :
  5.30 * Real.tan x * Real.tan (20 * π / 180) +
  Real.tan (20 * π / 180) * Real.tan (40 * π / 180) +
  Real.tan (40 * π / 180) * Real.tan x = 1 →
  ∃ k : ℤ, x = (30 + 180 * k) * π / 180 :=
by sorry

end tangent_equation_solution_l916_91607


namespace min_value_reciprocal_sum_l916_91649

theorem min_value_reciprocal_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 2 * x + y = 2) :
  (∀ x' y' : ℝ, x' > 0 → y' > 0 → 2 * x' + y' = 2 → 1 / x' + 1 / y' ≥ 3 / 2 + Real.sqrt 2) ∧
  (∃ x₀ y₀ : ℝ, x₀ > 0 ∧ y₀ > 0 ∧ 2 * x₀ + y₀ = 2 ∧ 1 / x₀ + 1 / y₀ = 3 / 2 + Real.sqrt 2) :=
by sorry

end min_value_reciprocal_sum_l916_91649


namespace linear_function_through_point_l916_91690

theorem linear_function_through_point (k : ℝ) : 
  (∀ x : ℝ, (k * x = k * 3) → (k * x = 1)) → k = 1/3 := by
  sorry

end linear_function_through_point_l916_91690


namespace logarithm_equality_l916_91659

theorem logarithm_equality (c d : ℝ) : 
  c = Real.log 400 / Real.log 4 → d = Real.log 20 / Real.log 2 → c = d := by
  sorry

end logarithm_equality_l916_91659


namespace gcd_properties_l916_91662

theorem gcd_properties (a b : ℤ) (h : Nat.gcd a.natAbs b.natAbs = 1) :
  (Nat.gcd (a + b).natAbs (a * b).natAbs = 1 ∧
   Nat.gcd (a - b).natAbs (a * b).natAbs = 1) ∧
  (Nat.gcd (a + b).natAbs (a - b).natAbs = 1 ∨
   Nat.gcd (a + b).natAbs (a - b).natAbs = 2) := by
  sorry

end gcd_properties_l916_91662


namespace no_students_in_both_l916_91691

/-- Represents the number of students in different language classes -/
structure LanguageClasses where
  total : ℕ
  onlyFrench : ℕ
  onlySpanish : ℕ
  neither : ℕ

/-- Calculates the number of students taking both French and Spanish -/
def studentsInBoth (classes : LanguageClasses) : ℕ :=
  classes.total - (classes.onlyFrench + classes.onlySpanish + classes.neither)

/-- Theorem: In the given scenario, no students are taking both French and Spanish -/
theorem no_students_in_both (classes : LanguageClasses)
  (h_total : classes.total = 28)
  (h_french : classes.onlyFrench = 5)
  (h_spanish : classes.onlySpanish = 10)
  (h_neither : classes.neither = 13) :
  studentsInBoth classes = 0 := by
  sorry

#eval studentsInBoth { total := 28, onlyFrench := 5, onlySpanish := 10, neither := 13 }

end no_students_in_both_l916_91691


namespace cone_height_l916_91665

-- Define the cone
structure Cone where
  surfaceArea : ℝ
  centralAngle : ℝ

-- Theorem statement
theorem cone_height (c : Cone) 
  (h1 : c.surfaceArea = π) 
  (h2 : c.centralAngle = 2 * π / 3) : 
  ∃ h : ℝ, h = Real.sqrt 2 ∧ h > 0 := by
  sorry

end cone_height_l916_91665


namespace children_toothpaste_sales_amount_l916_91604

/-- Calculates the total sales amount for children's toothpaste. -/
def total_sales_amount (num_boxes : ℕ) (packs_per_box : ℕ) (price_per_pack : ℕ) : ℕ :=
  num_boxes * packs_per_box * price_per_pack

/-- Proves that the total sales amount for the given conditions is 1200 yuan. -/
theorem children_toothpaste_sales_amount :
  total_sales_amount 12 25 4 = 1200 := by
  sorry

end children_toothpaste_sales_amount_l916_91604


namespace problem_solution_l916_91646

theorem problem_solution (a b : ℝ) : 
  ({1, a, b/a} : Set ℝ) = {0, a^2, a+b} → a^2015 + b^2015 = -1 := by
  sorry

end problem_solution_l916_91646


namespace tape_left_over_l916_91621

/-- Calculates the amount of tape left over after wrapping a rectangular field once -/
theorem tape_left_over (total_tape : ℕ) (width : ℕ) (length : ℕ) : 
  total_tape = 250 → width = 20 → length = 60 → 
  total_tape - 2 * (width + length) = 90 := by
  sorry

end tape_left_over_l916_91621


namespace initials_probability_l916_91608

/-- The number of students in the class -/
def class_size : ℕ := 30

/-- The number of consonants in the alphabet (excluding Y) -/
def num_consonants : ℕ := 21

/-- The number of consonants we're interested in (B, C, D) -/
def target_consonants : ℕ := 3

/-- The probability of selecting a student with initials starting with B, C, or D -/
def probability : ℚ := 1 / 21

theorem initials_probability :
  probability = (min class_size (target_consonants * (num_consonants - 1))) / (class_size * num_consonants) :=
sorry

end initials_probability_l916_91608


namespace simplify_expressions_l916_91670

theorem simplify_expressions (x y : ℝ) :
  (2 * (2 * x - y) - (x + y) = 3 * x - 3 * y) ∧
  (x^2 * y + (-3 * (2 * x * y - x^2 * y) - x * y) = 4 * x^2 * y - 7 * x * y) := by
sorry

end simplify_expressions_l916_91670


namespace cube_remainder_mod_nine_l916_91699

theorem cube_remainder_mod_nine (n : ℤ) :
  (n % 9 = 2 ∨ n % 9 = 5 ∨ n % 9 = 8) → n^3 % 9 = 8 := by
  sorry

end cube_remainder_mod_nine_l916_91699


namespace student_distribution_l916_91682

theorem student_distribution (total : ℝ) (third_year : ℝ) (second_year : ℝ)
  (h1 : third_year = 0.5 * total)
  (h2 : second_year = 0.3 * total)
  (h3 : total > 0) :
  second_year / (total - third_year) = 3 / 5 := by
sorry

end student_distribution_l916_91682
