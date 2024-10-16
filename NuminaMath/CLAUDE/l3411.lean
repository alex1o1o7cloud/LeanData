import Mathlib

namespace NUMINAMATH_CALUDE_three_person_job_time_specific_job_time_l3411_341163

/-- The time taken to complete a job when multiple people work together -/
def time_to_complete (rates : List ℚ) : ℚ :=
  1 / (rates.sum)

/-- Proof that the time taken by three people working together is correct -/
theorem three_person_job_time (man_days son_days father_days : ℚ) 
  (man_days_pos : 0 < man_days) 
  (son_days_pos : 0 < son_days) 
  (father_days_pos : 0 < father_days) : 
  time_to_complete [1/man_days, 1/son_days, 1/father_days] = 
  1 / (1/man_days + 1/son_days + 1/father_days) :=
by sorry

/-- Application to the specific problem -/
theorem specific_job_time : 
  time_to_complete [1/20, 1/25, 1/20] = 100/14 :=
by sorry

end NUMINAMATH_CALUDE_three_person_job_time_specific_job_time_l3411_341163


namespace NUMINAMATH_CALUDE_definite_integral_3x_squared_l3411_341109

theorem definite_integral_3x_squared : ∫ x in (1:ℝ)..2, 3 * x^2 = 7 := by sorry

end NUMINAMATH_CALUDE_definite_integral_3x_squared_l3411_341109


namespace NUMINAMATH_CALUDE_max_value_of_f_l3411_341136

def f (x : ℝ) := x^2 - 2*x - 5

theorem max_value_of_f :
  ∃ (M : ℝ), M = 3 ∧ 
  (∀ x : ℝ, -2 ≤ x ∧ x ≤ 2 → f x ≤ M) ∧
  (∃ x : ℝ, -2 ≤ x ∧ x ≤ 2 ∧ f x = M) := by
sorry

end NUMINAMATH_CALUDE_max_value_of_f_l3411_341136


namespace NUMINAMATH_CALUDE_fraction_to_decimal_l3411_341183

theorem fraction_to_decimal : 13 / 243 = 0.00416 := by
  sorry

end NUMINAMATH_CALUDE_fraction_to_decimal_l3411_341183


namespace NUMINAMATH_CALUDE_dress_making_hours_l3411_341185

def total_fabric : ℕ := 56
def fabric_per_dress : ℕ := 4
def hours_per_dress : ℕ := 3

theorem dress_making_hours : 
  (total_fabric / fabric_per_dress) * hours_per_dress = 42 := by
  sorry

end NUMINAMATH_CALUDE_dress_making_hours_l3411_341185


namespace NUMINAMATH_CALUDE_f_at_2_l3411_341167

theorem f_at_2 (f : ℝ → ℝ) (h : ∀ x, f x = 3 * x^2 + 5 * x - 1) : f 2 = 21 := by
  sorry

end NUMINAMATH_CALUDE_f_at_2_l3411_341167


namespace NUMINAMATH_CALUDE_sams_balloons_l3411_341101

theorem sams_balloons (fred_balloons : ℕ) (mary_balloons : ℕ) (total_balloons : ℕ) 
  (h1 : fred_balloons = 5)
  (h2 : mary_balloons = 7)
  (h3 : total_balloons = 18) :
  total_balloons - fred_balloons - mary_balloons = 6 :=
by sorry

end NUMINAMATH_CALUDE_sams_balloons_l3411_341101


namespace NUMINAMATH_CALUDE_quadratic_roots_product_l3411_341162

theorem quadratic_roots_product (p q P Q : ℝ) (α β γ δ : ℝ) 
  (h1 : α^2 + p*α + q = 0)
  (h2 : β^2 + p*β + q = 0)
  (h3 : γ^2 + P*γ + Q = 0)
  (h4 : δ^2 + P*δ + Q = 0) :
  (α - γ) * (β - γ) * (α - δ) * (β - δ) = P^2 * q - P * p * Q + Q^2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_product_l3411_341162


namespace NUMINAMATH_CALUDE_cubic_sum_zero_l3411_341172

theorem cubic_sum_zero (a b c : ℝ) (h_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c) 
  (h_sum : (a^2 / (b - c)^2) + (b^2 / (c - a)^2) + (c^2 / (a - b)^2) = 0) :
  (a^3 / (b - c)^3) + (b^3 / (c - a)^3) + (c^3 / (a - b)^3) = 0 := by
  sorry

end NUMINAMATH_CALUDE_cubic_sum_zero_l3411_341172


namespace NUMINAMATH_CALUDE_sophie_germain_identity_l3411_341126

theorem sophie_germain_identity (a b : ℝ) : 
  a^4 + 4*b^4 = (a^2 - 2*a*b + 2*b^2) * (a^2 + 2*a*b + 2*b^2) := by
  sorry

end NUMINAMATH_CALUDE_sophie_germain_identity_l3411_341126


namespace NUMINAMATH_CALUDE_whipped_cream_theorem_l3411_341193

/-- Represents the number of each type of baked good produced on odd and even days -/
structure BakingSchedule where
  odd_pumpkin : ℕ
  odd_apple : ℕ
  odd_chocolate : ℕ
  even_pumpkin : ℕ
  even_apple : ℕ
  even_chocolate : ℕ
  even_lemon : ℕ

/-- Represents the amount of whipped cream needed for each type of baked good -/
structure WhippedCreamRequirement where
  pumpkin : ℚ
  apple : ℚ
  chocolate : ℚ
  lemon : ℚ

/-- Represents the number of each type of baked good Tiffany eats -/
structure TiffanyEats where
  pumpkin : ℕ
  apple : ℕ
  chocolate : ℕ
  lemon : ℕ

/-- Calculates the number of cans of whipped cream needed given the baking schedule,
    whipped cream requirements, and what Tiffany eats -/
def whippedCreamNeeded (schedule : BakingSchedule) (requirement : WhippedCreamRequirement) 
                       (tiffanyEats : TiffanyEats) : ℕ :=
  sorry

theorem whipped_cream_theorem (schedule : BakingSchedule) (requirement : WhippedCreamRequirement) 
                               (tiffanyEats : TiffanyEats) : 
  schedule = {
    odd_pumpkin := 3, odd_apple := 2, odd_chocolate := 1,
    even_pumpkin := 2, even_apple := 4, even_chocolate := 2, even_lemon := 1
  } →
  requirement = {
    pumpkin := 2, apple := 1, chocolate := 3, lemon := 3/2
  } →
  tiffanyEats = {
    pumpkin := 2, apple := 5, chocolate := 1, lemon := 1
  } →
  whippedCreamNeeded schedule requirement tiffanyEats = 252 :=
by
  sorry


end NUMINAMATH_CALUDE_whipped_cream_theorem_l3411_341193


namespace NUMINAMATH_CALUDE_expression_as_square_of_binomial_l3411_341154

/-- Represents the expression (-4b-3a)(-3a+4b) -/
def expression (a b : ℝ) : ℝ := (-4*b - 3*a) * (-3*a + 4*b)

/-- Represents the square of binomial form (x - y)(x + y) = x^2 - y^2 -/
def squareOfBinomialForm (x y : ℝ) : ℝ := x^2 - y^2

/-- Theorem stating that the given expression can be rewritten in a form 
    related to the square of a binomial -/
theorem expression_as_square_of_binomial (a b : ℝ) : 
  ∃ (x y : ℝ), expression a b = squareOfBinomialForm x y := by
  sorry

end NUMINAMATH_CALUDE_expression_as_square_of_binomial_l3411_341154


namespace NUMINAMATH_CALUDE_count_non_divisible_eq_31_l3411_341108

/-- The product of proper positive integer divisors of n -/
def g_hat (n : ℕ) : ℕ := sorry

/-- Counts the number of integers n between 2 and 100 (inclusive) for which n does not divide g_hat(n) -/
def count_non_divisible : ℕ := sorry

/-- Theorem stating that the count of non-divisible numbers is 31 -/
theorem count_non_divisible_eq_31 : count_non_divisible = 31 := by sorry

end NUMINAMATH_CALUDE_count_non_divisible_eq_31_l3411_341108


namespace NUMINAMATH_CALUDE_expression_value_l3411_341169

theorem expression_value (x y : ℤ) (hx : x = 3) (hy : y = 2) : 3 * x - 4 * y = 1 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l3411_341169


namespace NUMINAMATH_CALUDE_corner_value_theorem_l3411_341105

/-- Represents a 3x3 grid with the given corner values -/
structure Grid :=
  (top_left : ℤ)
  (top_right : ℤ)
  (bottom_left : ℤ)
  (bottom_right : ℤ)
  (top_middle : ℤ)
  (left_middle : ℤ)
  (right_middle : ℤ)
  (bottom_middle : ℤ)
  (center : ℤ)

/-- Checks if all 2x2 subgrids have the same sum -/
def equal_subgrid_sums (g : Grid) : Prop :=
  g.top_left + g.top_middle + g.left_middle + g.center =
  g.top_middle + g.top_right + g.center + g.right_middle ∧
  g.left_middle + g.center + g.bottom_left + g.bottom_middle =
  g.center + g.right_middle + g.bottom_middle + g.bottom_right

/-- The main theorem -/
theorem corner_value_theorem (g : Grid) 
  (h1 : g.top_left = 2)
  (h2 : g.top_right = 4)
  (h3 : g.bottom_right = 3)
  (h4 : equal_subgrid_sums g) :
  g.bottom_left = 1 := by
  sorry

end NUMINAMATH_CALUDE_corner_value_theorem_l3411_341105


namespace NUMINAMATH_CALUDE_C_is_smallest_l3411_341151

def A : ℤ := 18 + 38

def B : ℤ := A - 26

def C : ℚ := B / 3

theorem C_is_smallest : C < A ∧ C < B := by
  sorry

end NUMINAMATH_CALUDE_C_is_smallest_l3411_341151


namespace NUMINAMATH_CALUDE_cyclic_inequality_l3411_341186

theorem cyclic_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a^2 + a*b + b^2) * (b^2 + b*c + c^2) * (c^2 + c*a + a^2) ≥ (a*b + b*c + c*a)^2 := by
  sorry

end NUMINAMATH_CALUDE_cyclic_inequality_l3411_341186


namespace NUMINAMATH_CALUDE_natural_number_pairs_l3411_341159

theorem natural_number_pairs : ∀ a b : ℕ, 
  (∃! (s1 s2 s3 s4 : Prop), 
    s1 = (∃ k : ℕ, a^2 + 4*a + 3 = k * b) ∧
    s2 = (a^2 + a*b - 6*b^2 - 2*a - 16*b - 8 = 0) ∧
    s3 = (∃ k : ℕ, a + 2*b + 1 = 4 * k) ∧
    s4 = Nat.Prime (a + 6*b + 1) ∧
    (s1 ∧ s2 ∧ s3 ∧ ¬s4 ∨
     s1 ∧ s2 ∧ ¬s3 ∧ s4 ∨
     s1 ∧ ¬s2 ∧ s3 ∧ s4 ∨
     ¬s1 ∧ s2 ∧ s3 ∧ s4)) →
  ((a = 6 ∧ b = 1) ∨ (a = 18 ∧ b = 7)) := by
sorry

end NUMINAMATH_CALUDE_natural_number_pairs_l3411_341159


namespace NUMINAMATH_CALUDE_jackie_breaks_l3411_341141

/-- Calculates the number of breaks Jackie takes during push-ups -/
def number_of_breaks (pushups_per_10_seconds : ℕ) (pushups_with_breaks : ℕ) (break_duration : ℕ) : ℕ :=
  let pushups_per_minute : ℕ := pushups_per_10_seconds * 6
  let missed_pushups : ℕ := pushups_per_minute - pushups_with_breaks
  let time_not_pushing : ℕ := missed_pushups * 10 / pushups_per_10_seconds
  time_not_pushing / break_duration

theorem jackie_breaks :
  number_of_breaks 5 22 8 = 2 :=
by sorry

end NUMINAMATH_CALUDE_jackie_breaks_l3411_341141


namespace NUMINAMATH_CALUDE_circle_center_theorem_l3411_341184

-- Define the parabola
def parabola (x : ℝ) : ℝ := x^2

-- Define the circle
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the conditions
def circle_passes_through (c : Circle) (p : ℝ × ℝ) : Prop :=
  let (x, y) := p
  let (cx, cy) := c.center
  (x - cx)^2 + (y - cy)^2 = c.radius^2

def circle_tangent_to_parabola (c : Circle) (p : ℝ × ℝ) : Prop :=
  let (x, y) := p
  let (cx, cy) := c.center
  y = parabola x ∧
  (x - cx)^2 + (y - cy)^2 = c.radius^2 ∧
  (2 * x * (x - cx) + 2 * (y - cy))^2 = 4 * ((x - cx)^2 + (y - cy)^2)

-- Theorem statement
theorem circle_center_theorem :
  ∃ (c : Circle),
    circle_passes_through c (0, 1) ∧
    circle_tangent_to_parabola c (2, 4) ∧
    c.center = (-16/5, 53/10) :=
sorry

end NUMINAMATH_CALUDE_circle_center_theorem_l3411_341184


namespace NUMINAMATH_CALUDE_strawberry_harvest_l3411_341142

/-- Represents the dimensions of a rectangular garden -/
structure GardenDimensions where
  length : ℝ
  width : ℝ

/-- Calculates the area of a rectangular garden -/
def gardenArea (d : GardenDimensions) : ℝ :=
  d.length * d.width

/-- Calculates the total number of plants in the garden -/
def totalPlants (d : GardenDimensions) (density : ℝ) : ℝ :=
  gardenArea d * density

/-- Calculates the total number of strawberries harvested -/
def totalStrawberries (d : GardenDimensions) (density : ℝ) (yield : ℝ) : ℝ :=
  totalPlants d density * yield

/-- Theorem: The total number of strawberries harvested is 5400 -/
theorem strawberry_harvest (d : GardenDimensions) (density : ℝ) (yield : ℝ)
    (h1 : d.length = 10)
    (h2 : d.width = 9)
    (h3 : density = 5)
    (h4 : yield = 12) :
    totalStrawberries d density yield = 5400 := by
  sorry

#eval totalStrawberries ⟨10, 9⟩ 5 12

end NUMINAMATH_CALUDE_strawberry_harvest_l3411_341142


namespace NUMINAMATH_CALUDE_parabola_sum_l3411_341182

def parabola (a b c : ℝ) (y : ℝ) : ℝ := a * y^2 + b * y + c

theorem parabola_sum (a b c : ℝ) :
  parabola a b c (-6) = 7 →
  parabola a b c (-4) = 5 →
  a + b + c = -35/2 := by sorry

end NUMINAMATH_CALUDE_parabola_sum_l3411_341182


namespace NUMINAMATH_CALUDE_square_with_prime_quotient_and_remainder_four_l3411_341158

theorem square_with_prime_quotient_and_remainder_four (n : ℕ) : 
  (n ^ 2 % 11 = 4 ∧ Nat.Prime ((n ^ 2 - 4) / 11)) ↔ n = 9 :=
sorry

end NUMINAMATH_CALUDE_square_with_prime_quotient_and_remainder_four_l3411_341158


namespace NUMINAMATH_CALUDE_students_in_grade_6_l3411_341152

/-- The number of students in Grade 6 at Evergreen Elementary School -/
theorem students_in_grade_6 (total : ℕ) (grade4 : ℕ) (grade5 : ℕ) 
  (h_total : total = 100)
  (h_grade4 : grade4 = 30)
  (h_grade5 : grade5 = 35) :
  total - (grade4 + grade5) = 35 := by
  sorry

end NUMINAMATH_CALUDE_students_in_grade_6_l3411_341152


namespace NUMINAMATH_CALUDE_ball_arrangements_count_l3411_341180

def num_red_balls : ℕ := 2
def num_yellow_balls : ℕ := 3
def num_white_balls : ℕ := 4
def total_balls : ℕ := num_red_balls + num_yellow_balls + num_white_balls

theorem ball_arrangements_count :
  (Nat.factorial total_balls) / (Nat.factorial num_red_balls * Nat.factorial num_yellow_balls * Nat.factorial num_white_balls) = 1260 := by
  sorry

end NUMINAMATH_CALUDE_ball_arrangements_count_l3411_341180


namespace NUMINAMATH_CALUDE_modified_cube_edges_l3411_341123

/-- Represents a modified cube -/
structure ModifiedCube where
  sideLength : ℕ
  smallCubeRemoved1 : ℕ
  smallCubeRemoved2 : ℕ
  largeCubeRemoved : ℕ

/-- Calculates the number of edges in a modified cube -/
def edgeCount (c : ModifiedCube) : ℕ := sorry

/-- Theorem stating that a specific modified cube has 22 edges -/
theorem modified_cube_edges :
  let c : ModifiedCube := {
    sideLength := 4,
    smallCubeRemoved1 := 1,
    smallCubeRemoved2 := 1,
    largeCubeRemoved := 2
  }
  edgeCount c = 22 := by sorry

end NUMINAMATH_CALUDE_modified_cube_edges_l3411_341123


namespace NUMINAMATH_CALUDE_arithmetic_sequence_property_l3411_341188

/-- An arithmetic sequence is a sequence where the difference between
    any two consecutive terms is constant. -/
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- Given an arithmetic sequence a where a₂ + a₇ = 6,
    prove that 3a₄ + a₆ = 12 -/
theorem arithmetic_sequence_property (a : ℕ → ℝ)
    (h_arithmetic : is_arithmetic_sequence a)
    (h_sum : a 2 + a 7 = 6) :
  3 * a 4 + a 6 = 12 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_property_l3411_341188


namespace NUMINAMATH_CALUDE_sheep_buying_problem_sheep_buying_problem_unique_l3411_341128

/-- The number of people buying the sheep -/
def num_people : ℕ := 21

/-- The price of the sheep in coins -/
def sheep_price : ℕ := 150

/-- Theorem stating the solution to the sheep-buying problem -/
theorem sheep_buying_problem :
  (∃ (n : ℕ) (p : ℕ),
    n = num_people ∧
    p = sheep_price ∧
    5 * n + 45 = p ∧
    7 * n + 3 = p) :=
by sorry

/-- Theorem proving the uniqueness of the solution -/
theorem sheep_buying_problem_unique :
  ∀ (n : ℕ) (p : ℕ),
    5 * n + 45 = p ∧
    7 * n + 3 = p →
    n = num_people ∧
    p = sheep_price :=
by sorry

end NUMINAMATH_CALUDE_sheep_buying_problem_sheep_buying_problem_unique_l3411_341128


namespace NUMINAMATH_CALUDE_linear_equation_with_integer_roots_l3411_341196

theorem linear_equation_with_integer_roots 
  (m : ℤ) (n : ℕ) 
  (h1 : m ≠ 1) 
  (h2 : n = 1) 
  (h3 : ∃ x : ℤ, (m - 1) * x - 3 = 0) :
  m = -2 ∨ m = 0 ∨ m = 2 ∨ m = 4 := by
sorry

end NUMINAMATH_CALUDE_linear_equation_with_integer_roots_l3411_341196


namespace NUMINAMATH_CALUDE_average_rope_length_l3411_341140

theorem average_rope_length (rope1 rope2 : ℝ) (h1 : rope1 = 2) (h2 : rope2 = 6) :
  (rope1 + rope2) / 2 = 4 := by
  sorry

end NUMINAMATH_CALUDE_average_rope_length_l3411_341140


namespace NUMINAMATH_CALUDE_certain_number_proof_l3411_341115

theorem certain_number_proof (A B C X : ℝ) : 
  A / B = 5 / 6 →
  B / C = 6 / 8 →
  C = 42 →
  A + C = B + X →
  X = 36.75 := by
sorry

end NUMINAMATH_CALUDE_certain_number_proof_l3411_341115


namespace NUMINAMATH_CALUDE_instantaneous_rate_of_change_at_zero_l3411_341118

noncomputable def f (x : ℝ) : ℝ := 2 * Real.exp (Real.sin x)

theorem instantaneous_rate_of_change_at_zero :
  deriv f 0 = 2 * Real.exp 0 := by
  sorry

end NUMINAMATH_CALUDE_instantaneous_rate_of_change_at_zero_l3411_341118


namespace NUMINAMATH_CALUDE_charge_per_mile_calculation_l3411_341130

/-- Proves that the charge per mile is $0.25 given the rental fee, total amount paid, and miles driven -/
theorem charge_per_mile_calculation (rental_fee total_paid miles_driven : ℚ) 
  (h1 : rental_fee = 20.99)
  (h2 : total_paid = 95.74)
  (h3 : miles_driven = 299) :
  (total_paid - rental_fee) / miles_driven = 0.25 := by
  sorry

end NUMINAMATH_CALUDE_charge_per_mile_calculation_l3411_341130


namespace NUMINAMATH_CALUDE_zoo_sea_lions_l3411_341107

theorem zoo_sea_lions (sea_lions : ℕ) (penguins : ℕ) : 
  (sea_lions : ℚ) / penguins = 4 / 11 →
  penguins = sea_lions + 84 →
  sea_lions = 48 := by
sorry

end NUMINAMATH_CALUDE_zoo_sea_lions_l3411_341107


namespace NUMINAMATH_CALUDE_negation_equivalence_l3411_341122

theorem negation_equivalence :
  (¬ ∃ x : ℝ, x^3 - x^2 + 1 > 0) ↔ (∀ x : ℝ, x^3 - x^2 + 1 ≤ 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_equivalence_l3411_341122


namespace NUMINAMATH_CALUDE_f_increasing_when_a_1_m_range_l3411_341156

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^3 + a*x^2 + a*x

-- Define monotonically increasing
def monotonically_increasing (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x < y → f x < f y

-- Theorem 1: f is monotonically increasing when a = 1
theorem f_increasing_when_a_1 :
  monotonically_increasing (f 1) := by sorry

-- Define the necessary but not sufficient condition
def necessary_not_sufficient (m : ℝ) : Prop :=
  (∀ a : ℝ, monotonically_increasing (f a) → |a - 1| ≤ m) ∧
  (∃ a : ℝ, |a - 1| ≤ m ∧ ¬monotonically_increasing (f a))

-- Theorem 2: The range of m is [0,1)
theorem m_range :
  ∀ m : ℝ, (m > 0 ∧ necessary_not_sufficient m) ↔ (0 ≤ m ∧ m < 1) := by sorry

end NUMINAMATH_CALUDE_f_increasing_when_a_1_m_range_l3411_341156


namespace NUMINAMATH_CALUDE_square_of_three_times_sqrt_two_l3411_341100

theorem square_of_three_times_sqrt_two : (3 * Real.sqrt 2) ^ 2 = 18 := by
  sorry

end NUMINAMATH_CALUDE_square_of_three_times_sqrt_two_l3411_341100


namespace NUMINAMATH_CALUDE_prob_A_given_B_value_l3411_341155

/-- The number of people visiting tourist spots -/
def num_people : ℕ := 4

/-- The number of tourist spots -/
def num_spots : ℕ := 4

/-- Event A: All 4 people visit different spots -/
def event_A : Prop := True

/-- Event B: Xiao Zhao visits a spot alone -/
def event_B : Prop := True

/-- The number of ways for 3 people to visit 3 spots -/
def ways_3_people_3_spots : ℕ := 3 * 3 * 3

/-- The number of ways for Xiao Zhao to visit a spot alone -/
def ways_xiao_zhao_alone : ℕ := num_spots * ways_3_people_3_spots

/-- The number of ways 4 people can visit different spots -/
def ways_all_different : ℕ := 4 * 3 * 2 * 1

/-- The probability of event A given event B -/
def prob_A_given_B : ℚ := ways_all_different / ways_xiao_zhao_alone

theorem prob_A_given_B_value : prob_A_given_B = 2 / 9 := by
  sorry

end NUMINAMATH_CALUDE_prob_A_given_B_value_l3411_341155


namespace NUMINAMATH_CALUDE_correct_calculation_result_l3411_341120

theorem correct_calculation_result (x : ℚ) : 
  (x * 6 = 96) → (x / 8 = 2) := by
  sorry

end NUMINAMATH_CALUDE_correct_calculation_result_l3411_341120


namespace NUMINAMATH_CALUDE_unique_solution_l3411_341143

/-- A function from positive reals to positive reals -/
def PositiveFunction := {f : ℝ → ℝ // ∀ x, x > 0 → f x > 0}

/-- The functional equation -/
def SatisfiesEquation (f : PositiveFunction) (c : ℝ) : Prop :=
  c > 0 ∧ ∀ x y, x > 0 → y > 0 → f.val ((c + 1) * x + f.val y) = f.val (x + 2 * y) + 2 * c * x

/-- The theorem statement -/
theorem unique_solution (f : PositiveFunction) (c : ℝ) 
  (h : SatisfiesEquation f c) : 
  ∀ x, x > 0 → f.val x = 2 * x :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_l3411_341143


namespace NUMINAMATH_CALUDE_same_terminal_side_l3411_341181

theorem same_terminal_side (k : ℤ) : ∃ k, (11 * π) / 6 = 2 * k * π - π / 6 := by
  sorry

end NUMINAMATH_CALUDE_same_terminal_side_l3411_341181


namespace NUMINAMATH_CALUDE_interview_probabilities_l3411_341160

structure InterviewScenario where
  prob_A_pass : ℝ
  prob_B_pass : ℝ
  prob_C_pass : ℝ

def exactly_one_pass (s : InterviewScenario) : ℝ :=
  s.prob_A_pass * (1 - s.prob_B_pass) * (1 - s.prob_C_pass) +
  (1 - s.prob_A_pass) * s.prob_B_pass * (1 - s.prob_C_pass) +
  (1 - s.prob_A_pass) * (1 - s.prob_B_pass) * s.prob_C_pass

def at_most_one_sign (s : InterviewScenario) : ℝ :=
  (1 - s.prob_A_pass) * (1 - s.prob_B_pass * s.prob_C_pass) +
  s.prob_A_pass * (1 - s.prob_B_pass * s.prob_C_pass)

theorem interview_probabilities (s : InterviewScenario) 
  (h1 : s.prob_A_pass = 1/4)
  (h2 : s.prob_B_pass = 1/3)
  (h3 : s.prob_C_pass = 1/3) :
  exactly_one_pass s = 4/9 ∧ at_most_one_sign s = 8/9 := by
  sorry

#check interview_probabilities

end NUMINAMATH_CALUDE_interview_probabilities_l3411_341160


namespace NUMINAMATH_CALUDE_successive_discounts_equivalence_l3411_341170

/-- Proves that three successive discounts are equivalent to a single discount --/
theorem successive_discounts_equivalence : 
  let original_price : ℝ := 800
  let discount1 : ℝ := 0.15
  let discount2 : ℝ := 0.10
  let discount3 : ℝ := 0.05
  let final_price : ℝ := original_price * (1 - discount1) * (1 - discount2) * (1 - discount3)
  let single_discount : ℝ := 0.27325
  final_price = original_price * (1 - single_discount) := by
  sorry

#check successive_discounts_equivalence

end NUMINAMATH_CALUDE_successive_discounts_equivalence_l3411_341170


namespace NUMINAMATH_CALUDE_spinster_cat_problem_l3411_341173

theorem spinster_cat_problem (S C : ℕ) : 
  S * 9 = C * 2 →  -- Ratio of spinsters to cats is 2:9
  C = S + 42 →     -- There are 42 more cats than spinsters
  S = 12           -- The number of spinsters is 12
:= by sorry

end NUMINAMATH_CALUDE_spinster_cat_problem_l3411_341173


namespace NUMINAMATH_CALUDE_arithmetic_sequence_properties_l3411_341191

def arithmetic_sequence (a : ℕ → ℚ) : Prop :=
  ∃ d : ℚ, ∀ n : ℕ, a (n + 1) - a n = d

def S (a : ℕ → ℚ) (n : ℕ) : ℚ :=
  (n : ℚ) * (a 1 + a n) / 2

theorem arithmetic_sequence_properties
  (a : ℕ → ℚ)
  (h_arith : arithmetic_sequence a)
  (h_a5 : a 5 = -3)
  (h_S5 : S a 5 = 0) :
  (∀ n : ℕ, a n = (3 * (3 - n : ℚ)) / 2) ∧
  (∀ n : ℕ, a n * S a n < 0 ↔ n = 4) :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_properties_l3411_341191


namespace NUMINAMATH_CALUDE_imaginary_unit_power_2016_l3411_341153

theorem imaginary_unit_power_2016 (i : ℂ) (h : i^2 = -1) : i^2016 = 1 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_unit_power_2016_l3411_341153


namespace NUMINAMATH_CALUDE_division_theorem_l3411_341179

theorem division_theorem (dividend divisor remainder quotient : ℕ) : 
  dividend = 141 →
  divisor = 17 →
  remainder = 5 →
  dividend = divisor * quotient + remainder →
  quotient = 8 := by
sorry

end NUMINAMATH_CALUDE_division_theorem_l3411_341179


namespace NUMINAMATH_CALUDE_book_length_is_300_l3411_341103

/-- The length of a book in pages -/
def book_length : ℕ := 300

/-- The fraction of the book Soja has finished reading -/
def finished_fraction : ℚ := 2/3

/-- The difference between pages read and pages left to read -/
def pages_difference : ℕ := 100

/-- Theorem stating that the book length is 300 pages -/
theorem book_length_is_300 : 
  book_length = 300 ∧ 
  finished_fraction * book_length - (1 - finished_fraction) * book_length = pages_difference := by
  sorry

end NUMINAMATH_CALUDE_book_length_is_300_l3411_341103


namespace NUMINAMATH_CALUDE_cool_drink_volume_cool_drink_volume_proof_l3411_341164

theorem cool_drink_volume : ℝ → Prop :=
  fun initial_volume =>
    let initial_jasmine_ratio := 0.1
    let added_jasmine := 8
    let added_water := 12
    let final_jasmine_ratio := 0.16
    let final_volume := initial_volume + added_jasmine + added_water
    initial_jasmine_ratio * initial_volume + added_jasmine = final_jasmine_ratio * final_volume →
    initial_volume = 80

theorem cool_drink_volume_proof : ∃ (v : ℝ), cool_drink_volume v :=
  sorry

end NUMINAMATH_CALUDE_cool_drink_volume_cool_drink_volume_proof_l3411_341164


namespace NUMINAMATH_CALUDE_chocolate_eggs_duration_l3411_341125

/-- The number of chocolate eggs Maddy has -/
def N : ℕ := 40

/-- The number of eggs Maddy eats per weekday -/
def eggs_per_day : ℕ := 2

/-- The number of weekdays in a week -/
def weekdays : ℕ := 5

/-- The number of weeks the chocolate eggs will last -/
def weeks_lasted : ℕ := N / (eggs_per_day * weekdays)

theorem chocolate_eggs_duration : weeks_lasted = 4 := by
  sorry

end NUMINAMATH_CALUDE_chocolate_eggs_duration_l3411_341125


namespace NUMINAMATH_CALUDE_tangent_line_equation_l3411_341171

-- Define the function f(x) = x^2
def f (x : ℝ) : ℝ := x^2

-- Define the derivative of f
def f' (x : ℝ) : ℝ := 2 * x

-- Theorem statement
theorem tangent_line_equation :
  ∃ (m b : ℝ), 
    (∀ x y : ℝ, y = m * x + b ↔ m * x - y + b = 0) ∧
    (m = f' 1) ∧
    (f 1 = m * 1 + b) ∧
    (m * 1 - f 1 + b = 0) ∧
    (m = 2 ∧ b = -1) := by
  sorry

end NUMINAMATH_CALUDE_tangent_line_equation_l3411_341171


namespace NUMINAMATH_CALUDE_range_of_m_l3411_341110

-- Define propositions p and q
def p (m : ℝ) : Prop := ∀ x > 0, m^2 + 2*m - 1 ≤ x + 1/x

def q (m : ℝ) : Prop := ∀ x, (5 - m^2)^x < (5 - m^2)^(x + 1)

-- Define the theorem
theorem range_of_m (m : ℝ) :
  (p m ∨ q m) ∧ ¬(p m ∧ q m) →
  (-3 ≤ m ∧ m ≤ -2) ∨ (1 < m ∧ m < 2) :=
by sorry

end NUMINAMATH_CALUDE_range_of_m_l3411_341110


namespace NUMINAMATH_CALUDE_transformation_is_rotation_and_scaling_l3411_341112

def rotation_90 : Matrix (Fin 2) (Fin 2) ℝ := !![0, -1; 1, 0]
def scaling_2 : Matrix (Fin 2) (Fin 2) ℝ := !![2, 0; 0, 2]
def transformation : Matrix (Fin 2) (Fin 2) ℝ := !![0, -2; 2, 0]

theorem transformation_is_rotation_and_scaling :
  transformation = scaling_2 * rotation_90 :=
sorry

end NUMINAMATH_CALUDE_transformation_is_rotation_and_scaling_l3411_341112


namespace NUMINAMATH_CALUDE_system_nonzero_solution_iff_condition_l3411_341198

/-- The system of equations has a non-zero solution iff 2abc + ab + bc + ca - 1 = 0 -/
theorem system_nonzero_solution_iff_condition (a b c : ℝ) :
  (∃ x y z : ℝ, (x ≠ 0 ∨ y ≠ 0 ∨ z ≠ 0) ∧
    x = b * y + c * z ∧
    y = c * z + a * x ∧
    z = a * x + b * y) ↔
  2 * a * b * c + a * b + b * c + c * a - 1 = 0 := by
sorry

end NUMINAMATH_CALUDE_system_nonzero_solution_iff_condition_l3411_341198


namespace NUMINAMATH_CALUDE_logarithm_equality_l3411_341113

-- Define the common logarithm (base 10)
noncomputable def lg (x : ℝ) := Real.log x / Real.log 10

-- State the theorem
theorem logarithm_equality :
  lg (25 / 16) - 2 * lg (5 / 9) + lg (32 / 81) = lg 2 := by
  sorry

end NUMINAMATH_CALUDE_logarithm_equality_l3411_341113


namespace NUMINAMATH_CALUDE_simplify_expression_l3411_341102

theorem simplify_expression (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) :
  (x^2 + y^2)⁻¹ * (x⁻¹ + y⁻¹) = (x^3*y + x*y^3)⁻¹ * (x + y) :=
by sorry

end NUMINAMATH_CALUDE_simplify_expression_l3411_341102


namespace NUMINAMATH_CALUDE_twins_age_product_difference_l3411_341133

theorem twins_age_product_difference (current_age : ℕ) (h : current_age = 8) : 
  (current_age + 1) * (current_age + 1) - current_age * current_age = 17 := by
  sorry

end NUMINAMATH_CALUDE_twins_age_product_difference_l3411_341133


namespace NUMINAMATH_CALUDE_inequality_solution_set_l3411_341145

def solution_set : Set ℝ := {x | 2 ≤ x ∧ x ≤ 3}

theorem inequality_solution_set : 
  ∀ x : ℝ, x ∈ solution_set ↔ x^2 - 5*x + 6 ≤ 0 := by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l3411_341145


namespace NUMINAMATH_CALUDE_perpendicular_implies_parallel_l3411_341146

structure Space where
  Line : Type
  Plane : Type

variable (S : Space)

-- Define perpendicular relation between a line and a plane
def perpendicular (l : S.Line) (p : S.Plane) : Prop :=
  sorry

-- Define parallel relation between two planes
def parallel (p1 p2 : S.Plane) : Prop :=
  sorry

-- Theorem statement
theorem perpendicular_implies_parallel
  (a : S.Line) (α β : S.Plane)
  (h1 : perpendicular S a α)
  (h2 : perpendicular S a β) :
  parallel S α β :=
sorry

end NUMINAMATH_CALUDE_perpendicular_implies_parallel_l3411_341146


namespace NUMINAMATH_CALUDE_intersection_line_canonical_l3411_341175

-- Define the planes
def plane1 (x y z : ℝ) : Prop := 3 * x + 4 * y + 3 * z + 1 = 0
def plane2 (x y z : ℝ) : Prop := 2 * x - 4 * y - 2 * z + 4 = 0

-- Define the canonical form of the line
def canonical_line (x y z : ℝ) : Prop := (x + 1) / 4 = (y - 1/2) / 12 ∧ (y - 1/2) / 12 = z / (-20)

-- Theorem statement
theorem intersection_line_canonical : 
  ∀ x y z : ℝ, plane1 x y z ∧ plane2 x y z → canonical_line x y z :=
by
  sorry

end NUMINAMATH_CALUDE_intersection_line_canonical_l3411_341175


namespace NUMINAMATH_CALUDE_visit_either_not_both_l3411_341189

/-- The probability of visiting either Chile or Madagascar, but not both -/
theorem visit_either_not_both (p_chile p_madagascar : ℝ) 
  (h_chile : p_chile = 0.30)
  (h_madagascar : p_madagascar = 0.50) :
  p_chile + p_madagascar - p_chile * p_madagascar = 0.65 := by
  sorry

end NUMINAMATH_CALUDE_visit_either_not_both_l3411_341189


namespace NUMINAMATH_CALUDE_larger_solution_quadratic_l3411_341168

theorem larger_solution_quadratic : ∃ (x y : ℝ), x ≠ y ∧ 
  x^2 - 9*x - 22 = 0 ∧ 
  y^2 - 9*y - 22 = 0 ∧ 
  (∀ z : ℝ, z^2 - 9*z - 22 = 0 → z = x ∨ z = y) ∧
  max x y = 11 := by
sorry

end NUMINAMATH_CALUDE_larger_solution_quadratic_l3411_341168


namespace NUMINAMATH_CALUDE_octagon_cannot_tile_l3411_341111

/-- A regular polygon with n sides --/
structure RegularPolygon (n : ℕ) where
  sides : n ≥ 3

/-- The interior angle of a regular polygon with n sides --/
def interiorAngle (n : ℕ) (p : RegularPolygon n) : ℚ :=
  180 - (360 / n)

/-- A regular polygon can tile the plane if its interior angle divides 360° evenly --/
def canTilePlane (n : ℕ) (p : RegularPolygon n) : Prop :=
  ∃ k : ℕ, k * interiorAngle n p = 360

/-- The set of regular polygons we're considering --/
def consideredPolygons : Set (Σ n, RegularPolygon n) :=
  {⟨3, ⟨by norm_num⟩⟩, ⟨4, ⟨by norm_num⟩⟩, ⟨6, ⟨by norm_num⟩⟩, ⟨8, ⟨by norm_num⟩⟩}

theorem octagon_cannot_tile :
  ∀ p ∈ consideredPolygons, ¬(canTilePlane p.1 p.2) ↔ p.1 = 8 := by
  sorry

#check octagon_cannot_tile

end NUMINAMATH_CALUDE_octagon_cannot_tile_l3411_341111


namespace NUMINAMATH_CALUDE_line_L_equation_ellipse_C_equation_l3411_341117

-- Define the line L
def line_L (x y : ℝ) : Prop := x/4 + y/2 = 1

-- Define the ellipse C
def ellipse_C (x y : ℝ) : Prop := x^2/25 + y^2/9 = 1

-- Theorem for line L
theorem line_L_equation :
  ∀ (a b : ℝ), a > 0 ∧ b > 0 →
  (∀ (x y : ℝ), x/a + y/b = 1 → (x = 2 ∧ y = 1)) →
  (1/2 * a * b = 4) →
  (∀ (x y : ℝ), line_L x y ↔ x/a + y/b = 1) :=
sorry

-- Theorem for ellipse C
theorem ellipse_C_equation :
  let e : ℝ := 0.8
  let c : ℝ := 4
  let a : ℝ := c / e
  let b : ℝ := Real.sqrt (a^2 - c^2)
  ∀ (x y : ℝ), ellipse_C x y ↔ x^2/a^2 + y^2/b^2 = 1 :=
sorry

end NUMINAMATH_CALUDE_line_L_equation_ellipse_C_equation_l3411_341117


namespace NUMINAMATH_CALUDE_function_shape_is_graph_l3411_341132

/-- A function from real numbers to real numbers -/
def RealFunction := ℝ → ℝ

/-- A point in the Cartesian coordinate system -/
def CartesianPoint := ℝ × ℝ

/-- The set of all points representing a function in the Cartesian coordinate system -/
def FunctionPoints (f : RealFunction) : Set CartesianPoint :=
  {p : CartesianPoint | ∃ x : ℝ, p = (x, f x)}

/-- The graph of a function is the set of all points representing that function -/
def Graph (f : RealFunction) : Set CartesianPoint := FunctionPoints f

/-- Theorem: The shape formed by all points plotted in the Cartesian coordinate system 
    that represent a function is called the graph of the function -/
theorem function_shape_is_graph (f : RealFunction) : 
  FunctionPoints f = Graph f := by sorry

end NUMINAMATH_CALUDE_function_shape_is_graph_l3411_341132


namespace NUMINAMATH_CALUDE_absolute_value_inequality_l3411_341178

theorem absolute_value_inequality (a b : ℝ) (h : |a - b| > 2) :
  ∀ x : ℝ, |x - a| + |x - b| > 2 :=
by sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_l3411_341178


namespace NUMINAMATH_CALUDE_common_root_value_l3411_341127

-- Define the polynomials
def poly1 (x C : ℝ) : ℝ := x^3 + C*x^2 + 15
def poly2 (x D : ℝ) : ℝ := x^3 + D*x + 35

-- Theorem statement
theorem common_root_value (C D : ℝ) :
  ∃ (p : ℝ), 
    (poly1 p C = 0 ∧ poly2 p D = 0) ∧ 
    (∃ (q r : ℝ), p * q * r = -15) ∧
    (∃ (s t : ℝ), p * s * t = -35) →
    p = Real.rpow 525 (1/3) := by
  sorry

end NUMINAMATH_CALUDE_common_root_value_l3411_341127


namespace NUMINAMATH_CALUDE_symmetric_circle_equation_l3411_341144

/-- Given a circle with equation x²+y²-4x=0, this theorem states that 
    the equation of the circle symmetric to it with respect to the line y=x 
    is x²+y²-4y=0 -/
theorem symmetric_circle_equation (x y : ℝ) : 
  (∀ x y, x^2 + y^2 - 4*x = 0 → (x^2 + y^2 - 4*y = 0 ↔ 
    ∃ x' y', x'^2 + y'^2 - 4*x' = 0 ∧ x = y' ∧ y = x')) := by
  sorry

end NUMINAMATH_CALUDE_symmetric_circle_equation_l3411_341144


namespace NUMINAMATH_CALUDE_inequality_implies_equality_l3411_341199

theorem inequality_implies_equality (a b : ℝ) 
  (h_pos_a : a > 0) (h_pos_b : b > 0)
  (h_ineq : Real.log a + Real.log (b^2) ≥ 2*a + b^2/2 - 2) :
  a - 2*b = 1/2 - 2*Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_inequality_implies_equality_l3411_341199


namespace NUMINAMATH_CALUDE_product_25_sum_0_l3411_341134

theorem product_25_sum_0 (a b c d : ℤ) : 
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d → a * b * c * d = 25 → a + b + c + d = 0 := by
  sorry

end NUMINAMATH_CALUDE_product_25_sum_0_l3411_341134


namespace NUMINAMATH_CALUDE_diamond_equation_solution_l3411_341177

-- Define the binary operation ◇
noncomputable def diamond (a b : ℝ) : ℝ := a / b

-- State the theorem
theorem diamond_equation_solution :
  (∀ (a b c : ℝ), a ≠ 0 → b ≠ 0 → c ≠ 0 → diamond a (diamond b c) = diamond (diamond a b) c) →
  (∀ (a : ℝ), a ≠ 0 → diamond a a = 1) →
  diamond 504 (diamond 12 (25 / 21)) = 50 := by
  sorry

end NUMINAMATH_CALUDE_diamond_equation_solution_l3411_341177


namespace NUMINAMATH_CALUDE_least_positive_integer_with_congruences_l3411_341104

theorem least_positive_integer_with_congruences : ∃ (b : ℕ), 
  b > 0 ∧ 
  b % 3 = 2 ∧ 
  b % 5 = 4 ∧ 
  b % 6 = 5 ∧ 
  b % 7 = 6 ∧ 
  (∀ (x : ℕ), x > 0 ∧ x % 3 = 2 ∧ x % 5 = 4 ∧ x % 6 = 5 ∧ x % 7 = 6 → x ≥ b) ∧
  b = 209 :=
by sorry

end NUMINAMATH_CALUDE_least_positive_integer_with_congruences_l3411_341104


namespace NUMINAMATH_CALUDE_population_average_age_l3411_341137

theorem population_average_age
  (ratio_women_men : ℚ)
  (avg_age_women : ℚ)
  (avg_age_men : ℚ)
  (h_ratio : ratio_women_men = 10 / 9)
  (h_women_age : avg_age_women = 36)
  (h_men_age : avg_age_men = 33) :
  (ratio_women_men * avg_age_women + avg_age_men) / (ratio_women_men + 1) = 34 + 13 / 19 :=
by sorry

end NUMINAMATH_CALUDE_population_average_age_l3411_341137


namespace NUMINAMATH_CALUDE_unique_modular_solution_l3411_341131

theorem unique_modular_solution : ∃! n : ℤ, 0 ≤ n ∧ n ≤ 10 ∧ n ≡ 50000 [ZMOD 11] ∧ n = 5 := by
  sorry

end NUMINAMATH_CALUDE_unique_modular_solution_l3411_341131


namespace NUMINAMATH_CALUDE_f_maximized_at_three_tenths_l3411_341161

/-- The probability that exactly k out of n items are defective, given probability p for each item -/
def binomial_probability (n k : ℕ) (p : ℝ) : ℝ :=
  (n.choose k) * p^k * (1 - p)^(n - k)

/-- The probability that exactly 3 out of 10 items are defective -/
def f (p : ℝ) : ℝ := binomial_probability 10 3 p

/-- The theorem stating that f(p) is maximized when p = 3/10 -/
theorem f_maximized_at_three_tenths (p : ℝ) (h1 : 0 < p) (h2 : p < 1) :
  ∃ (max_p : ℝ), max_p = 3/10 ∧ ∀ q, 0 < q → q < 1 → f q ≤ f max_p :=
sorry

end NUMINAMATH_CALUDE_f_maximized_at_three_tenths_l3411_341161


namespace NUMINAMATH_CALUDE_ratio_simplification_l3411_341192

theorem ratio_simplification (A B C : ℚ) : 
  (A / B = 5 / 3 / (29 / 6)) → 
  (C / A = (11 / 5) / (11 / 3)) → 
  ∃ (k : ℚ), k * A = 10 ∧ k * B = 29 ∧ k * C = 6 :=
by sorry

end NUMINAMATH_CALUDE_ratio_simplification_l3411_341192


namespace NUMINAMATH_CALUDE_intercepts_satisfy_equation_intercepts_unique_l3411_341187

/-- The line equation -/
def line_equation (x y : ℝ) : Prop := 5 * x - 2 * y - 10 = 0

/-- The x-intercept of the line -/
def x_intercept : ℝ := 2

/-- The y-intercept of the line -/
def y_intercept : ℝ := -5

/-- Theorem stating that the x-intercept and y-intercept satisfy the line equation -/
theorem intercepts_satisfy_equation : 
  line_equation x_intercept 0 ∧ line_equation 0 y_intercept := by
  sorry

/-- Theorem stating that the x-intercept and y-intercept are unique -/
theorem intercepts_unique :
  ∀ x y : ℝ, line_equation x 0 → x = x_intercept ∧
  ∀ x y : ℝ, line_equation 0 y → y = y_intercept := by
  sorry

end NUMINAMATH_CALUDE_intercepts_satisfy_equation_intercepts_unique_l3411_341187


namespace NUMINAMATH_CALUDE_tan_two_implies_sum_l3411_341147

theorem tan_two_implies_sum (θ : Real) (h : Real.tan θ = 2) :
  (Real.sin θ + Real.cos θ) / Real.sin θ + Real.sin θ ^ 2 = 23 / 10 := by
  sorry

end NUMINAMATH_CALUDE_tan_two_implies_sum_l3411_341147


namespace NUMINAMATH_CALUDE_max_distance_to_line_l3411_341106

/-- Given a line ax + by + c = 0 where a, b, and c form an arithmetic sequence,
    the maximum distance from the origin (0, 0) to this line is √5. -/
theorem max_distance_to_line (a b c : ℝ) :
  (a + c = 2 * b) →  -- arithmetic sequence condition
  (∃ (x y : ℝ), a * x + b * y + c = 0) →  -- line exists
  (∀ (x y : ℝ), a * x + b * y + c = 0 → (x^2 + y^2 : ℝ) ≤ 5) ∧
  (∃ (x y : ℝ), a * x + b * y + c = 0 ∧ x^2 + y^2 = 5) :=
by sorry

end NUMINAMATH_CALUDE_max_distance_to_line_l3411_341106


namespace NUMINAMATH_CALUDE_polynomial_sum_l3411_341119

-- Define the polynomials
def f (x : ℝ) : ℝ := 2*x^3 - 4*x^2 + 2*x - 5
def g (x : ℝ) : ℝ := -3*x^2 + 4*x - 7
def h (x : ℝ) : ℝ := 6*x^3 + x^2 + 3*x + 2

-- State the theorem
theorem polynomial_sum :
  ∀ x : ℝ, f x + g x + h x = 8*x^3 - 6*x^2 + 9*x - 10 :=
by
  sorry

end NUMINAMATH_CALUDE_polynomial_sum_l3411_341119


namespace NUMINAMATH_CALUDE_first_discount_percentage_l3411_341176

theorem first_discount_percentage
  (list_price : ℝ)
  (final_price : ℝ)
  (second_discount : ℝ)
  (h1 : list_price = 70)
  (h2 : final_price = 61.74)
  (h3 : second_discount = 0.01999999999999997)
  : ∃ (first_discount : ℝ),
    first_discount = 0.1 ∧
    final_price = list_price * (1 - first_discount) * (1 - second_discount) :=
by sorry

end NUMINAMATH_CALUDE_first_discount_percentage_l3411_341176


namespace NUMINAMATH_CALUDE_limit_proof_l3411_341139

open Real

theorem limit_proof (ε : ℝ) (hε : ε > 0) :
  ∃ δ : ℝ, δ > 0 ∧
  ∀ x : ℝ, 0 < |x + 7/2| ∧ |x + 7/2| < δ →
    |(2*x^2 + 13*x + 21) / (2*x + 7) + 1/2| < ε :=
by
  sorry

end NUMINAMATH_CALUDE_limit_proof_l3411_341139


namespace NUMINAMATH_CALUDE_inequality_for_three_positives_l3411_341157

theorem inequality_for_three_positives (x₁ x₂ x₃ : ℝ) (h₁ : x₁ > 0) (h₂ : x₂ > 0) (h₃ : x₃ > 0) :
  (x₁ * x₂ / x₃) + (x₂ * x₃ / x₁) + (x₃ * x₁ / x₂) ≥ x₁ + x₂ + x₃ ∧
  ((x₁ * x₂ / x₃) + (x₂ * x₃ / x₁) + (x₃ * x₁ / x₂) = x₁ + x₂ + x₃ ↔ x₁ = x₂ ∧ x₂ = x₃) :=
by sorry

end NUMINAMATH_CALUDE_inequality_for_three_positives_l3411_341157


namespace NUMINAMATH_CALUDE_bananas_in_E_l3411_341138

/-- The number of baskets -/
def num_baskets : ℕ := 5

/-- The average number of fruits per basket -/
def avg_fruits_per_basket : ℕ := 25

/-- The number of fruits in basket A -/
def fruits_in_A : ℕ := 15

/-- The number of fruits in basket B -/
def fruits_in_B : ℕ := 30

/-- The number of fruits in basket C -/
def fruits_in_C : ℕ := 20

/-- The number of fruits in basket D -/
def fruits_in_D : ℕ := 25

/-- Theorem: The number of bananas in basket E is 35 -/
theorem bananas_in_E : 
  num_baskets * avg_fruits_per_basket - (fruits_in_A + fruits_in_B + fruits_in_C + fruits_in_D) = 35 := by
  sorry

end NUMINAMATH_CALUDE_bananas_in_E_l3411_341138


namespace NUMINAMATH_CALUDE_trigonometric_equality_l3411_341166

theorem trigonometric_equality (θ : Real) (h : Real.sin (3 * Real.pi + θ) = 1/2) :
  (Real.cos (3 * Real.pi + θ)) / (Real.cos θ * (Real.cos (Real.pi + θ) - 1)) +
  (Real.cos (θ - 4 * Real.pi)) / (Real.cos (θ + 2 * Real.pi) * Real.cos (3 * Real.pi + θ) + Real.cos (-θ)) = 8 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_equality_l3411_341166


namespace NUMINAMATH_CALUDE_tangent_line_equation_l3411_341165

/-- The curve function -/
def f (x : ℝ) : ℝ := 2 * x^2 + 1

/-- The derivative of the curve function -/
def f' (x : ℝ) : ℝ := 4 * x

/-- The point of tangency -/
def P : ℝ × ℝ := (-1, 3)

/-- The slope of the tangent line at P -/
def k : ℝ := f' P.1

/-- The equation of the tangent line -/
def tangent_line (x y : ℝ) : Prop := 4 * x + y + 1 = 0

theorem tangent_line_equation :
  ∀ x y : ℝ, (x, y) ∈ {(x, y) | tangent_line x y} ↔
  y - P.2 = k * (x - P.1) ∧ y = f x := by sorry

end NUMINAMATH_CALUDE_tangent_line_equation_l3411_341165


namespace NUMINAMATH_CALUDE_star_problem_l3411_341174

-- Define the ⭐ operation
def star (x y : ℚ) : ℚ := (x + y) / 4

-- Theorem statement
theorem star_problem : star (star 3 9) 4 = 7 / 4 := by sorry

end NUMINAMATH_CALUDE_star_problem_l3411_341174


namespace NUMINAMATH_CALUDE_sin_cos_sixth_power_sum_l3411_341190

theorem sin_cos_sixth_power_sum (θ : Real) (h : Real.sin (2 * θ) = 1/3) :
  Real.sin θ ^ 6 + Real.cos θ ^ 6 = 11/12 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_sixth_power_sum_l3411_341190


namespace NUMINAMATH_CALUDE_card_arrangement_count_l3411_341121

/-- Represents a board with a given number of cells -/
structure Board :=
  (cells : Nat)

/-- Represents a set of cards with a given count -/
structure CardSet :=
  (count : Nat)

/-- Calculates the number of possible arrangements of cards on a board -/
def possibleArrangements (board : Board) (cards : CardSet) : Nat :=
  board.cells - cards.count + 1

/-- The theorem to be proved -/
theorem card_arrangement_count :
  let board := Board.mk 1994
  let cards := CardSet.mk 1000
  let arrangements := possibleArrangements board cards
  arrangements = 995 ∧ arrangements < 500000 := by
  sorry

end NUMINAMATH_CALUDE_card_arrangement_count_l3411_341121


namespace NUMINAMATH_CALUDE_probability_two_blue_l3411_341116

/-- Represents a jar with red and blue buttons -/
structure Jar where
  red : ℕ
  blue : ℕ

/-- Calculates the total number of buttons in a jar -/
def Jar.total (j : Jar) : ℕ := j.red + j.blue

/-- Represents the state of both jars after button removal -/
structure JarState where
  c : Jar
  d : Jar

/-- Defines the initial state of Jar C -/
def initial_jar_c : Jar := { red := 6, blue := 10 }

/-- Defines the button removal process -/
def remove_buttons (j : Jar) (n : ℕ) : JarState :=
  { c := { red := j.red - n, blue := j.blue - n },
    d := { red := n, blue := n } }

/-- Theorem stating the probability of choosing two blue buttons -/
theorem probability_two_blue (n : ℕ) : 
  let initial_total := initial_jar_c.total
  let final_state := remove_buttons initial_jar_c n
  final_state.c.total = (3 * initial_total) / 4 →
  (final_state.c.blue : ℚ) / final_state.c.total * 
  (final_state.d.blue : ℚ) / final_state.d.total = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_probability_two_blue_l3411_341116


namespace NUMINAMATH_CALUDE_train_carriages_l3411_341197

theorem train_carriages (num_trains : ℕ) (rows_per_carriage : ℕ) (wheels_per_row : ℕ) (total_wheels : ℕ) :
  num_trains = 4 → rows_per_carriage = 3 → wheels_per_row = 5 → total_wheels = 240 →
  (total_wheels / (rows_per_carriage * wheels_per_row)) / num_trains = 4 :=
by
  sorry

end NUMINAMATH_CALUDE_train_carriages_l3411_341197


namespace NUMINAMATH_CALUDE_right_triangle_sqrt_2_3_5_l3411_341135

theorem right_triangle_sqrt_2_3_5 :
  let a := Real.sqrt 2
  let b := Real.sqrt 3
  let c := Real.sqrt 5
  (a * a + b * b = c * c) ∧ (0 < a) ∧ (0 < b) ∧ (0 < c) :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_sqrt_2_3_5_l3411_341135


namespace NUMINAMATH_CALUDE_fraction_simplification_and_rationalization_l3411_341114

theorem fraction_simplification_and_rationalization :
  (3 : ℝ) / (Real.sqrt 75 + Real.sqrt 48 + Real.sqrt 12) = Real.sqrt 3 / 11 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_and_rationalization_l3411_341114


namespace NUMINAMATH_CALUDE_beth_comic_books_percentage_l3411_341129

theorem beth_comic_books_percentage
  (total_books : ℕ)
  (novel_percentage : ℚ)
  (graphic_novels : ℕ)
  (h1 : total_books = 120)
  (h2 : novel_percentage = 65/100)
  (h3 : graphic_novels = 18) :
  (total_books - (novel_percentage * total_books).floor - graphic_novels) / total_books = 1/5 := by
sorry

end NUMINAMATH_CALUDE_beth_comic_books_percentage_l3411_341129


namespace NUMINAMATH_CALUDE_fraction_value_l3411_341149

theorem fraction_value (x : ℝ) (h : 1 - 4/x + 4/(x^2) = 0) : 2/x = 1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_value_l3411_341149


namespace NUMINAMATH_CALUDE_mountain_hut_distance_l3411_341150

/-- The distance from the mountain hut to the station -/
def distance : ℝ := 15

/-- The time (in hours) from when the coach spoke until the train departs -/
def train_departure_time : ℝ := 3

theorem mountain_hut_distance :
  (distance / 4 = train_departure_time + 3/4) ∧
  (distance / 6 = train_departure_time - 1/2) →
  distance = 15 := by sorry

end NUMINAMATH_CALUDE_mountain_hut_distance_l3411_341150


namespace NUMINAMATH_CALUDE_parallel_lines_a_value_l3411_341124

/-- Two lines in the plane, represented by their equations --/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Check if two lines are parallel --/
def parallel (l₁ l₂ : Line) : Prop :=
  l₁.a * l₂.b = l₁.b * l₂.a

/-- Check if two lines are identical --/
def identical (l₁ l₂ : Line) : Prop :=
  ∃ (k : ℝ), k ≠ 0 ∧ l₁.a = k * l₂.a ∧ l₁.b = k * l₂.b ∧ l₁.c = k * l₂.c

theorem parallel_lines_a_value (a : ℝ) :
  let l₁ : Line := { a := a, b := 3, c := 1 }
  let l₂ : Line := { a := 2, b := a + 1, c := 1 }
  parallel l₁ l₂ ∧ ¬ identical l₁ l₂ → a = -3 := by
  sorry

end NUMINAMATH_CALUDE_parallel_lines_a_value_l3411_341124


namespace NUMINAMATH_CALUDE_set_operations_and_intersection_intersection_empty_iff_m_range_l3411_341194

def A : Set ℝ := {x | -4 < x ∧ x < 2}
def B : Set ℝ := {x | x < -5 ∨ x > 1}
def C (m : ℝ) : Set ℝ := {x | m - 1 ≤ x ∧ x ≤ m + 1}

theorem set_operations_and_intersection :
  (A ∪ B = {x | x < -5 ∨ x > -4}) ∧
  (A ∩ (Set.univ \ B) = {x | -4 < x ∧ x ≤ 1}) := by sorry

theorem intersection_empty_iff_m_range (m : ℝ) :
  (B ∩ C m = ∅) ↔ (-4 ≤ m ∧ m ≤ 0) := by sorry

end NUMINAMATH_CALUDE_set_operations_and_intersection_intersection_empty_iff_m_range_l3411_341194


namespace NUMINAMATH_CALUDE_min_socks_for_different_colors_l3411_341148

theorem min_socks_for_different_colors :
  let total_blue_socks : ℕ := 6
  let total_red_socks : ℕ := 6
  let min_socks : ℕ := 7
  ∀ (selected : ℕ), selected ≥ min_socks →
    ∃ (blue red : ℕ), blue + red = selected ∧
      blue ≤ total_blue_socks ∧
      red ≤ total_red_socks ∧
      (blue > 0 ∧ red > 0) :=
by sorry

end NUMINAMATH_CALUDE_min_socks_for_different_colors_l3411_341148


namespace NUMINAMATH_CALUDE_cross_number_puzzle_l3411_341195

def is_multiple_of_7 (n : ℕ) : Prop := ∃ k : ℕ, n = 7 * k

def is_multiple_of_odd_prime_square (n : ℕ) : Prop :=
  ∃ p : ℕ, Prime p ∧ Odd p ∧ ∃ k : ℕ, n = k * (p^2)

def is_internal_angle (n : ℕ) : Prop :=
  ∃ m : ℕ, m > 2 ∧ n = (m - 2) * 180 / m

def is_proper_factor (a b : ℕ) : Prop :=
  a ≠ 1 ∧ a ≠ b ∧ b % a = 0

theorem cross_number_puzzle :
  ∀ (across_1 across_3 across_5 down_1 down_2 down_4 : ℕ),
    across_1 > 0 ∧ across_3 > 0 ∧ across_5 > 0 ∧
    down_1 > 0 ∧ down_2 > 0 ∧ down_4 > 0 →
    is_multiple_of_7 across_1 →
    across_5 > 10 →
    is_multiple_of_odd_prime_square down_1 ∧ ¬(∃ k : ℕ, down_1 = k^2) ∧ ¬(∃ k : ℕ, down_1 = k^3) →
    is_internal_angle down_2 ∧ 170 < down_2 ∧ down_2 < 180 →
    is_proper_factor down_4 across_5 ∧ ¬is_proper_factor down_4 down_1 →
    across_3 = 961 := by
  sorry

end NUMINAMATH_CALUDE_cross_number_puzzle_l3411_341195
