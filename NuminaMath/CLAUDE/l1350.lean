import Mathlib

namespace ball_count_proof_l1350_135010

theorem ball_count_proof (a : ℕ) (h1 : a > 0) (h2 : 3 ≤ a) :
  (3 : ℝ) / a = 1/4 → a = 12 := by
  sorry

end ball_count_proof_l1350_135010


namespace square_minus_self_sum_l1350_135078

theorem square_minus_self_sum : (3^2 - 3) - (4^2 - 4) + (5^2 - 5) - (6^2 - 6) = -16 := by
  sorry

end square_minus_self_sum_l1350_135078


namespace sweet_bitter_fruits_problem_l1350_135023

/-- Represents the problem of buying sweet and bitter fruits --/
theorem sweet_bitter_fruits_problem 
  (x y : ℕ) -- x is the number of sweet fruits, y is the number of bitter fruits
  (h1 : x + y = 99) -- total number of fruits
  (h2 : 3 * x + (1/3) * y = 97) -- total cost in wen
  : 
  -- The system of equations correctly represents the problem
  (x + y = 99 ∧ 3 * x + (1/3) * y = 97) := by
  sorry


end sweet_bitter_fruits_problem_l1350_135023


namespace book_arrangement_count_l1350_135016

theorem book_arrangement_count :
  let math_books : ℕ := 4
  let english_books : ℕ := 5
  let science_books : ℕ := 2
  let subject_groups : ℕ := 3
  let total_arrangements : ℕ :=
    (Nat.factorial subject_groups) *
    (Nat.factorial math_books) *
    (Nat.factorial english_books) *
    (Nat.factorial science_books)
  total_arrangements = 34560 := by
  sorry

end book_arrangement_count_l1350_135016


namespace quadratic_function_properties_l1350_135086

/-- A quadratic function f(x) = ax^2 - bx satisfying certain conditions -/
def f (a b : ℝ) (x : ℝ) : ℝ := a * x^2 - b * x

/-- The theorem stating the properties of the quadratic function -/
theorem quadratic_function_properties (a b : ℝ) :
  (f a b 2 = 0) →
  (∃ x : ℝ, (f a b x = x) ∧ (∀ y : ℝ, f a b y = y → y = x)) →
  ((∀ x : ℝ, f a b x = -1/2 * x^2 + x) ∧
   (∀ x : ℝ, x ∈ Set.Icc 0 3 → f a b x ≤ 1/2) ∧
   (∃ x : ℝ, x ∈ Set.Icc 0 3 ∧ f a b x = 1/2)) :=
by sorry


end quadratic_function_properties_l1350_135086


namespace problem_statement_l1350_135070

theorem problem_statement (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + 2 * y = 3) :
  (∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ x₀ + 2 * y₀ = 3 ∧ y₀ / x₀ + 3 / y₀ = 4 ∧ 
    ∀ (x' y' : ℝ), x' > 0 → y' > 0 → x' + 2 * y' = 3 → y' / x' + 3 / y' ≥ 4) ∧
  (∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ x₀ + 2 * y₀ = 3 ∧ x₀ * y₀ = 9 / 8 ∧ 
    ∀ (x' y' : ℝ), x' > 0 → y' > 0 → x' + 2 * y' = 3 → x' * y' ≤ 9 / 8) ∧
  (∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ x₀ + 2 * y₀ = 3 ∧ x₀^2 + 4 * y₀^2 = 9 / 2 ∧ 
    ∀ (x' y' : ℝ), x' > 0 → y' > 0 → x' + 2 * y' = 3 → x'^2 + 4 * y'^2 ≥ 9 / 2) ∧
  ¬(∀ (x' y' : ℝ), x' > 0 → y' > 0 → x' + 2 * y' = 3 → Real.sqrt x' + Real.sqrt (2 * y') ≥ 2) :=
by sorry

end problem_statement_l1350_135070


namespace min_perimeter_triangle_l1350_135046

/-- Given a triangle with two sides of length 51 and 67 units, and the third side being an integer,
    the minimum possible perimeter is 135 units. -/
theorem min_perimeter_triangle (a b x : ℕ) (ha : a = 51) (hb : b = 67) : 
  (a + b > x ∧ a + x > b ∧ b + x > a) → (∀ y : ℕ, (a + b > y ∧ a + y > b ∧ b + y > a) → x ≤ y) →
  a + b + x = 135 := by
sorry

end min_perimeter_triangle_l1350_135046


namespace c_value_theorem_l1350_135079

-- Define the complex number i
def i : ℂ := Complex.I

-- Define the equation for c
def c_equation (a b : ℕ+) : ℂ := (a + b * i) ^ 3 - 107 * i

-- State the theorem
theorem c_value_theorem (a b c : ℕ+) :
  (c_equation a b).re = c ∧ (c_equation a b).im = 0 → c = 198 := by
  sorry

end c_value_theorem_l1350_135079


namespace find_b_l1350_135097

-- Define the ratio relationship
def ratio_relation (x y z : ℚ) : Prop :=
  ∃ (k : ℚ), x = 4 * k ∧ y = 3 * k ∧ z = 7 * k

-- Define the main theorem
theorem find_b (x y z b : ℚ) :
  ratio_relation x y z →
  y = 15 * b - 5 * z + 25 →
  z = 21 →
  b = 89 / 15 := by
  sorry


end find_b_l1350_135097


namespace l_shape_area_l1350_135090

/-- The area of an "L" shape formed by removing a smaller rectangle from a larger rectangle -/
theorem l_shape_area (large_length large_width small_length_diff small_width_diff : ℕ) : 
  large_length = 10 →
  large_width = 7 →
  small_length_diff = 3 →
  small_width_diff = 3 →
  (large_length * large_width) - ((large_length - small_length_diff) * (large_width - small_width_diff)) = 42 := by
sorry

end l_shape_area_l1350_135090


namespace b_alone_time_l1350_135051

-- Define the work rates
def work_rate_b : ℚ := 1
def work_rate_a : ℚ := 2 * work_rate_b
def work_rate_c : ℚ := 3 * work_rate_a

-- Define the total work (completed job)
def total_work : ℚ := 1

-- Define the time taken by all three together
def total_time : ℚ := 9

-- Theorem to prove
theorem b_alone_time (h1 : work_rate_a = 2 * work_rate_b)
                     (h2 : work_rate_c = 3 * work_rate_a)
                     (h3 : (work_rate_a + work_rate_b + work_rate_c) * total_time = total_work) :
  total_work / work_rate_b = 81 := by
  sorry


end b_alone_time_l1350_135051


namespace existence_of_special_number_l1350_135076

theorem existence_of_special_number :
  ∃ N : ℕ, 
    (∃ a b : ℕ, a < 150 ∧ b < 150 ∧ b = a + 1 ∧ ¬(a ∣ N) ∧ ¬(b ∣ N)) ∧
    (∀ k : ℕ, k ≤ 150 → (k ∣ N) ∨ k = a ∨ k = b) :=
by sorry

end existence_of_special_number_l1350_135076


namespace geometry_relations_l1350_135098

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (perpendicular : Line → Plane → Prop)
variable (subset : Line → Plane → Prop)
variable (parallel_planes : Plane → Plane → Prop)
variable (perpendicular_lines : Line → Line → Prop)
variable (parallel_lines : Line → Line → Prop)
variable (perpendicular_planes : Plane → Plane → Prop)

-- State the theorem
theorem geometry_relations 
  (l m : Line) (α β : Plane)
  (h1 : perpendicular l α)
  (h2 : subset m β) :
  (parallel_planes α β → perpendicular_lines l m) ∧
  ¬(perpendicular_lines l m → parallel_planes α β) ∧
  ¬(perpendicular_planes α β → parallel_lines l m) ∧
  (parallel_lines l m → perpendicular_planes α β) :=
sorry

end geometry_relations_l1350_135098


namespace inequality_counterexample_l1350_135092

theorem inequality_counterexample :
  ∃ (a b c d : ℝ), a > b ∧ c > d ∧ a + d ≤ b + c := by
  sorry

end inequality_counterexample_l1350_135092


namespace chess_tournament_games_l1350_135012

/-- The number of games in a chess tournament where each player plays every other player twice -/
def num_games (n : ℕ) : ℕ := n * (n - 1)

/-- Theorem: In a chess tournament with 12 players, where every player plays twice with each opponent, 
    the total number of games played is 264. -/
theorem chess_tournament_games : num_games 12 * 2 = 264 := by
  sorry

end chess_tournament_games_l1350_135012


namespace second_sale_price_is_270_l1350_135067

/-- Represents the clock selling scenario in a shop --/
structure ClockSale where
  originalCost : ℝ
  firstSaleMarkup : ℝ
  buybackPercentage : ℝ
  secondSaleProfit : ℝ
  costDifference : ℝ

/-- Calculates the second selling price of the clock --/
def secondSellingPrice (sale : ClockSale) : ℝ :=
  let firstSalePrice := sale.originalCost * (1 + sale.firstSaleMarkup)
  let buybackPrice := firstSalePrice * sale.buybackPercentage
  buybackPrice * (1 + sale.secondSaleProfit)

/-- Theorem stating the second selling price is $270 given the conditions --/
theorem second_sale_price_is_270 (sale : ClockSale)
  (h1 : sale.firstSaleMarkup = 0.2)
  (h2 : sale.buybackPercentage = 0.5)
  (h3 : sale.secondSaleProfit = 0.8)
  (h4 : sale.originalCost - (sale.originalCost * (1 + sale.firstSaleMarkup) * sale.buybackPercentage) = sale.costDifference)
  (h5 : sale.costDifference = 100)
  : secondSellingPrice sale = 270 := by
  sorry

#eval secondSellingPrice {
  originalCost := 250,
  firstSaleMarkup := 0.2,
  buybackPercentage := 0.5,
  secondSaleProfit := 0.8,
  costDifference := 100
}

end second_sale_price_is_270_l1350_135067


namespace quadratic_equation_one_l1350_135053

theorem quadratic_equation_one (x : ℝ) : 2 * (2 * x - 1)^2 = 8 ↔ x = 3/2 ∨ x = -1/2 := by sorry

end quadratic_equation_one_l1350_135053


namespace total_fruits_grown_special_technique_watermelons_special_pineapples_l1350_135088

/-- Represents the fruit growing data for a person -/
structure FruitData where
  watermelons : ℕ
  pineapples : ℕ
  mangoes : ℕ
  organic_watermelons : ℕ
  hydroponic_watermelons : ℕ
  dry_season_pineapples : ℕ
  vertical_pineapples : ℕ

/-- The fruit growing data for Jason -/
def jason : FruitData := {
  watermelons := 37,
  pineapples := 56,
  mangoes := 0,
  organic_watermelons := 15,
  hydroponic_watermelons := 0,
  dry_season_pineapples := 23,
  vertical_pineapples := 0
}

/-- The fruit growing data for Mark -/
def mark : FruitData := {
  watermelons := 68,
  pineapples := 27,
  mangoes := 0,
  organic_watermelons := 0,
  hydroponic_watermelons := 21,
  dry_season_pineapples := 0,
  vertical_pineapples := 17
}

/-- The fruit growing data for Sandy -/
def sandy : FruitData := {
  watermelons := 11,
  pineapples := 14,
  mangoes := 42,
  organic_watermelons := 0,
  hydroponic_watermelons := 0,
  dry_season_pineapples := 0,
  vertical_pineapples := 0
}

/-- Calculate the total fruits for a person -/
def totalFruits (data : FruitData) : ℕ :=
  data.watermelons + data.pineapples + data.mangoes

/-- Theorem stating the total number of fruits grown by all three people -/
theorem total_fruits_grown :
  totalFruits jason + totalFruits mark + totalFruits sandy = 255 := by
  sorry

/-- Theorem stating the number of watermelons grown using special techniques -/
theorem special_technique_watermelons :
  jason.organic_watermelons + mark.hydroponic_watermelons = 36 := by
  sorry

/-- Theorem stating the number of pineapples grown in dry season or vertically -/
theorem special_pineapples :
  jason.dry_season_pineapples + mark.vertical_pineapples = 40 := by
  sorry

end total_fruits_grown_special_technique_watermelons_special_pineapples_l1350_135088


namespace parabola_vertex_coordinates_l1350_135031

/-- The vertex of the parabola y = -x^2 + 4x - 5 has coordinates (2, -1) -/
theorem parabola_vertex_coordinates :
  let f (x : ℝ) := -x^2 + 4*x - 5
  ∃! (a b : ℝ), (∀ x, f x ≤ f a) ∧ f a = b ∧ a = 2 ∧ b = -1 := by
  sorry

end parabola_vertex_coordinates_l1350_135031


namespace inverse_proportion_relationship_l1350_135060

theorem inverse_proportion_relationship (x₁ x₂ y₁ y₂ : ℝ) 
  (h1 : y₁ = 2 / x₁)
  (h2 : y₂ = 2 / x₂)
  (h3 : x₁ > 0)
  (h4 : 0 > x₂) :
  y₁ > y₂ := by
  sorry

end inverse_proportion_relationship_l1350_135060


namespace nested_square_root_value_l1350_135061

theorem nested_square_root_value :
  ∀ y : ℝ, y = Real.sqrt (3 + y) → y = (1 + Real.sqrt 13) / 2 := by
  sorry

end nested_square_root_value_l1350_135061


namespace cultural_shirt_production_theorem_l1350_135085

/-- Represents the production and pricing of cultural shirts --/
structure CulturalShirtProduction where
  first_batch_cost : ℝ
  second_batch_cost : ℝ
  second_batch_quantity_multiplier : ℝ
  second_batch_cost_increase : ℝ
  discount_rate : ℝ
  discount_quantity : ℕ
  target_profit_margin : ℝ

/-- Calculates the cost per shirt in the first batch and the price per shirt for a given profit margin --/
def calculate_shirt_costs_and_price (prod : CulturalShirtProduction) :
  (ℝ × ℝ) :=
  sorry

/-- Theorem stating the correct cost and price for the given conditions --/
theorem cultural_shirt_production_theorem (prod : CulturalShirtProduction)
  (h1 : prod.first_batch_cost = 3000)
  (h2 : prod.second_batch_cost = 6600)
  (h3 : prod.second_batch_quantity_multiplier = 2)
  (h4 : prod.second_batch_cost_increase = 3)
  (h5 : prod.discount_rate = 0.6)
  (h6 : prod.discount_quantity = 30)
  (h7 : prod.target_profit_margin = 0.5) :
  calculate_shirt_costs_and_price prod = (30, 50) :=
  sorry

end cultural_shirt_production_theorem_l1350_135085


namespace election_probabilities_l1350_135037

structure Student where
  name : String
  prob_elected : ℚ

def A : Student := { name := "A", prob_elected := 4/5 }
def B : Student := { name := "B", prob_elected := 3/5 }
def C : Student := { name := "C", prob_elected := 7/10 }

def students : List Student := [A, B, C]

-- Probability that exactly one student is elected
def prob_exactly_one_elected (students : List Student) : ℚ :=
  sorry

-- Probability that at most two students are elected
def prob_at_most_two_elected (students : List Student) : ℚ :=
  sorry

theorem election_probabilities :
  (prob_exactly_one_elected students = 47/250) ∧
  (prob_at_most_two_elected students = 83/125) := by
  sorry

end election_probabilities_l1350_135037


namespace lawn_mowing_problem_l1350_135002

/-- The number of additional people needed to mow a lawn in a shorter time -/
def additional_people_needed (initial_people initial_time target_time : ℕ) : ℕ :=
  (initial_people * initial_time / target_time) - initial_people

/-- Proof that 24 additional people are needed to mow the lawn in 2 hours -/
theorem lawn_mowing_problem :
  additional_people_needed 8 8 2 = 24 := by
  sorry

end lawn_mowing_problem_l1350_135002


namespace largest_root_bound_l1350_135048

/-- A polynomial of degree 4 with constrained coefficients -/
def ConstrainedPoly (b a₂ a₁ a₀ : ℝ) : ℝ → ℝ :=
  fun x ↦ x^4 + b*x^3 + a₂*x^2 + a₁*x + a₀

/-- The set of all constrained polynomials -/
def ConstrainedPolySet : Set (ℝ → ℝ) :=
  {p | ∃ b a₂ a₁ a₀, |b| < 3 ∧ |a₂| < 2 ∧ |a₁| < 2 ∧ |a₀| < 2 ∧ p = ConstrainedPoly b a₂ a₁ a₀}

theorem largest_root_bound :
  (∃ p ∈ ConstrainedPolySet, ∃ r, 3 < r ∧ r < 4 ∧ p r = 0) ∧
  (∀ p ∈ ConstrainedPolySet, ∀ r ≥ 4, p r ≠ 0) :=
sorry

end largest_root_bound_l1350_135048


namespace intersection_distance_and_max_value_l1350_135017

/-- Curve C₁ in polar coordinates -/
def C₁ (ρ : ℝ) : Prop := ρ = 1

/-- Curve C₂ in parametric form -/
def C₂ (t x y : ℝ) : Prop := x = 1 + t ∧ y = 2 + t

/-- Point M on C₁ -/
def M (x y : ℝ) : Prop := x^2 + y^2 = 1

theorem intersection_distance_and_max_value :
  ∃ (A B : ℝ × ℝ),
    (∀ ρ, C₁ ρ → (A.1^2 + A.2^2 = ρ^2 ∧ B.1^2 + B.2^2 = ρ^2)) ∧
    (∃ t₁ t₂, C₂ t₁ A.1 A.2 ∧ C₂ t₂ B.1 B.2) ∧
    (A.1 - B.1)^2 + (A.2 - B.2)^2 = 2 ∧
    (∀ x y, M x y → (x + 1) * (y + 1) ≤ 3/2 + Real.sqrt 2) ∧
    (∃ x y, M x y ∧ (x + 1) * (y + 1) = 3/2 + Real.sqrt 2) :=
by sorry

end intersection_distance_and_max_value_l1350_135017


namespace cos_sixty_degrees_l1350_135093

theorem cos_sixty_degrees : Real.cos (60 * π / 180) = 1 / 2 := by
  sorry

end cos_sixty_degrees_l1350_135093


namespace smallest_n_with_9_and_terminating_l1350_135014

def has_digit_9 (n : ℕ) : Prop :=
  ∃ d : ℕ, d < 10 ∧ d = 9 ∧ ∃ k m : ℕ, n = k * 10 + d + m * 100

def is_terminating_decimal (n : ℕ) : Prop :=
  ∃ m k : ℕ, n = 2^m * 5^k

theorem smallest_n_with_9_and_terminating : 
  (∀ n : ℕ, n > 0 ∧ n < 4096 → ¬(is_terminating_decimal n ∧ has_digit_9 n)) ∧ 
  (is_terminating_decimal 4096 ∧ has_digit_9 4096) :=
sorry

end smallest_n_with_9_and_terminating_l1350_135014


namespace sqrt_x_minus_one_meaningful_l1350_135049

theorem sqrt_x_minus_one_meaningful (x : ℝ) : 
  (∃ y : ℝ, y ^ 2 = x - 1) → x ≥ 1 := by sorry

end sqrt_x_minus_one_meaningful_l1350_135049


namespace arithmetic_sequence_specific_sum_l1350_135080

/-- An arithmetic sequence with sum S_n of its first n terms. -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  S : ℕ → ℝ  -- The sum function
  is_arithmetic : ∀ n : ℕ, a (n + 1) - a n = a 1 - a 0
  sum_formula : ∀ n : ℕ, S n = n * (a 0 + a (n - 1)) / 2

/-- Given an arithmetic sequence with specific sum values, prove n = 4. -/
theorem arithmetic_sequence_specific_sum (seq : ArithmeticSequence) 
  (h1 : seq.S 6 = 36)
  (h2 : seq.S 12 = 144)
  (h3 : ∃ n : ℕ, seq.S (6 * n) = 576) :
  ∃ n : ℕ, n = 4 ∧ seq.S (6 * n) = 576 := by
  sorry

end arithmetic_sequence_specific_sum_l1350_135080


namespace johns_grocery_spend_l1350_135059

/-- Represents the cost of John's purchase at the grocery store. -/
def grocery_purchase (chip_price corn_chip_price : ℚ) (chip_quantity corn_chip_quantity : ℕ) : ℚ :=
  chip_price * chip_quantity + corn_chip_price * corn_chip_quantity

/-- Proves that John's total spend is $45 given the specified conditions. -/
theorem johns_grocery_spend :
  grocery_purchase 2 1.5 15 10 = 45 := by
sorry

end johns_grocery_spend_l1350_135059


namespace ball_probabilities_l1350_135064

/-- The number of red balls in the bag -/
def num_red : ℕ := 3

/-- The number of white balls in the bag -/
def num_white : ℕ := 2

/-- The total number of balls in the bag -/
def total_balls : ℕ := num_red + num_white

/-- The number of balls drawn -/
def num_drawn : ℕ := 2

theorem ball_probabilities :
  (num_red * (num_red - 1) / (total_balls * (total_balls - 1)) = 3 / 10) ∧
  (1 - (num_white * (num_white - 1) / (total_balls * (total_balls - 1))) = 9 / 10) :=
sorry

end ball_probabilities_l1350_135064


namespace divisibility_by_seven_l1350_135091

theorem divisibility_by_seven (a b : ℤ) : (10 * a + b) % 7 = 0 ↔ (a - 2 * b) % 7 = 0 := by
  sorry

end divisibility_by_seven_l1350_135091


namespace hema_rahul_ratio_l1350_135065

-- Define variables for ages
variable (Raj Ravi Hema Rahul : ℚ)

-- Define the conditions
axiom raj_older : Raj = Ravi + 3
axiom hema_younger : Hema = Ravi - 2
axiom raj_triple : Raj = 3 * Rahul
axiom raj_twenty : Raj = 20
axiom raj_hema_ratio : Raj = Hema + (1/3) * Hema

-- Theorem to prove
theorem hema_rahul_ratio : Hema / Rahul = 2.25 := by
  sorry

end hema_rahul_ratio_l1350_135065


namespace inequality_solution_l1350_135084

theorem inequality_solution :
  ∃! x : ℝ, (Real.sqrt (x^3 + 2*x - 58) + 5) * |x^3 - 7*x^2 + 13*x - 3| ≤ 0 ∧ x = 2 + Real.sqrt 3 := by
  sorry

end inequality_solution_l1350_135084


namespace binary_1101101000_to_octal_1550_l1350_135026

def binary_to_decimal (b : List Bool) : Nat :=
  b.foldl (fun acc x => 2 * acc + if x then 1 else 0) 0

def decimal_to_octal (n : Nat) : List Nat :=
  if n < 8 then [n]
  else (n % 8) :: decimal_to_octal (n / 8)

theorem binary_1101101000_to_octal_1550 :
  let binary : List Bool := [false, false, false, true, false, true, true, false, true, true]
  let octal : List Nat := [0, 5, 5, 1]
  decimal_to_octal (binary_to_decimal binary) = octal.reverse := by
  sorry

end binary_1101101000_to_octal_1550_l1350_135026


namespace counterexample_odd_composite_plus_two_prime_l1350_135096

theorem counterexample_odd_composite_plus_two_prime :
  ∃ n : ℕ, 
    Odd n ∧ 
    ¬ Prime n ∧ 
    n > 1 ∧ 
    ¬ Prime (n + 2) ∧
    n = 25 :=
by
  sorry


end counterexample_odd_composite_plus_two_prime_l1350_135096


namespace max_common_ratio_geometric_sequence_l1350_135057

/-- Given a geometric sequence {a_n} satisfying a_1(a_2 + a_3) = 6a_1 - 9, 
    the maximum value of the common ratio q is (-1 + √5) / 2 -/
theorem max_common_ratio_geometric_sequence (a : ℕ → ℝ) (q : ℝ) :
  (∀ n, a (n + 1) = a n * q) →  -- geometric sequence condition
  a 1 * (a 2 + a 3) = 6 * a 1 - 9 →  -- given equation
  q ≤ (-1 + Real.sqrt 5) / 2 ∧
  ∃ (a : ℕ → ℝ), (∀ n, a (n + 1) = a n * q) ∧ 
    a 1 * (a 2 + a 3) = 6 * a 1 - 9 ∧ 
    q = (-1 + Real.sqrt 5) / 2 := by
  sorry

#check max_common_ratio_geometric_sequence

end max_common_ratio_geometric_sequence_l1350_135057


namespace square_perimeter_from_diagonal_l1350_135011

theorem square_perimeter_from_diagonal (d : ℝ) (h : d = 20) :
  let s := Real.sqrt ((d^2) / 2)
  4 * s = 40 * Real.sqrt 2 := by sorry

end square_perimeter_from_diagonal_l1350_135011


namespace complex_modulus_equality_l1350_135082

theorem complex_modulus_equality (x y : ℝ) :
  (Complex.I + 1) * x = Complex.I * y + 1 →
  Complex.abs (x + Complex.I * y) = Real.sqrt 2 := by
  sorry

end complex_modulus_equality_l1350_135082


namespace parabola_minimum_y_value_l1350_135087

/-- The minimum y-value of the parabola y = 3x^2 + 6x + 4 is 1 -/
theorem parabola_minimum_y_value :
  let f : ℝ → ℝ := fun x ↦ 3 * x^2 + 6 * x + 4
  ∃ x₀ : ℝ, ∀ x : ℝ, f x₀ ≤ f x ∧ f x₀ = 1 :=
by
  sorry

end parabola_minimum_y_value_l1350_135087


namespace investment_income_is_500_l1350_135034

/-- Calculates the total yearly income from a set of investments -/
def totalYearlyIncome (totalAmount : ℝ) (firstInvestment : ℝ) (firstRate : ℝ) 
                      (secondInvestment : ℝ) (secondRate : ℝ) (remainderRate : ℝ) : ℝ :=
  let remainderInvestment := totalAmount - firstInvestment - secondInvestment
  firstInvestment * firstRate + secondInvestment * secondRate + remainderInvestment * remainderRate

/-- Theorem: The total yearly income from the given investment strategy is $500 -/
theorem investment_income_is_500 : 
  totalYearlyIncome 10000 4000 0.05 3500 0.04 0.064 = 500 := by
  sorry

end investment_income_is_500_l1350_135034


namespace characterization_of_m_l1350_135029

theorem characterization_of_m (m : ℕ+) : 
  (∃ p : ℕ, Prime p ∧ ∀ n : ℕ+, ¬(p ∣ n^n.val - m.val)) ↔ m ≠ 1 := by
  sorry

end characterization_of_m_l1350_135029


namespace unreachable_zero_l1350_135015

/-- Represents a point in 2D space -/
structure Point where
  x : ℤ
  y : ℤ

/-- The set of possible moves -/
inductive Move where
  | swap : Move
  | scale : Move
  | negate : Move
  | increment : Move
  | decrement : Move

/-- Apply a move to a point -/
def applyMove (p : Point) (m : Move) : Point :=
  match m with
  | Move.swap => ⟨p.y, p.x⟩
  | Move.scale => ⟨3 * p.x, -2 * p.y⟩
  | Move.negate => ⟨-2 * p.x, 3 * p.y⟩
  | Move.increment => ⟨p.x + 1, p.y + 4⟩
  | Move.decrement => ⟨p.x - 1, p.y - 4⟩

/-- The sum of coordinates modulo 5 -/
def sumMod5 (p : Point) : ℤ :=
  (p.x + p.y) % 5

/-- Theorem: It's impossible to reach (0, 0) from (0, 1) using the given moves -/
theorem unreachable_zero : 
  ∀ (moves : List Move), 
    let finalPoint := moves.foldl applyMove ⟨0, 1⟩
    sumMod5 finalPoint ≠ 0 := by
  sorry


end unreachable_zero_l1350_135015


namespace higher_power_of_two_divisibility_l1350_135039

theorem higher_power_of_two_divisibility (n k : ℕ) : 
  ∃ i ∈ Finset.range k, ∀ j ∈ Finset.range k, j ≠ i → 
    (∃ m : ℕ, (n + i + 1) = 2^m * (2*l + 1) ∧ 
              ∀ p : ℕ, (n + j + 1) = 2^p * (2*q + 1) → m > p) :=
by sorry

end higher_power_of_two_divisibility_l1350_135039


namespace stating_binary_arithmetic_equality_l1350_135007

/-- Converts a binary number (represented as a list of bits) to a natural number. -/
def binary_to_nat (bits : List Bool) : ℕ :=
  bits.foldr (fun b acc => 2 * acc + if b then 1 else 0) 0

/-- Represents the binary number 1101₂ -/
def b1101 : List Bool := [true, true, false, true]

/-- Represents the binary number 111₂ -/
def b111 : List Bool := [true, true, true]

/-- Represents the binary number 101₂ -/
def b101 : List Bool := [true, false, true]

/-- Represents the binary number 1001₂ -/
def b1001 : List Bool := [true, false, false, true]

/-- Represents the binary number 11₂ -/
def b11 : List Bool := [true, true]

/-- Represents the binary number 10101₂ (the expected result) -/
def b10101 : List Bool := [true, false, true, false, true]

/-- 
Theorem stating that the binary arithmetic operation 
1101₂ + 111₂ - 101₂ + 1001₂ - 11₂ equals 10101₂
-/
theorem binary_arithmetic_equality : 
  binary_to_nat b1101 + binary_to_nat b111 - binary_to_nat b101 + 
  binary_to_nat b1001 - binary_to_nat b11 = binary_to_nat b10101 := by
  sorry

end stating_binary_arithmetic_equality_l1350_135007


namespace power_three_124_mod_7_l1350_135095

theorem power_three_124_mod_7 : 3^124 % 7 = 4 := by
  sorry

end power_three_124_mod_7_l1350_135095


namespace stating_pipeline_equation_l1350_135019

/-- Represents the total length of the pipeline in meters -/
def total_length : ℝ := 3000

/-- Represents the increase in daily work efficiency as a decimal -/
def efficiency_increase : ℝ := 0.25

/-- Represents the number of days the project is completed ahead of schedule -/
def days_ahead : ℝ := 20

/-- 
Theorem stating that the equation correctly represents the relationship 
between the original daily pipeline laying rate and the given conditions
-/
theorem pipeline_equation (x : ℝ) : 
  total_length / ((1 + efficiency_increase) * x) - total_length / x = days_ahead := by
  sorry

end stating_pipeline_equation_l1350_135019


namespace intersection_P_Q_l1350_135000

-- Define the sets P and Q
def P : Set ℝ := {-1, 0, Real.sqrt 2}
def Q : Set ℝ := {y | ∃ θ : ℝ, y = Real.sin θ}

-- State the theorem
theorem intersection_P_Q : P ∩ Q = {-1, 0} := by sorry

end intersection_P_Q_l1350_135000


namespace inclination_angle_sqrt3x_plus_y_minus2_l1350_135063

/-- The inclination angle of a line given by the equation √3x + y - 2 = 0 is 120°. -/
theorem inclination_angle_sqrt3x_plus_y_minus2 :
  let line : ℝ → ℝ → Prop := λ x y ↦ Real.sqrt 3 * x + y - 2 = 0
  ∃ α : ℝ, α = 120 * (π / 180) ∧ 
    ∀ x y : ℝ, line x y → Real.tan α = -Real.sqrt 3 :=
by sorry

end inclination_angle_sqrt3x_plus_y_minus2_l1350_135063


namespace average_string_length_l1350_135021

theorem average_string_length : 
  let string_lengths : List ℝ := [1.5, 4.5, 6, 3]
  let n : ℕ := string_lengths.length
  let sum : ℝ := string_lengths.sum
  sum / n = 3.75 := by
sorry

end average_string_length_l1350_135021


namespace negation_of_existence_proposition_l1350_135068

theorem negation_of_existence_proposition :
  (¬ ∃ n : ℕ, n^2 > 2^n) ↔ (∀ n : ℕ, n^2 ≤ 2^n) := by
  sorry

end negation_of_existence_proposition_l1350_135068


namespace bisection_method_accuracy_l1350_135001

theorem bisection_method_accuracy (f : ℝ → ℝ) (x₀ : ℝ) :
  ContinuousOn f (Set.Ioi 0) →
  Irrational x₀ →
  x₀ ∈ Set.Ioo 2 3 →
  f x₀ = 0 →
  ∃ (a b : ℝ), a < x₀ ∧ x₀ < b ∧ b - a ≤ 1 / 2^9 ∧ b - a > 1 / 2^8 := by
  sorry

end bisection_method_accuracy_l1350_135001


namespace parents_disagreeing_with_tuition_increase_l1350_135050

theorem parents_disagreeing_with_tuition_increase 
  (total_parents : ℕ) 
  (agree_percentage : ℚ) 
  (h1 : total_parents = 800) 
  (h2 : agree_percentage = 1/5) : 
  (1 - agree_percentage) * total_parents = 640 := by
  sorry

end parents_disagreeing_with_tuition_increase_l1350_135050


namespace greatest_integer_b_for_no_negative_nine_l1350_135077

theorem greatest_integer_b_for_no_negative_nine : ∃ (b : ℤ), 
  (∀ x : ℝ, 3 * x^2 + b * x + 15 ≠ -9) ∧
  (∀ c : ℤ, c > b → ∃ x : ℝ, 3 * x^2 + c * x + 15 = -9) ∧
  b = 16 := by
  sorry

end greatest_integer_b_for_no_negative_nine_l1350_135077


namespace maries_daily_rent_is_24_l1350_135028

/-- Represents Marie's bakery finances --/
structure BakeryFinances where
  cashRegisterCost : ℕ
  dailyBreadLoaves : ℕ
  breadPrice : ℕ
  dailyCakes : ℕ
  cakePrice : ℕ
  dailyElectricityCost : ℕ
  daysToPayCashRegister : ℕ

/-- Calculates the daily rent given the bakery finances --/
def calculateDailyRent (finances : BakeryFinances) : ℕ :=
  let dailyRevenue := finances.dailyBreadLoaves * finances.breadPrice + finances.dailyCakes * finances.cakePrice
  let dailyProfit := finances.cashRegisterCost / finances.daysToPayCashRegister
  dailyRevenue - dailyProfit - finances.dailyElectricityCost

/-- Theorem stating that Marie's daily rent is $24 --/
theorem maries_daily_rent_is_24 (finances : BakeryFinances)
    (h1 : finances.cashRegisterCost = 1040)
    (h2 : finances.dailyBreadLoaves = 40)
    (h3 : finances.breadPrice = 2)
    (h4 : finances.dailyCakes = 6)
    (h5 : finances.cakePrice = 12)
    (h6 : finances.dailyElectricityCost = 2)
    (h7 : finances.daysToPayCashRegister = 8) :
    calculateDailyRent finances = 24 := by
  sorry

end maries_daily_rent_is_24_l1350_135028


namespace grid_number_is_333_l1350_135032

/-- Represents a shape type -/
inductive Shape : Type
| A
| B
| C

/-- Represents a row in the grid -/
structure Row :=
  (shape : Shape)
  (count : Nat)

/-- The problem setup -/
def grid_setup : List Row :=
  [⟨Shape.A, 3⟩, ⟨Shape.B, 3⟩, ⟨Shape.C, 3⟩]

/-- Converts a list of rows to a natural number -/
def rows_to_number (rows : List Row) : Nat :=
  rows.foldl (fun acc row => acc * 10 + row.count) 0

/-- The main theorem -/
theorem grid_number_is_333 :
  rows_to_number grid_setup = 333 := by
  sorry

end grid_number_is_333_l1350_135032


namespace sum_of_coefficients_l1350_135040

theorem sum_of_coefficients (a a₁ a₂ a₃ a₄ a₅ : ℝ) : 
  (∀ x y : ℝ, (x - 2*y)^5 = a*x^5 + a₁*x^4*y + a₂*x^3*y^2 + a₃*x^2*y^3 + a₄*x*y^4 + a₅*y^5) →
  a + a₁ + a₂ + a₃ + a₄ + a₅ = -1 := by
  sorry

end sum_of_coefficients_l1350_135040


namespace school_bus_seats_l1350_135043

theorem school_bus_seats (total_students : ℕ) (num_buses : ℕ) (h1 : total_students = 60) (h2 : num_buses = 6) (h3 : total_students % num_buses = 0) :
  total_students / num_buses = 10 := by
sorry

end school_bus_seats_l1350_135043


namespace five_solutions_for_f_f_x_eq_8_l1350_135006

noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ -2 then x^2 - 1 else x + 4

theorem five_solutions_for_f_f_x_eq_8 :
  ∃! (s : Finset ℝ), s.card = 5 ∧ ∀ x : ℝ, x ∈ s ↔ f (f x) = 8 :=
sorry

end five_solutions_for_f_f_x_eq_8_l1350_135006


namespace number_ordering_l1350_135056

def a : ℕ := 62398
def b : ℕ := 63298
def c : ℕ := 62389
def d : ℕ := 63289

theorem number_ordering : b > d ∧ d > a ∧ a > c := by sorry

end number_ordering_l1350_135056


namespace solution_set_characterization_l1350_135038

/-- The set of solutions to the equation z + y² + x³ = xyz with x = gcd(y, z) -/
def SolutionSet : Set (ℕ × ℕ × ℕ) :=
  {s | s.1 > 0 ∧ s.2.1 > 0 ∧ s.2.2 > 0 ∧
       s.2.2 + s.2.1^2 + s.1^3 = s.1 * s.2.1 * s.2.2 ∧
       s.1 = Nat.gcd s.2.1 s.2.2}

theorem solution_set_characterization :
  SolutionSet = {(1, 2, 5), (1, 3, 5), (2, 2, 4), (2, 6, 4)} := by sorry

end solution_set_characterization_l1350_135038


namespace smallest_common_factor_l1350_135062

theorem smallest_common_factor (n : ℕ) : 
  (∃ k : ℕ, k > 1 ∧ k ∣ (9*n - 2) ∧ k ∣ (7*n + 3)) → n ≥ 23 :=
sorry

end smallest_common_factor_l1350_135062


namespace simplify_expression_l1350_135058

theorem simplify_expression (b : ℝ) : 3*b*(3*b^2 + 2*b) - 2*b^2 = 9*b^3 + 4*b^2 := by
  sorry

end simplify_expression_l1350_135058


namespace initial_speed_satisfies_conditions_l1350_135004

/-- Represents the initial speed of the car in km/h -/
def V : ℝ := 60

/-- Represents the distance from A to B in km -/
def distance : ℝ := 300

/-- Represents the increase in speed on the return journey in km/h -/
def speed_increase : ℝ := 16

/-- Represents the time after which the speed was increased on the return journey in hours -/
def time_before_increase : ℝ := 1.2

/-- Represents the time difference between the outward and return journeys in hours -/
def time_difference : ℝ := 0.8

/-- Theorem stating that the initial speed satisfies the given conditions -/
theorem initial_speed_satisfies_conditions :
  (distance / V - time_difference = 
   time_before_increase + (distance - V * time_before_increase) / (V + speed_increase)) := by
  sorry

end initial_speed_satisfies_conditions_l1350_135004


namespace cos_power_six_expansion_l1350_135025

theorem cos_power_six_expansion (b₁ b₂ b₃ b₄ b₅ b₆ : ℝ) :
  (∀ θ : ℝ, Real.cos θ ^ 6 = b₁ * Real.cos θ + b₂ * Real.cos (2 * θ) + b₃ * Real.cos (3 * θ) +
    b₄ * Real.cos (4 * θ) + b₅ * Real.cos (5 * θ) + b₆ * Real.cos (6 * θ)) →
  b₁^2 + b₂^2 + b₃^2 + b₄^2 + b₅^2 + b₆^2 = 131 / 512 :=
by sorry

end cos_power_six_expansion_l1350_135025


namespace all_squares_similar_l1350_135052

/-- A square is a quadrilateral with all sides equal and all angles 90 degrees. -/
structure Square where
  side : ℝ
  side_positive : side > 0

/-- Similarity of shapes means they have the same shape but not necessarily the same size. -/
def are_similar (s1 s2 : Square) : Prop :=
  ∃ k : ℝ, k > 0 ∧ s1.side = k * s2.side

/-- Any two squares are similar. -/
theorem all_squares_similar (s1 s2 : Square) : are_similar s1 s2 := by
  sorry

end all_squares_similar_l1350_135052


namespace determinant_inequality_range_l1350_135022

theorem determinant_inequality_range (x : ℝ) : 
  (Matrix.det !![x + 3, x^2; 1, 4] < 0) ↔ (x ∈ Set.Iio (-2) ∪ Set.Ioi 6) := by
  sorry

end determinant_inequality_range_l1350_135022


namespace remaining_money_proof_l1350_135072

def calculate_remaining_money (initial_amount apples_price milk_price oranges_price candy_price eggs_price apples_discount milk_discount : ℚ) : ℚ :=
  let discounted_apples_price := apples_price * (1 - apples_discount)
  let discounted_milk_price := milk_price * (1 - milk_discount)
  let total_spent := discounted_apples_price + discounted_milk_price + oranges_price + candy_price + eggs_price
  initial_amount - total_spent

theorem remaining_money_proof :
  calculate_remaining_money 95 25 8 14 6 12 (15/100) (10/100) = 6891/200 :=
by sorry

end remaining_money_proof_l1350_135072


namespace smallest_n_for_P_less_than_threshold_l1350_135036

/-- The probability of drawing n-1 white marbles followed by a red marble -/
def P (n : ℕ) : ℚ := 1 / (n * (n + 1))

/-- The number of boxes -/
def num_boxes : ℕ := 3000

theorem smallest_n_for_P_less_than_threshold :
  (∀ k < 55, P k ≥ 1 / num_boxes) ∧
  P 55 < 1 / num_boxes :=
sorry

end smallest_n_for_P_less_than_threshold_l1350_135036


namespace exists_line_with_perpendicular_chord_l1350_135044

-- Define the ellipse C
def C (x y : ℝ) : Prop := x^2 + 2*y^2 = 8

-- Define the line l
def l (x y m : ℝ) : Prop := y = x + m

-- Define the condition for A and B being on the ellipse C and line l
def on_ellipse_and_line (x₁ y₁ x₂ y₂ m : ℝ) : Prop :=
  C x₁ y₁ ∧ C x₂ y₂ ∧ l x₁ y₁ m ∧ l x₂ y₂ m

-- Define the condition for AB being perpendicular to OA and OB
def perpendicular_chord (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  x₁ * x₂ + y₁ * y₂ = 0

-- Main theorem
theorem exists_line_with_perpendicular_chord :
  ∃ m : ℝ, m = 4 * Real.sqrt 3 / 3 ∨ m = -4 * Real.sqrt 3 / 3 ∧
  ∃ x₁ y₁ x₂ y₂ : ℝ, 
    on_ellipse_and_line x₁ y₁ x₂ y₂ m ∧
    perpendicular_chord x₁ y₁ x₂ y₂ :=
  sorry

end exists_line_with_perpendicular_chord_l1350_135044


namespace fraction_under_eleven_l1350_135033

theorem fraction_under_eleven (total : ℕ) (between_eleven_and_thirteen : ℚ) (thirteen_and_above : ℕ) :
  total = 45 →
  between_eleven_and_thirteen = 2 / 5 →
  thirteen_and_above = 12 →
  (total : ℚ) - between_eleven_and_thirteen * total - (thirteen_and_above : ℚ) = 1 / 3 * total :=
by sorry

end fraction_under_eleven_l1350_135033


namespace checkerboard_probability_l1350_135027

/-- Represents a rectangular checkerboard -/
structure Checkerboard where
  length : ℕ
  width : ℕ

/-- Calculates the total number of squares on the checkerboard -/
def totalSquares (board : Checkerboard) : ℕ :=
  board.length * board.width

/-- Calculates the number of squares not touching or adjacent to any edge -/
def innerSquares (board : Checkerboard) : ℕ :=
  (board.length - 4) * (board.width - 4)

/-- The probability of choosing a square not touching or adjacent to any edge -/
def innerSquareProbability (board : Checkerboard) : ℚ :=
  innerSquares board / totalSquares board

theorem checkerboard_probability :
  ∃ (board : Checkerboard), board.length = 10 ∧ board.width = 6 ∧
  innerSquareProbability board = 1 / 5 := by
  sorry

end checkerboard_probability_l1350_135027


namespace simplify_expression_l1350_135030

theorem simplify_expression (x : ℝ) : 3*x + 5*x^2 + 2 - (9 - 4*x - 5*x^2) = 10*x^2 + 7*x - 7 := by
  sorry

end simplify_expression_l1350_135030


namespace online_store_problem_l1350_135041

/-- Represents the purchase and selling prices of products A and B -/
structure Prices where
  purchaseA : ℝ
  purchaseB : ℝ
  sellingA : ℝ
  sellingB : ℝ

/-- Represents the first purchase conditions -/
structure FirstPurchase where
  totalItems : ℕ
  totalCost : ℝ

/-- Represents the second purchase conditions -/
structure SecondPurchase where
  totalItems : ℕ
  maxCost : ℝ

/-- Represents the sales conditions for product B -/
structure BSales where
  initialSales : ℕ
  additionalSalesPerReduction : ℕ

/-- Main theorem stating the solutions to the problem -/
theorem online_store_problem 
  (prices : Prices)
  (firstPurchase : FirstPurchase)
  (secondPurchase : SecondPurchase)
  (bSales : BSales)
  (h1 : prices.purchaseA = 30)
  (h2 : prices.purchaseB = 25)
  (h3 : prices.sellingA = 45)
  (h4 : prices.sellingB = 37)
  (h5 : firstPurchase.totalItems = 30)
  (h6 : firstPurchase.totalCost = 850)
  (h7 : secondPurchase.totalItems = 80)
  (h8 : secondPurchase.maxCost = 2200)
  (h9 : bSales.initialSales = 4)
  (h10 : bSales.additionalSalesPerReduction = 2) :
  (∃ (x y : ℕ), x + y = firstPurchase.totalItems ∧ 
    prices.purchaseA * x + prices.purchaseB * y = firstPurchase.totalCost ∧ 
    x = 20 ∧ y = 10) ∧
  (∃ (m : ℕ), m ≤ secondPurchase.totalItems ∧ 
    prices.purchaseA * m + prices.purchaseB * (secondPurchase.totalItems - m) ≤ secondPurchase.maxCost ∧
    (prices.sellingA - prices.purchaseA) * m + (prices.sellingB - prices.purchaseB) * (secondPurchase.totalItems - m) = 2520 ∧
    m = 40) ∧
  (∃ (a₁ a₂ : ℝ), (12 - a₁) * (bSales.initialSales + 2 * a₁) = 90 ∧
    (12 - a₂) * (bSales.initialSales + 2 * a₂) = 90 ∧
    a₁ = 3 ∧ a₂ = 7) := by
  sorry

end online_store_problem_l1350_135041


namespace sqrt_sum_fractions_l1350_135020

theorem sqrt_sum_fractions : 
  Real.sqrt (2 * ((1 : ℝ) / 25 + (1 : ℝ) / 36)) = (Real.sqrt 122) / 30 := by
  sorry

end sqrt_sum_fractions_l1350_135020


namespace negation_of_universal_proposition_l1350_135005

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, Real.exp x > x^2) ↔ (∃ x : ℝ, Real.exp x ≤ x^2) := by sorry

end negation_of_universal_proposition_l1350_135005


namespace slope_of_line_l1350_135089

/-- The slope of a line passing through two points is 1 -/
theorem slope_of_line (M N : ℝ × ℝ) (h1 : M = (-Real.sqrt 3, Real.sqrt 2)) 
  (h2 : N = (-Real.sqrt 2, Real.sqrt 3)) : 
  (N.2 - M.2) / (N.1 - M.1) = 1 := by
  sorry

end slope_of_line_l1350_135089


namespace square_cut_perimeter_l1350_135024

/-- Given a square with perimeter 64 inches, prove that cutting a right triangle
    with hypotenuse equal to one side and translating it results in a new figure
    with perimeter 32 + 16√2 inches. -/
theorem square_cut_perimeter (square_perimeter : ℝ) (h1 : square_perimeter = 64) :
  let side_length : ℝ := square_perimeter / 4
  let triangle_leg : ℝ := side_length * Real.sqrt 2 / 2
  let new_perimeter : ℝ := 2 * side_length + 2 * triangle_leg
  new_perimeter = 32 + 16 * Real.sqrt 2 := by
  sorry


end square_cut_perimeter_l1350_135024


namespace integer_between_sqrt27_and_7_l1350_135018

theorem integer_between_sqrt27_and_7 (x : ℤ) :
  (Real.sqrt 27 < x) ∧ (x < 7) → x = 6 := by
  sorry

end integer_between_sqrt27_and_7_l1350_135018


namespace divisibility_by_three_l1350_135042

theorem divisibility_by_three (a b : ℕ) : 
  (3 ∣ (a * b)) → (3 ∣ a) ∨ (3 ∣ b) := by
  sorry

end divisibility_by_three_l1350_135042


namespace cookie_area_theorem_l1350_135035

/-- Represents a rectangular cookie with length and width -/
structure Cookie where
  length : ℝ
  width : ℝ

/-- Calculates the area of a cookie -/
def Cookie.area (c : Cookie) : ℝ := c.length * c.width

/-- Calculates the circumference of two cookies placed horizontally -/
def combined_circumference (c : Cookie) : ℝ := 2 * (2 * c.length + c.width)

theorem cookie_area_theorem (c : Cookie) 
  (h1 : combined_circumference c = 70)
  (h2 : c.width = 15) : 
  c.area = 150 := by
  sorry

end cookie_area_theorem_l1350_135035


namespace root_implies_a_value_l1350_135075

theorem root_implies_a_value (a : ℝ) : (2 * (-1)^2 + a * (-1) - 1 = 0) → a = 1 := by
  sorry

end root_implies_a_value_l1350_135075


namespace jennys_bottle_cap_distance_l1350_135073

theorem jennys_bottle_cap_distance (x : ℝ) : 
  (x + (1/3) * x) + 21 = (15 + 2 * 15) → x = 18 := by sorry

end jennys_bottle_cap_distance_l1350_135073


namespace expression_equality_l1350_135083

theorem expression_equality : 
  2013 * (2015/2014) + 2014 * (2016/2015) + 4029/(2014 * 2015) = 4029 := by
  sorry

end expression_equality_l1350_135083


namespace average_attendance_theorem_l1350_135066

/-- Calculates the average daily attendance for a week given the attendance data --/
def averageDailyAttendance (
  mondayAttendance : ℕ)
  (tuesdayAttendance : ℕ)
  (wednesdayToFridayAttendance : ℕ)
  (saturdayAttendance : ℕ)
  (sundayAttendance : ℕ)
  (absenteesJoiningWednesday : ℕ)
  (tuesdayOnlyAttendees : ℕ) : ℚ :=
  let totalAttendance := 
    mondayAttendance + 
    tuesdayAttendance + 
    (wednesdayToFridayAttendance + absenteesJoiningWednesday) + 
    wednesdayToFridayAttendance * 2 + 
    saturdayAttendance + 
    sundayAttendance
  totalAttendance / 7

/-- Theorem stating that the average daily attendance is 78/7 given the specific attendance data --/
theorem average_attendance_theorem :
  averageDailyAttendance 10 15 10 8 12 3 2 = 78 / 7 := by
  sorry

end average_attendance_theorem_l1350_135066


namespace equation_solution_l1350_135003

theorem equation_solution : ∃! x : ℝ, x ≥ 2 ∧ 
  Real.sqrt (x + 5 - 6 * Real.sqrt (x - 2)) + Real.sqrt (x + 10 - 8 * Real.sqrt (x - 2)) = 3 ∧ 
  x = 44.25 := by
  sorry

end equation_solution_l1350_135003


namespace f_20_5_l1350_135055

/-- 
  f(n, m) represents the number of possible increasing arithmetic sequences 
  that can be formed by selecting m terms from the numbers 1, 2, 3, ..., n
-/
def f (n m : ℕ) : ℕ :=
  sorry

/-- Helper function to check if a sequence is valid -/
def is_valid_sequence (seq : List ℕ) (n : ℕ) : Prop :=
  sorry

theorem f_20_5 : f 20 5 = 40 := by
  sorry

end f_20_5_l1350_135055


namespace harmonic_sum_equals_one_third_l1350_135045

-- Define the harmonic number sequence
def H : ℕ → ℚ
  | 0 => 0
  | n + 1 => H n + 1 / (n + 1)

-- Define the summand of the series
def summand (n : ℕ) : ℚ := 1 / ((n + 2 : ℚ) * H (n + 1) * H (n + 2))

-- State the theorem
theorem harmonic_sum_equals_one_third :
  ∑' n, summand n = 1 / 3 := by sorry

end harmonic_sum_equals_one_third_l1350_135045


namespace g_3_6_neg1_eq_one_seventh_l1350_135071

/-- The function g as defined in the problem -/
def g (a b c : ℚ) : ℚ := (2 * c + a) / (b - c)

/-- Theorem stating that g(3, 6, -1) = 1/7 -/
theorem g_3_6_neg1_eq_one_seventh : g 3 6 (-1) = 1/7 := by sorry

end g_3_6_neg1_eq_one_seventh_l1350_135071


namespace x_value_l1350_135047

theorem x_value (w y z x : ℤ) 
  (hw : w = 90)
  (hz : z = 4 * w + 40)
  (hy : y = 3 * z + 15)
  (hx : x = 2 * y + 6) : 
  x = 2436 := by
  sorry

end x_value_l1350_135047


namespace intersection_of_M_and_N_l1350_135094

def M : Set ℕ := {1, 2, 3, 4}
def N : Set ℕ := {2, 4, 6, 8}

theorem intersection_of_M_and_N :
  M ∩ N = {2, 4} := by sorry

end intersection_of_M_and_N_l1350_135094


namespace quadratic_inequality_solution_set_l1350_135008

theorem quadratic_inequality_solution_set :
  let f : ℝ → ℝ := fun x ↦ (x + 2) * (x - 3)
  {x : ℝ | f x < 0} = {x : ℝ | -2 < x ∧ x < 3} := by sorry

end quadratic_inequality_solution_set_l1350_135008


namespace abc_value_l1350_135081

theorem abc_value (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (eq1 : a + 1/b = 5)
  (eq2 : b + 1/c = 2)
  (eq3 : c + 1/a = 3) :
  a * b * c = 1 := by
sorry

end abc_value_l1350_135081


namespace arithmetic_sequence_sum_l1350_135074

/-- An arithmetic sequence. -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The sum of the first four terms of the sequence equals 30. -/
def SumEquals30 (a : ℕ → ℝ) : Prop :=
  a 1 + a 2 + a 3 + a 4 = 30

theorem arithmetic_sequence_sum (a : ℕ → ℝ) 
  (h1 : ArithmeticSequence a) (h2 : SumEquals30 a) : a 2 + a 3 = 15 := by
  sorry

end arithmetic_sequence_sum_l1350_135074


namespace sum_of_a_and_b_l1350_135013

theorem sum_of_a_and_b (a b : ℚ) 
  (eq1 : 2 * a + 5 * b = 47) 
  (eq2 : 7 * a + 2 * b = 54) : 
  a + b = -103 / 31 := by
sorry

end sum_of_a_and_b_l1350_135013


namespace parabola_equation_l1350_135009

/-- A parabola with the given properties has the equation y² = 4x -/
theorem parabola_equation (p : ℝ) (h₁ : p > 0) : 
  (∃ M : ℝ × ℝ, M.1 = 3 ∧ 
   ∃ F : ℝ × ℝ, F.1 = p/2 ∧ F.2 = 0 ∧ 
   (M.1 - F.1)^2 + (M.2 - F.2)^2 = (2*p)^2) →
  (∀ x y : ℝ, y^2 = 2*p*x ↔ y^2 = 4*x) :=
by sorry

end parabola_equation_l1350_135009


namespace gear_speed_ratio_l1350_135099

/-- Represents the number of teeth on a gear -/
structure Gear where
  teeth : ℕ

/-- Represents the angular speed of a gear in revolutions per minute -/
structure AngularSpeed where
  rpm : ℝ

/-- Represents a system of four meshed gears -/
structure GearSystem where
  A : Gear
  B : Gear
  C : Gear
  D : Gear

/-- The theorem stating the ratio of angular speeds for four meshed gears -/
theorem gear_speed_ratio (system : GearSystem) 
  (ωA ωB ωC ωD : AngularSpeed) :
  ωA.rpm * system.A.teeth = ωB.rpm * system.B.teeth ∧
  ωB.rpm * system.B.teeth = ωC.rpm * system.C.teeth ∧
  ωC.rpm * system.C.teeth = ωD.rpm * system.D.teeth →
  ∃ (k : ℝ), k > 0 ∧
    ωA.rpm = k * (system.B.teeth * system.C.teeth * system.D.teeth) ∧
    ωB.rpm = k * (system.A.teeth * system.C.teeth * system.D.teeth) ∧
    ωC.rpm = k * (system.A.teeth * system.B.teeth * system.D.teeth) ∧
    ωD.rpm = k * (system.A.teeth * system.B.teeth * system.C.teeth) :=
by sorry

end gear_speed_ratio_l1350_135099


namespace at_least_one_is_one_l1350_135054

theorem at_least_one_is_one (a b c : ℝ) 
  (sum_eq : a + b + c = 1/a + 1/b + 1/c) 
  (product_eq : a * b * c = 1) : 
  a = 1 ∨ b = 1 ∨ c = 1 := by
  sorry

end at_least_one_is_one_l1350_135054


namespace intersection_parallel_line_l1350_135069

/-- The equation of a line passing through the intersection of two given lines and parallel to a third line -/
theorem intersection_parallel_line (a b c d e f g h i : ℝ) :
  (∃ x y : ℝ, a * x + b * y + c = 0 ∧ d * x + e * y + f = 0) →  -- Intersection exists
  (g ≠ 0 ∨ h ≠ 0) →  -- Third line is not degenerate
  (a * h ≠ b * g ∨ d * h ≠ e * g) →  -- At least one of the first two lines is not parallel to the third
  (∃ k : ℝ, k ≠ 0 ∧ 
    ∀ x y : ℝ, (a * x + b * y + c = 0 ∧ d * x + e * y + f = 0) →
    ∃ t : ℝ, g * x + h * y + i + t * (a * x + b * y + c) = 0 ∧
            g * x + h * y + i + t * (d * x + e * y + f) = 0) →
  ∃ j : ℝ, ∀ x y : ℝ, 
    (a * x + b * y + c = 0 ∧ d * x + e * y + f = 0) →
    g * x + h * y + j = 0
  := by sorry

end intersection_parallel_line_l1350_135069
