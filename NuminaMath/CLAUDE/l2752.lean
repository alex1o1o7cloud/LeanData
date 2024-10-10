import Mathlib

namespace car_material_cost_is_100_l2752_275224

/-- Represents the factory's production and sales data -/
structure FactoryData where
  car_production : Nat
  car_price : Nat
  motorcycle_production : Nat
  motorcycle_price : Nat
  motorcycle_material_cost : Nat
  profit_difference : Nat

/-- Calculates the cost of materials for car production -/
def calculate_car_material_cost (data : FactoryData) : Nat :=
  data.motorcycle_production * data.motorcycle_price - 
  data.motorcycle_material_cost - 
  (data.car_production * data.car_price - 
  data.profit_difference)

/-- Theorem stating that the cost of materials for car production is $100 -/
theorem car_material_cost_is_100 (data : FactoryData) 
  (h1 : data.car_production = 4)
  (h2 : data.car_price = 50)
  (h3 : data.motorcycle_production = 8)
  (h4 : data.motorcycle_price = 50)
  (h5 : data.motorcycle_material_cost = 250)
  (h6 : data.profit_difference = 50) :
  calculate_car_material_cost data = 100 := by
  sorry

end car_material_cost_is_100_l2752_275224


namespace sqrt_inequality_l2752_275215

theorem sqrt_inequality (m : ℝ) (h : m > 1) :
  Real.sqrt (m + 1) - Real.sqrt m < Real.sqrt m - Real.sqrt (m - 1) := by
  sorry

end sqrt_inequality_l2752_275215


namespace stream_speed_l2752_275210

/-- Proves that given a man rowing 84 km downstream and 60 km upstream, each taking 4 hours, the speed of the stream is 3 km/h. -/
theorem stream_speed (downstream_distance : ℝ) (upstream_distance : ℝ) (time : ℝ) 
  (h1 : downstream_distance = 84)
  (h2 : upstream_distance = 60)
  (h3 : time = 4) :
  let boat_speed := (downstream_distance + upstream_distance) / (2 * time)
  let stream_speed := (downstream_distance - upstream_distance) / (4 * time)
  stream_speed = 3 := by
  sorry

end stream_speed_l2752_275210


namespace factorization_proof_l2752_275202

theorem factorization_proof (m n : ℝ) : m^2 - m*n = m*(m - n) := by
  sorry

end factorization_proof_l2752_275202


namespace w_over_y_value_l2752_275265

theorem w_over_y_value (w x y : ℝ) 
  (h1 : w / x = 2 / 3)
  (h2 : (x + y) / y = 1.6) :
  w / y = 0.4 := by
sorry

end w_over_y_value_l2752_275265


namespace arithmetic_geometric_subset_l2752_275239

/-- An arithmetic sequence with specific properties -/
structure ArithmeticSequence where
  a : ℕ → ℚ
  is_arithmetic : ∀ n, a (n + 1) - a n = a 2 - a 1
  a3_eq_3 : a 3 = 3
  a5_eq_6 : a 5 = 6
  geometric_subset : ∃ m, (a 3) * (a m) = (a 5)^2

/-- The theorem stating that m = 9 for the given conditions -/
theorem arithmetic_geometric_subset (seq : ArithmeticSequence) :
  ∃ m, seq.a m = 12 ∧ m = 9 := by
  sorry

end arithmetic_geometric_subset_l2752_275239


namespace tarantula_legs_tarantula_leg_count_l2752_275238

/-- The number of tarantulas in one egg sac -/
def tarantulas_per_sac : ℕ := 1000

/-- The number of baby tarantula legs in one less than 5 egg sacs -/
def total_legs : ℕ := 32000

/-- The number of egg sacs containing the total legs -/
def num_sacs : ℕ := 5 - 1

/-- Proves that a tarantula has 8 legs -/
theorem tarantula_legs : ℕ :=
  8

/-- Proves that the number of legs a tarantula has is 8 -/
theorem tarantula_leg_count : tarantula_legs = 8 := by
  sorry

end tarantula_legs_tarantula_leg_count_l2752_275238


namespace max_pairs_after_loss_l2752_275263

/-- Represents the types of shoes -/
inductive ShoeType
| Sneaker
| Sandal
| Boot

/-- Represents the colors of shoes -/
inductive ShoeColor
| Red
| Blue
| Green
| Black

/-- Represents the sizes of shoes -/
inductive ShoeSize
| Size6
| Size7
| Size8

/-- Represents a shoe with its type, color, and size -/
structure Shoe :=
  (type : ShoeType)
  (color : ShoeColor)
  (size : ShoeSize)

/-- Represents the initial collection of shoes -/
def initial_collection : Finset Shoe := sorry

/-- The number of shoes lost -/
def shoes_lost : Nat := 9

/-- Theorem stating the maximum number of complete pairs after losing shoes -/
theorem max_pairs_after_loss :
  ∃ (remaining_collection : Finset Shoe),
    remaining_collection ⊆ initial_collection ∧
    (initial_collection.card - remaining_collection.card = shoes_lost) ∧
    (∀ (s : Shoe), s ∈ remaining_collection →
      ∃ (s' : Shoe), s' ∈ remaining_collection ∧ s ≠ s' ∧
        s.type = s'.type ∧ s.color = s'.color ∧ s.size = s'.size) ∧
    remaining_collection.card = 36 := by
  sorry

end max_pairs_after_loss_l2752_275263


namespace cosine_adjacent_extrema_distance_l2752_275204

/-- The distance between adjacent highest and lowest points on the graph of y = cos(x+1) is √(π² + 4) -/
theorem cosine_adjacent_extrema_distance : 
  let f : ℝ → ℝ := λ x => Real.cos (x + 1)
  ∃ (x₁ x₂ : ℝ), x₁ < x₂ ∧ x₂ - x₁ = π ∧
    f x₁ = 1 ∧ f x₂ = -1 ∧
    Real.sqrt (π^2 + 4) = Real.sqrt ((x₂ - x₁)^2 + (f x₁ - f x₂)^2) :=
by sorry

end cosine_adjacent_extrema_distance_l2752_275204


namespace cubic_factorization_sum_of_squares_l2752_275276

theorem cubic_factorization_sum_of_squares (p q r s t u : ℤ) :
  (∀ x : ℝ, 729 * x^3 + 64 = (p * x^2 + q * x + r) * (s * x^2 + t * x + u)) →
  p^2 + q^2 + r^2 + s^2 + t^2 + u^2 = 8210 :=
by sorry

end cubic_factorization_sum_of_squares_l2752_275276


namespace square_circle_area_ratio_l2752_275289

theorem square_circle_area_ratio (s r : ℝ) (h : s > 0) (k : r > 0) 
  (perimeter_relation : 4 * s = 2 * π * r) : 
  (s^2) / (π * r^2) = π := by
  sorry

end square_circle_area_ratio_l2752_275289


namespace election_majority_proof_l2752_275242

/-- 
In an election with a total of 4500 votes, where the winning candidate receives 60% of the votes,
prove that the majority of votes by which the candidate won is 900.
-/
theorem election_majority_proof (total_votes : ℕ) (winning_percentage : ℚ) : 
  total_votes = 4500 →
  winning_percentage = 60 / 100 →
  (winning_percentage * total_votes : ℚ).num / (winning_percentage * total_votes : ℚ).den -
  ((1 - winning_percentage) * total_votes : ℚ).num / ((1 - winning_percentage) * total_votes : ℚ).den = 900 := by
  sorry

end election_majority_proof_l2752_275242


namespace quadratic_equation_solution_l2752_275232

theorem quadratic_equation_solution : 
  ∃ x₁ x₂ : ℝ, x₁ = 2 + Real.sqrt 2 ∧ x₂ = 2 - Real.sqrt 2 ∧
  (x₁^2 - 4*x₁ + 2 = 0) ∧ (x₂^2 - 4*x₂ + 2 = 0) := by
  sorry

end quadratic_equation_solution_l2752_275232


namespace min_distance_to_one_l2752_275269

open Complex

/-- Given a complex number z satisfying the equation, the minimum value of |z - 1| is √2 -/
theorem min_distance_to_one (z : ℂ) 
  (h : Complex.abs ((z^2 + 1) / (z + I)) + Complex.abs ((z^2 + 4*I - 3) / (z - I + 2)) = 4) :
  ∃ (min_dist : ℝ), (∀ (w : ℂ), Complex.abs ((w^2 + 1) / (w + I)) + Complex.abs ((w^2 + 4*I - 3) / (w - I + 2)) = 4 → 
    Complex.abs (w - 1) ≥ min_dist) ∧ min_dist = Real.sqrt 2 :=
sorry

end min_distance_to_one_l2752_275269


namespace age_squares_sum_l2752_275209

theorem age_squares_sum (T J A : ℕ) 
  (sum_TJ : T + J = 23)
  (sum_JA : J + A = 24)
  (sum_TA : T + A = 25) :
  T^2 + J^2 + A^2 = 434 := by
sorry

end age_squares_sum_l2752_275209


namespace intersection_of_A_and_B_l2752_275217

-- Define the sets A and B
def A : Set ℝ := {x | x - 1 > 1}
def B : Set ℝ := {x | x < 3}

-- State the theorem
theorem intersection_of_A_and_B :
  A ∩ B = {x : ℝ | 2 < x ∧ x < 3} := by sorry

end intersection_of_A_and_B_l2752_275217


namespace number_added_to_2q_l2752_275280

theorem number_added_to_2q (x y q : ℤ) (some_number : ℤ) : 
  x = some_number + 2 * q →
  y = 4 * q + 41 →
  (q = 7 → x = y) →
  some_number = 55 := by
sorry

end number_added_to_2q_l2752_275280


namespace inscribed_squares_ratio_l2752_275291

/-- An isosceles right triangle with leg length 6 -/
structure IsoscelesRightTriangle where
  leg_length : ℝ
  leg_length_eq : leg_length = 6

/-- A square inscribed in the triangle with one vertex at the right angle -/
structure InscribedSquareA (triangle : IsoscelesRightTriangle) where
  side_length : ℝ
  vertex_at_right_angle : True
  side_along_leg : True

/-- A square inscribed in the triangle with one side along the other leg -/
structure InscribedSquareB (triangle : IsoscelesRightTriangle) where
  side_length : ℝ
  side_along_leg : True

/-- The theorem statement -/
theorem inscribed_squares_ratio 
  (triangle : IsoscelesRightTriangle) 
  (square_a : InscribedSquareA triangle) 
  (square_b : InscribedSquareB triangle) : 
  square_a.side_length / square_b.side_length = 1 := by
  sorry

end inscribed_squares_ratio_l2752_275291


namespace AB_squared_is_8_l2752_275216

/-- The parabola equation -/
def parabola (x y : ℝ) : Prop := y = 3 * x^2 + 4 * x + 2

/-- Point A on the parabola -/
def A : ℝ × ℝ := sorry

/-- Point B on the parabola -/
def B : ℝ × ℝ := sorry

/-- The origin is the midpoint of AB -/
axiom origin_is_midpoint : (A.1 + B.1) / 2 = 0 ∧ (A.2 + B.2) / 2 = 0

/-- A and B are on the parabola -/
axiom A_on_parabola : parabola A.1 A.2
axiom B_on_parabola : parabola B.1 B.2

/-- The square of the length of AB -/
def AB_squared : ℝ := (A.1 - B.1)^2 + (A.2 - B.2)^2

/-- Theorem: The square of the length of AB is 8 -/
theorem AB_squared_is_8 : AB_squared = 8 := sorry

end AB_squared_is_8_l2752_275216


namespace x_leq_y_l2752_275252

theorem x_leq_y (a b c : ℝ) (h1 : a > b) (h2 : b > c) :
  Real.sqrt ((a - b) * (b - c)) ≤ (a - c) / 2 := by
  sorry

end x_leq_y_l2752_275252


namespace pairing_probability_l2752_275231

theorem pairing_probability (n : ℕ) (h : n = 28) :
  (1 : ℚ) / (n - 1) = 1 / 27 :=
sorry

end pairing_probability_l2752_275231


namespace nail_trimming_sounds_l2752_275235

/-- The number of nails per customer -/
def nails_per_customer : ℕ := 20

/-- The number of customers -/
def num_customers : ℕ := 6

/-- The total number of nail trimming sounds -/
def total_sounds : ℕ := nails_per_customer * num_customers

theorem nail_trimming_sounds : total_sounds = 120 := by
  sorry

end nail_trimming_sounds_l2752_275235


namespace johns_allowance_l2752_275298

/-- John's weekly allowance problem -/
theorem johns_allowance (A : ℚ) : 
  (3/5 : ℚ) * A + (1/3 : ℚ) * (A - (3/5 : ℚ) * A) + (9/10 : ℚ) = A → A = (27/8 : ℚ) := by
  sorry

end johns_allowance_l2752_275298


namespace distance_after_walking_l2752_275268

/-- The distance between two people walking in opposite directions for 1.5 hours -/
theorem distance_after_walking (jay_speed : ℝ) (paul_speed : ℝ) (time : ℝ) : 
  jay_speed = 0.75 * (60 / 15) →  -- Jay's speed in miles per hour
  paul_speed = 2.5 * (60 / 30) →  -- Paul's speed in miles per hour
  time = 1.5 →                    -- Time in hours
  jay_speed * time + paul_speed * time = 12 := by
  sorry

end distance_after_walking_l2752_275268


namespace units_digit_of_expression_l2752_275205

theorem units_digit_of_expression : ∃ n : ℕ, (13 + Real.sqrt 196)^21 + (13 - Real.sqrt 196)^21 = 10 * n + 3 := by
  sorry

end units_digit_of_expression_l2752_275205


namespace number_thought_of_l2752_275203

theorem number_thought_of (x : ℝ) : (x / 5 + 8 = 61) → x = 265 := by
  sorry

end number_thought_of_l2752_275203


namespace f_evaluation_l2752_275287

def f (x : ℝ) : ℝ := 3 * x^3 - 2 * x^2 + 4 * x - 7

theorem f_evaluation : 2 * f 2 + 3 * f (-2) = -107 := by
  sorry

end f_evaluation_l2752_275287


namespace evaluate_expression_l2752_275275

theorem evaluate_expression : 3^0 + 9^5 / 9^3 = 82 := by
  sorry

end evaluate_expression_l2752_275275


namespace trajectory_of_M_l2752_275230

/-- Given points A(-1,0) and B(1,0), and a point M(x,y), if the ratio of the slope of AM
    to the slope of BM is 3, then x = -2 -/
theorem trajectory_of_M (x y : ℝ) (hx : x ≠ 1 ∧ x ≠ -1) (hy : y ≠ 0) :
  (y / (x + 1)) / (y / (x - 1)) = 3 → x = -2 := by
  sorry

#check trajectory_of_M

end trajectory_of_M_l2752_275230


namespace trapezium_height_l2752_275237

theorem trapezium_height (a b area : ℝ) (ha : a > 0) (hb : b > 0) (harea : area > 0) :
  a = 4 → b = 5 → area = 27 →
  (area = (a + b) * (area / ((a + b) / 2)) / 2) →
  area / ((a + b) / 2) = 6 := by
  sorry

end trapezium_height_l2752_275237


namespace sqrt_inequality_l2752_275257

theorem sqrt_inequality (x : ℝ) : 0 < x → (Real.sqrt (x + 1) < 3 * x - 2 ↔ x > 3) := by
  sorry

end sqrt_inequality_l2752_275257


namespace quadrilateral_area_is_2007_l2752_275284

/-- The area of a quadrilateral with vertices at (1, 3), (1, 1), (3, 1), and (2007, 2008) -/
def quadrilateral_area : ℝ :=
  let A : ℝ × ℝ := (1, 3)
  let B : ℝ × ℝ := (1, 1)
  let C : ℝ × ℝ := (3, 1)
  let D : ℝ × ℝ := (2007, 2008)
  -- Area calculation goes here
  0  -- Placeholder, replace with actual calculation

/-- Theorem stating that the area of the quadrilateral is 2007 square units -/
theorem quadrilateral_area_is_2007 : quadrilateral_area = 2007 := by
  sorry

end quadrilateral_area_is_2007_l2752_275284


namespace average_correction_l2752_275212

def correct_average (num_students : ℕ) (initial_average : ℚ) (wrong_mark : ℚ) (correct_mark : ℚ) : ℚ :=
  (num_students * initial_average - (wrong_mark - correct_mark)) / num_students

theorem average_correction (num_students : ℕ) (initial_average : ℚ) (wrong_mark : ℚ) (correct_mark : ℚ)
  (h1 : num_students = 30)
  (h2 : initial_average = 60)
  (h3 : wrong_mark = 90)
  (h4 : correct_mark = 15) :
  correct_average num_students initial_average wrong_mark correct_mark = 57.5 := by
sorry

end average_correction_l2752_275212


namespace hundredthDigitOf7Over33_l2752_275267

-- Define the fraction
def f : ℚ := 7 / 33

-- Define a function to get the nth digit after the decimal point
noncomputable def nthDigitAfterDecimal (q : ℚ) (n : ℕ) : ℕ := sorry

-- Theorem statement
theorem hundredthDigitOf7Over33 : nthDigitAfterDecimal f 100 = 1 := by sorry

end hundredthDigitOf7Over33_l2752_275267


namespace sin_cos_sum_l2752_275219

theorem sin_cos_sum (θ : Real) (h : Real.sin θ * Real.cos θ = 1/8) :
  Real.sin θ + Real.cos θ = Real.sqrt 5 / 2 := by
  sorry

end sin_cos_sum_l2752_275219


namespace fourth_root_256_times_cube_root_64_times_sqrt_16_l2752_275295

theorem fourth_root_256_times_cube_root_64_times_sqrt_16 : 
  (256 : ℝ) ^ (1/4) * (64 : ℝ) ^ (1/3) * (16 : ℝ) ^ (1/2) = 32 := by sorry

end fourth_root_256_times_cube_root_64_times_sqrt_16_l2752_275295


namespace trig_inequality_l2752_275256

theorem trig_inequality (x y z : ℝ) (h1 : 0 < z) (h2 : z < y) (h3 : y < x) (h4 : x < π/2) :
  π/2 + 2 * Real.sin x * Real.cos y + 2 * Real.sin y * Real.cos z > 
  Real.sin (2*x) + Real.sin (2*y) + Real.sin (2*z) := by
  sorry

end trig_inequality_l2752_275256


namespace product_sum_and_32_l2752_275247

theorem product_sum_and_32 : (12 + 25 + 52 + 21) * 32 = 3520 := by
  sorry

end product_sum_and_32_l2752_275247


namespace rock_collecting_contest_l2752_275278

/-- The rock collecting contest between Sydney and Conner --/
theorem rock_collecting_contest 
  (sydney_initial : ℕ) 
  (conner_initial : ℕ) 
  (sydney_day1 : ℕ) 
  (conner_day1_multiplier : ℕ) 
  (conner_day2 : ℕ) 
  (sydney_day3_multiplier : ℕ) 
  (h1 : sydney_initial = 837) 
  (h2 : conner_initial = 723) 
  (h3 : sydney_day1 = 4) 
  (h4 : conner_day1_multiplier = 8) 
  (h5 : conner_day2 = 123) 
  (h6 : sydney_day3_multiplier = 2) : 
  ∃ (conner_day3 : ℕ), conner_day3 ≥ 27 ∧ 
    conner_initial + conner_day1_multiplier * sydney_day1 + conner_day2 + conner_day3 ≥ 
    sydney_initial + sydney_day1 + sydney_day3_multiplier * (conner_day1_multiplier * sydney_day1) :=
by
  sorry

end rock_collecting_contest_l2752_275278


namespace employee_count_l2752_275277

theorem employee_count : 
  ∀ (E : ℕ) (M : ℝ),
    M = 0.99 * (E : ℝ) →
    (M - 99.99999999999991) / (E : ℝ) = 0.98 →
    E = 10000 :=
by sorry

end employee_count_l2752_275277


namespace units_digit_of_S_is_3_l2752_275218

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def S : ℕ := (List.range 12).map (λ i => factorial (i + 1)) |>.sum

theorem units_digit_of_S_is_3 : S % 10 = 3 := by
  sorry

end units_digit_of_S_is_3_l2752_275218


namespace maggots_eaten_second_feeding_correct_l2752_275261

/-- Given the total number of maggots served, the number of maggots laid out and eaten in the first feeding,
    and the number laid out in the second feeding, calculate the number of maggots eaten in the second feeding. -/
def maggots_eaten_second_feeding (total_served : ℕ) (first_feeding_laid_out : ℕ) (first_feeding_eaten : ℕ) (second_feeding_laid_out : ℕ) : ℕ :=
  total_served - first_feeding_eaten - second_feeding_laid_out

/-- Theorem stating that the number of maggots eaten in the second feeding is correct -/
theorem maggots_eaten_second_feeding_correct
  (total_served : ℕ)
  (first_feeding_laid_out : ℕ)
  (first_feeding_eaten : ℕ)
  (second_feeding_laid_out : ℕ)
  (h1 : total_served = 20)
  (h2 : first_feeding_laid_out = 10)
  (h3 : first_feeding_eaten = 1)
  (h4 : second_feeding_laid_out = 10) :
  maggots_eaten_second_feeding total_served first_feeding_laid_out first_feeding_eaten second_feeding_laid_out = 9 :=
by
  sorry

end maggots_eaten_second_feeding_correct_l2752_275261


namespace sum_of_solutions_is_zero_l2752_275223

theorem sum_of_solutions_is_zero (x₁ x₂ : ℝ) : 
  (|x₁ - 20| + |x₁ + 20| = 2020) ∧ 
  (|x₂ - 20| + |x₂ + 20| = 2020) ∧ 
  (∀ x : ℝ, |x - 20| + |x + 20| = 2020 → x = x₁ ∨ x = x₂) →
  x₁ + x₂ = 0 := by
sorry

end sum_of_solutions_is_zero_l2752_275223


namespace gcf_36_54_l2752_275296

theorem gcf_36_54 : Nat.gcd 36 54 = 18 := by
  sorry

end gcf_36_54_l2752_275296


namespace total_planting_area_is_2600_l2752_275229

/-- Represents the number of trees to be planted for each tree chopped -/
structure PlantingRatio :=
  (oak : ℕ)
  (pine : ℕ)

/-- Represents the number of trees chopped in each half of the year -/
structure TreesChopped :=
  (oak : ℕ)
  (pine : ℕ)

/-- Represents the space required for planting each type of tree -/
structure PlantingSpace :=
  (oak : ℕ)
  (pine : ℕ)

/-- Calculates the total area needed for tree planting during the entire year -/
def totalPlantingArea (ratio : PlantingRatio) (firstHalf : TreesChopped) (secondHalf : TreesChopped) (space : PlantingSpace) : ℕ :=
  let oakArea := (firstHalf.oak * ratio.oak * space.oak)
  let pineArea := ((firstHalf.pine + secondHalf.pine) * ratio.pine * space.pine)
  oakArea + pineArea

/-- Theorem stating that the total area needed for tree planting is 2600 m² -/
theorem total_planting_area_is_2600 (ratio : PlantingRatio) (firstHalf : TreesChopped) (secondHalf : TreesChopped) (space : PlantingSpace) :
  ratio.oak = 4 →
  ratio.pine = 2 →
  firstHalf.oak = 100 →
  firstHalf.pine = 100 →
  secondHalf.oak = 150 →
  secondHalf.pine = 150 →
  space.oak = 4 →
  space.pine = 2 →
  totalPlantingArea ratio firstHalf secondHalf space = 2600 :=
by
  sorry

end total_planting_area_is_2600_l2752_275229


namespace yellow_roses_count_l2752_275297

/-- The number of yellow roses on the third rose bush -/
def yellow_roses : ℕ := 20

theorem yellow_roses_count :
  let red_roses : ℕ := 12
  let pink_roses : ℕ := 18
  let orange_roses : ℕ := 8
  let red_picked : ℕ := red_roses / 2
  let pink_picked : ℕ := pink_roses / 2
  let orange_picked : ℕ := orange_roses / 4
  let yellow_picked : ℕ := yellow_roses / 4
  let total_picked : ℕ := 22
  red_picked + pink_picked + orange_picked + yellow_picked = total_picked →
  yellow_roses = 20 := by
sorry

end yellow_roses_count_l2752_275297


namespace solve_jump_rope_problem_l2752_275244

def jump_rope_problem (cindy_time betsy_time tina_time : ℝ) : Prop :=
  cindy_time = 12 ∧
  tina_time = 3 * betsy_time ∧
  tina_time = cindy_time + 6 ∧
  betsy_time / cindy_time = 1 / 2

theorem solve_jump_rope_problem :
  ∃ (betsy_time tina_time : ℝ),
    jump_rope_problem 12 betsy_time tina_time :=
by
  sorry

end solve_jump_rope_problem_l2752_275244


namespace infinitely_many_divisible_by_2009_l2752_275259

/-- Define the sequence a_n -/
def a_seq (a : ℕ+) : ℕ → ℕ
  | 0 => a
  | n + 1 => a_seq a n + 40^(Nat.factorial (n + 1))

/-- Theorem: The sequence a_n has infinitely many numbers divisible by 2009 -/
theorem infinitely_many_divisible_by_2009 (a : ℕ+) :
  ∀ k : ℕ, ∃ n > k, 2009 ∣ a_seq a n :=
sorry

end infinitely_many_divisible_by_2009_l2752_275259


namespace hockey_league_games_l2752_275213

/-- The number of teams in the hockey league -/
def num_teams : ℕ := 19

/-- The number of times each team faces every other team -/
def games_per_pair : ℕ := 10

/-- The total number of games played in the season -/
def total_games : ℕ := num_teams * (num_teams - 1) / 2 * games_per_pair

theorem hockey_league_games :
  total_games = 1710 :=
sorry

end hockey_league_games_l2752_275213


namespace q_n_limit_zero_l2752_275270

def q_n (n : ℕ+) : ℕ := Nat.minFac (n + 1)

theorem q_n_limit_zero : 
  ∀ ε > 0, ∃ N : ℕ, ∀ n : ℕ+, n.val > N → (q_n n : ℝ) / n.val < ε :=
sorry

end q_n_limit_zero_l2752_275270


namespace profit_maximization_l2752_275246

/-- Represents the price reduction in yuan -/
def x : ℝ := 2.5

/-- Represents the initial selling price in yuan -/
def initial_price : ℝ := 60

/-- Represents the cost price in yuan -/
def cost_price : ℝ := 40

/-- Represents the initial weekly sales in items -/
def initial_sales : ℝ := 300

/-- Represents the increase in sales for each yuan of price reduction -/
def sales_increase_rate : ℝ := 20

/-- The profit function based on the price reduction x -/
def profit_function (x : ℝ) : ℝ :=
  (initial_price - x) * (initial_sales + sales_increase_rate * x) -
  cost_price * (initial_sales + sales_increase_rate * x)

/-- The maximum profit achieved -/
def max_profit : ℝ := 6125

theorem profit_maximization :
  profit_function x = max_profit ∧
  ∀ y, 0 ≤ y ∧ y < initial_price - cost_price →
    profit_function y ≤ max_profit :=
by sorry

end profit_maximization_l2752_275246


namespace factorial_ratio_l2752_275290

theorem factorial_ratio : Nat.factorial 11 / Nat.factorial 10 = 11 := by
  sorry

end factorial_ratio_l2752_275290


namespace aftershave_dilution_l2752_275227

/-- Proves that adding 10 ounces of water to a 12-ounce bottle of 60% alcohol solution, 
    then removing 4 ounces of the mixture, results in a 40% alcohol solution. -/
theorem aftershave_dilution (initial_volume : ℝ) (initial_concentration : ℝ) 
  (water_added : ℝ) (removed_amount : ℝ) (final_concentration : ℝ) :
  initial_volume = 12 ∧ 
  initial_concentration = 0.6 ∧ 
  water_added = 10 ∧ 
  removed_amount = 4 ∧ 
  final_concentration = 0.4 →
  let initial_alcohol := initial_volume * initial_concentration
  let total_volume := initial_volume + water_added
  let final_volume := total_volume - removed_amount
  initial_alcohol / final_volume = final_concentration :=
by
  sorry

#check aftershave_dilution

end aftershave_dilution_l2752_275227


namespace car_trade_profit_percentage_l2752_275248

/-- Calculates the profit percentage on the original price when a trader buys a car at a discount and sells it at an increase. -/
theorem car_trade_profit_percentage 
  (original_price : ℝ) 
  (discount_rate : ℝ) 
  (increase_rate : ℝ) 
  (h1 : original_price > 0)
  (h2 : discount_rate = 0.20)
  (h3 : increase_rate = 0.50) : 
  (((1 - discount_rate) * (1 + increase_rate) - 1) * 100 = 20) := by
sorry

end car_trade_profit_percentage_l2752_275248


namespace max_value_of_a_l2752_275254

/-- An odd function that is increasing on the non-negative reals -/
structure OddIncreasingFunction (f : ℝ → ℝ) : Prop where
  odd : ∀ x, f (-x) = -f x
  increasing_nonneg : ∀ x y, 0 ≤ x → x ≤ y → f x ≤ f y

/-- The condition relating f, a, x, and t -/
def condition (f : ℝ → ℝ) (a : ℝ) : Prop :=
  ∀ x t, x ∈ Set.Icc 1 2 → t ∈ Set.Icc 1 2 →
    f (x^2 + a*x + a) ≤ f (-a*t^2 - t + 1)

theorem max_value_of_a (f : ℝ → ℝ) (hf : OddIncreasingFunction f) :
  (∃ a, condition f a) → (∀ a, condition f a → a ≤ -1) ∧ (condition f (-1)) :=
sorry

end max_value_of_a_l2752_275254


namespace apple_difference_l2752_275200

theorem apple_difference (ben_apples phillip_apples tom_apples : ℕ) : 
  ben_apples > phillip_apples →
  tom_apples = (3 * ben_apples) / 8 →
  phillip_apples = 40 →
  tom_apples = 18 →
  ben_apples - phillip_apples = 8 := by
sorry

end apple_difference_l2752_275200


namespace ascending_order_of_a_l2752_275207

theorem ascending_order_of_a (a : ℝ) (h : a^2 - a < 0) :
  -a < -a^2 ∧ -a^2 < a^2 ∧ a^2 < a :=
by sorry

end ascending_order_of_a_l2752_275207


namespace ln_inequality_l2752_275283

-- Define the natural logarithm function
noncomputable def f (x : ℝ) := Real.log x

-- State the theorem
theorem ln_inequality (x : ℝ) (h : x > 0) : f x ≤ x - 1 := by
  -- Define the derivative of f
  have f_deriv : ∀ x > 0, deriv f x = 1 / x := by sorry
  
  -- f(1) = 0
  have f_at_one : f 1 = 0 := by sorry
  
  -- The tangent line at x = 1 is y = x - 1
  have tangent_line : ∀ x, x - 1 = (x - 1) * (deriv f 1) + f 1 := by sorry
  
  -- The tangent line is above the graph of f for x > 0
  have tangent_above : ∀ x > 0, f x ≤ x - 1 := by sorry
  
  -- Apply the tangent_above property to prove the inequality
  exact tangent_above x h

end ln_inequality_l2752_275283


namespace intersection_of_curves_l2752_275279

/-- Prove that if a curve C₁ defined by θ = π/6 (ρ ∈ ℝ) intersects with a curve C₂ defined by 
    x = a + √2 cos θ, y = √2 sin θ (where a > 0) at two points A and B, and the distance |AB| = 2, 
    then a = 2. -/
theorem intersection_of_curves (a : ℝ) (h_a : a > 0) : 
  ∃ (A B : ℝ × ℝ),
    (∃ (ρ₁ ρ₂ : ℝ), 
      A.1 = ρ₁ * Real.cos (π/6) ∧ A.2 = ρ₁ * Real.sin (π/6) ∧
      B.1 = ρ₂ * Real.cos (π/6) ∧ B.2 = ρ₂ * Real.sin (π/6)) ∧
    (∃ (θ₁ θ₂ : ℝ),
      A.1 = a + Real.sqrt 2 * Real.cos θ₁ ∧ A.2 = Real.sqrt 2 * Real.sin θ₁ ∧
      B.1 = a + Real.sqrt 2 * Real.cos θ₂ ∧ B.2 = Real.sqrt 2 * Real.sin θ₂) ∧
    Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 2 →
  a = 2 := by
  sorry


end intersection_of_curves_l2752_275279


namespace sequence_growth_l2752_275258

theorem sequence_growth (a : ℕ → ℤ) (h1 : a 1 > a 0) (h2 : a 1 > 0)
  (h3 : ∀ r : ℕ, r ≤ 98 → a (r + 2) = 3 * a (r + 1) - 2 * a r) :
  a 100 > 299 := by
  sorry

end sequence_growth_l2752_275258


namespace triangular_array_sum_of_digits_l2752_275285

/-- The sum of the first n positive integers -/
def triangular_number (n : ℕ) : ℕ := n * (n + 1) / 2

/-- Sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sum_of_digits (n / 10)

/-- The number of rows in the triangular array -/
def N : ℕ := 77

theorem triangular_array_sum_of_digits :
  (triangular_number N = 3003) ∧ (sum_of_digits N = 14) := by
  sorry

end triangular_array_sum_of_digits_l2752_275285


namespace solve_abc_values_l2752_275222

theorem solve_abc_values (A B : Set ℝ) (a b c : ℝ) :
  A = {x : ℝ | x^2 - a*x - 2 = 0} →
  B = {x : ℝ | x^3 + b*x + c = 0} →
  -2 ∈ A ∩ B →
  A ∩ B = A →
  a = -1 ∧ b = -3 ∧ c = 2 := by
sorry

end solve_abc_values_l2752_275222


namespace paper_tape_overlap_l2752_275271

/-- Given 12 sheets of paper tape, each 18 cm long, glued to form a round loop
    with a perimeter of 210 cm and overlapped by the same length,
    the length of each overlapped part is 5 mm. -/
theorem paper_tape_overlap (num_sheets : ℕ) (sheet_length : ℝ) (perimeter : ℝ) :
  num_sheets = 12 →
  sheet_length = 18 →
  perimeter = 210 →
  (num_sheets * sheet_length - perimeter) / num_sheets * 10 = 5 :=
by sorry

end paper_tape_overlap_l2752_275271


namespace unique_prime_in_range_l2752_275243

theorem unique_prime_in_range : ∃! n : ℕ, 
  70 ≤ n ∧ n ≤ 90 ∧ 
  Nat.gcd n 15 = 5 ∧ 
  Nat.Prime n ∧
  n = 85 := by
sorry

end unique_prime_in_range_l2752_275243


namespace auto_shop_discount_l2752_275241

theorem auto_shop_discount (part_cost : ℕ) (num_parts : ℕ) (total_discount : ℕ) : 
  part_cost = 80 → num_parts = 7 → total_discount = 121 → 
  part_cost * num_parts - total_discount = 439 := by
  sorry

end auto_shop_discount_l2752_275241


namespace ratio_x_to_y_l2752_275251

theorem ratio_x_to_y (x y : ℝ) (h : y = 0.2 * x) : x / y = 5 := by
  sorry

end ratio_x_to_y_l2752_275251


namespace triangle_problem_l2752_275266

/-- Given a triangle ABC with the specified properties, prove AC = 5 and ∠A = 120° --/
theorem triangle_problem (A B C : ℝ) (BC AB AC : ℝ) (angleA : ℝ) :
  BC = 7 →
  AB = 3 →
  (Real.sin C) / (Real.sin B) = 3/5 →
  AC = 5 ∧ angleA = 120 * π / 180 :=
by sorry

end triangle_problem_l2752_275266


namespace calculation_proof_l2752_275226

theorem calculation_proof : (30 / (7 + 2 - 3)) * 4 = 20 := by
  sorry

end calculation_proof_l2752_275226


namespace right_isosceles_triangle_area_l2752_275234

/-- The area of a right isosceles triangle with perimeter 3p -/
theorem right_isosceles_triangle_area (p : ℝ) :
  let a : ℝ := p * (3 / (2 + Real.sqrt 2))
  let b : ℝ := a
  let c : ℝ := Real.sqrt 2 * a
  let perimeter : ℝ := a + b + c
  let area : ℝ := (1 / 2) * a * b
  (perimeter = 3 * p) → (area = (9 * p^2 * (3 - 2 * Real.sqrt 2)) / 4) :=
by sorry

end right_isosceles_triangle_area_l2752_275234


namespace circular_garden_radius_l2752_275211

theorem circular_garden_radius (r : ℝ) (h : r > 0) : 2 * π * r = (1/3) * π * r^2 → r = 6 := by
  sorry

end circular_garden_radius_l2752_275211


namespace no_real_roots_quadratic_l2752_275273

theorem no_real_roots_quadratic (a : ℝ) : 
  (∀ x : ℝ, 3 * x^2 + 2 * a * x + 1 ≠ 0) ↔ a ∈ Set.Ioo (-Real.sqrt 3) (Real.sqrt 3) := by
sorry

end no_real_roots_quadratic_l2752_275273


namespace bbq_ice_cost_chad_bbq_ice_cost_l2752_275288

/-- The cost of ice for a BBQ given the number of people, ice needed per person, and ice price --/
theorem bbq_ice_cost (people : ℕ) (ice_per_person : ℕ) (pack_size : ℕ) (pack_price : ℚ) : ℚ :=
  let total_ice := people * ice_per_person
  let packs_needed := (total_ice + pack_size - 1) / pack_size  -- Ceiling division
  packs_needed * pack_price

/-- Proof that the cost of ice for Chad's BBQ is $9.00 --/
theorem chad_bbq_ice_cost : bbq_ice_cost 15 2 10 3 = 9 := by
  sorry

end bbq_ice_cost_chad_bbq_ice_cost_l2752_275288


namespace second_day_to_full_distance_ratio_l2752_275208

/-- Represents a three-day hike with given distances --/
structure ThreeDayHike where
  fullDistance : ℕ
  firstDayDistance : ℕ
  thirdDayDistance : ℕ

/-- Calculates the second day distance --/
def secondDayDistance (hike : ThreeDayHike) : ℕ :=
  hike.fullDistance - (hike.firstDayDistance + hike.thirdDayDistance)

/-- Theorem: The ratio of the second day distance to the full hike distance is 1:2 --/
theorem second_day_to_full_distance_ratio 
  (hike : ThreeDayHike) 
  (h1 : hike.fullDistance = 50) 
  (h2 : hike.firstDayDistance = 10) 
  (h3 : hike.thirdDayDistance = 15) : 
  (secondDayDistance hike) * 2 = hike.fullDistance := by
  sorry

end second_day_to_full_distance_ratio_l2752_275208


namespace smallest_two_digit_prime_with_reverse_composite_l2752_275253

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(d ∣ n)

def is_composite (n : ℕ) : Prop := n > 1 ∧ ∃ d : ℕ, d > 1 ∧ d < n ∧ d ∣ n

def reverse_digits (n : ℕ) : ℕ :=
  let ones := n % 10
  let tens := n / 10
  ones * 10 + tens

def is_two_digit (n : ℕ) : Prop := n ≥ 10 ∧ n ≤ 99

def tens_digit (n : ℕ) : ℕ := n / 10

theorem smallest_two_digit_prime_with_reverse_composite :
  ∃ (n : ℕ), 
    is_two_digit n ∧ 
    is_prime n ∧ 
    tens_digit n = 2 ∧ 
    is_composite (reverse_digits n) ∧
    (∀ m : ℕ, is_two_digit m → is_prime m → tens_digit m = 2 → 
      is_composite (reverse_digits m) → n ≤ m) ∧
    n = 23 := by sorry

end smallest_two_digit_prime_with_reverse_composite_l2752_275253


namespace quadratic_inequality_range_l2752_275233

theorem quadratic_inequality_range (m : ℝ) : 
  (∀ t : ℝ, m * t^2 - m * t + 4 > 0) ↔ (0 ≤ m ∧ m < 16) :=
sorry

end quadratic_inequality_range_l2752_275233


namespace illuminated_area_ratio_l2752_275282

theorem illuminated_area_ratio (r : ℝ) (h : r > 0) :
  let sphere_radius := r
  let light_distance := 3 * r
  let illuminated_area := 2 * Real.pi * r * (r - r / 4)
  let cone_base_radius := r / 4 * Real.sqrt 15
  let cone_slant_height := r * Real.sqrt 15
  let cone_lateral_area := Real.pi * cone_base_radius * cone_slant_height
  illuminated_area / cone_lateral_area = 2 / 5 := by
sorry

end illuminated_area_ratio_l2752_275282


namespace monotonic_increasing_condition_l2752_275294

/-- Given a function f(x) = (1/3)x^3 + x^2 - ax + 3a that is monotonically increasing
    in the interval [1, 2], prove that a ≤ 3 -/
theorem monotonic_increasing_condition (a : ℝ) :
  (∀ x ∈ Set.Icc 1 2, Monotone (fun x => (1/3) * x^3 + x^2 - a*x + 3*a)) →
  a ≤ 3 := by
  sorry

end monotonic_increasing_condition_l2752_275294


namespace root_in_interval_implies_m_range_l2752_275262

theorem root_in_interval_implies_m_range (m : ℝ) :
  (∃ x, x^2 + 2*x - m = 0 ∧ 2 < x ∧ x < 3) →
  (8 < m ∧ m < 15) := by
  sorry

end root_in_interval_implies_m_range_l2752_275262


namespace hyperbola_eccentricity_l2752_275249

theorem hyperbola_eccentricity 
  (a b c : ℝ) 
  (ha : a > 0) 
  (hb : b > 0) 
  (hyperbola_eq : ∀ x y, x^2 / a^2 - y^2 / b^2 = 1)
  (circle_eq : ∀ x y, x^2 + y^2 = c^2)
  (asymptote_eq : ∀ x, b / a * x = x)
  (point_M : ∃ x y, x^2 + y^2 = c^2 ∧ y = b / a * x ∧ x = a ∧ y = b)
  (distance_condition : ∀ x y, x^2 + y^2 = c^2 ∧ y = b / a * x → 
    Real.sqrt ((x + c)^2 + y^2) - Real.sqrt ((x - c)^2 + y^2) = 2 * b)
  (relation_abc : b^2 = a^2 - c^2)
  (eccentricity_def : c / a = e) :
  e^2 = (Real.sqrt 5 + 1) / 2 :=
by sorry

end hyperbola_eccentricity_l2752_275249


namespace triangle_properties_l2752_275240

-- Define a structure for our triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define our main theorem
theorem triangle_properties (t : Triangle) 
  (h1 : t.b * Real.sin t.A = (3 * t.b - t.c) * Real.sin t.B)
  (h2 : t.a + t.b + t.c = 8) :
  (2 * Real.sin t.A = 3 * Real.sin t.B → t.c = 3) ∧
  (t.a = t.c → Real.cos (2 * t.B) = 17 / 81) := by
  sorry


end triangle_properties_l2752_275240


namespace sum_of_distinct_prime_divisors_1800_l2752_275225

def sum_of_distinct_prime_divisors (n : ℕ) : ℕ :=
  (Nat.factors n).toFinset.sum id

theorem sum_of_distinct_prime_divisors_1800 :
  sum_of_distinct_prime_divisors 1800 = 10 := by
  sorry

end sum_of_distinct_prime_divisors_1800_l2752_275225


namespace charles_reading_time_l2752_275201

-- Define the parameters of the problem
def total_pages : ℕ := 96
def pages_per_day : ℕ := 8

-- Define the function to calculate the number of days
def days_to_finish (total : ℕ) (per_day : ℕ) : ℕ := total / per_day

-- Theorem statement
theorem charles_reading_time : days_to_finish total_pages pages_per_day = 12 := by
  sorry

end charles_reading_time_l2752_275201


namespace intersection_equal_angles_not_always_perpendicular_l2752_275260

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (intersect : Plane → Plane → Line)
variable (angle_with : Line → Plane → ℝ)
variable (perpendicular : Line → Line → Prop)

-- State the theorem
theorem intersection_equal_angles_not_always_perpendicular
  (m n : Line) (α β : Plane) :
  ¬(∀ (m n : Line) (α β : Plane),
    (intersect α β = m) →
    (angle_with n α = angle_with n β) →
    (perpendicular m n)) :=
sorry

end intersection_equal_angles_not_always_perpendicular_l2752_275260


namespace rectangular_prism_diagonals_l2752_275292

/-- A rectangular prism with length 4, width 3, and height 2 -/
structure RectangularPrism where
  length : ℕ := 4
  width : ℕ := 3
  height : ℕ := 2

/-- The number of diagonals in a rectangular prism -/
def num_diagonals (prism : RectangularPrism) : ℕ := sorry

/-- Theorem stating that a rectangular prism with length 4, width 3, and height 2 has 16 diagonals -/
theorem rectangular_prism_diagonals :
  ∀ (prism : RectangularPrism), num_diagonals prism = 16 := by sorry

end rectangular_prism_diagonals_l2752_275292


namespace f_composition_value_l2752_275264

noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 0 then Real.sqrt x else Real.cos x

theorem f_composition_value : f (f (-Real.pi/3)) = Real.sqrt 2 / 2 := by
  sorry

end f_composition_value_l2752_275264


namespace circle_center_theorem_l2752_275255

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

end circle_center_theorem_l2752_275255


namespace quadratic_equation_q_value_l2752_275220

theorem quadratic_equation_q_value : ∀ (p q : ℝ),
  (∃ x : ℝ, 3 * x^2 + p * x + q = 0 ∧ x = -3) →
  (∃ x₁ x₂ : ℝ, 3 * x₁^2 + p * x₁ + q = 0 ∧ 3 * x₂^2 + p * x₂ + q = 0 ∧ x₁ + x₂ = -2) →
  q = -9 := by
sorry

end quadratic_equation_q_value_l2752_275220


namespace geometric_sequence_ratio_l2752_275299

/-- Given a geometric sequence {a_n} with sum S_n of the first n terms,
    if a_3 + 2a_6 = 0, then S_3/S_6 = 2 -/
theorem geometric_sequence_ratio (a : ℕ → ℝ) (S : ℕ → ℝ) :
  (∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q) →  -- Geometric sequence condition
  (∀ n : ℕ, S n = (a 1) * (1 - q^n) / (1 - q)) →  -- Sum formula
  a 3 + 2 * a 6 = 0 →  -- Given condition
  S 3 / S 6 = 2 := by
  sorry

end geometric_sequence_ratio_l2752_275299


namespace gcd_4536_8721_l2752_275214

theorem gcd_4536_8721 : Nat.gcd 4536 8721 = 3 := by
  sorry

end gcd_4536_8721_l2752_275214


namespace find_a_value_l2752_275236

theorem find_a_value (x : ℝ) (a : ℝ) : 
  (2 * x - 3 = 5 * x - 2 * a) → (x = 1) → (a = 3) := by
  sorry

end find_a_value_l2752_275236


namespace balloon_altitude_l2752_275221

/-- Calculates the altitude of a balloon given temperature conditions -/
theorem balloon_altitude 
  (temp_decrease_rate : ℝ) -- Temperature decrease rate per 1000 meters
  (ground_temp : ℝ)        -- Ground temperature in °C
  (balloon_temp : ℝ)       -- Balloon temperature in °C
  (h : temp_decrease_rate = 6)
  (i : ground_temp = 5)
  (j : balloon_temp = -2) :
  (ground_temp - balloon_temp) / temp_decrease_rate = 7 / 6 := by
sorry

end balloon_altitude_l2752_275221


namespace optimal_price_maximizes_profit_l2752_275228

/-- Represents the profit function for a product with given pricing and sales conditions -/
def profit_function (x : ℝ) : ℝ :=
  (x - 8) * (100 - (x - 10) * 10)

/-- The selling price that maximizes profit -/
def optimal_price : ℝ := 14

theorem optimal_price_maximizes_profit :
  ∀ (x : ℝ), profit_function x ≤ profit_function optimal_price :=
sorry

#check optimal_price_maximizes_profit

end optimal_price_maximizes_profit_l2752_275228


namespace max_stores_visited_l2752_275286

theorem max_stores_visited (total_stores : ℕ) (total_visits : ℕ) (double_visitors : ℕ) (total_shoppers : ℕ) 
  (h1 : total_stores = 8)
  (h2 : total_visits = 23)
  (h3 : double_visitors = 8)
  (h4 : total_shoppers = 12)
  (h5 : double_visitors * 2 ≤ total_visits)
  (h6 : ∀ n : ℕ, n ≤ total_shoppers → n > 0) :
  ∃ max_visits : ℕ, max_visits = 7 ∧ 
    (∀ person : ℕ, person ≤ total_shoppers → 
      ∃ visits : ℕ, visits ≤ max_visits ∧ 
        (double_visitors * 2 + (total_shoppers - double_visitors) * visits ≤ total_visits)) :=
by sorry

end max_stores_visited_l2752_275286


namespace mod_fifteen_equivalence_l2752_275281

theorem mod_fifteen_equivalence :
  ∃! n : ℤ, 0 ≤ n ∧ n ≤ 14 ∧ n ≡ 14567 [ZMOD 15] ∧ n = 2 := by
  sorry

end mod_fifteen_equivalence_l2752_275281


namespace find_y_value_l2752_275274

theorem find_y_value (x y : ℕ) (h1 : 2^x - 2^y = 3 * 2^10) (h2 : x = 12) : y = 10 := by
  sorry

end find_y_value_l2752_275274


namespace coffee_price_increase_l2752_275272

/-- Given the conditions of the tea and coffee pricing problem, prove that the price of coffee increased by 100% from June to July. -/
theorem coffee_price_increase (june_price : ℝ) (july_mixture_price : ℝ) (july_tea_price : ℝ) :
  -- In June, the price of green tea and coffee were the same
  -- In July, the price of green tea dropped by 70%
  -- In July, a mixture of equal quantities of green tea and coffee costs $3.45 for 3 lbs
  -- In July, a pound of green tea costs $0.3
  june_price > 0 ∧
  july_mixture_price = 3.45 ∧
  july_tea_price = 0.3 ∧
  july_tea_price = june_price * 0.3 →
  -- The price of coffee increased by 100%
  (((july_mixture_price - 3 * july_tea_price / 2) * 2 / 3 - june_price) / june_price) * 100 = 100 :=
by sorry

end coffee_price_increase_l2752_275272


namespace power_three_mod_five_l2752_275245

theorem power_three_mod_five : 3^2023 % 5 = 2 := by
  sorry

end power_three_mod_five_l2752_275245


namespace quadratic_function_constraint_l2752_275293

theorem quadratic_function_constraint (a b c : ℝ) : 
  (∃ a' ∈ Set.Icc 1 2, ∀ x ∈ Set.Icc 1 2, a' * x^2 + b * x + c ≤ 1) →
  (7 * b + 5 * c ≤ -6 ∧ ∃ b' c', 7 * b' + 5 * c' = -6) :=
by sorry

end quadratic_function_constraint_l2752_275293


namespace train_distance_theorem_l2752_275250

/-- The distance between two trains after 8 hours of travel, given their initial positions and speeds -/
def distance_between_trains (initial_distance : ℝ) (speed1 speed2 : ℝ) (time : ℝ) : Set ℝ :=
  let distance1 := speed1 * time
  let distance2 := speed2 * time
  let diff := distance2 - distance1
  {initial_distance + diff, initial_distance - diff}

/-- Theorem stating the distance between two trains after 8 hours -/
theorem train_distance_theorem :
  distance_between_trains 892 40 48 8 = {956, 828} :=
by sorry

end train_distance_theorem_l2752_275250


namespace locus_is_hyperbola_l2752_275206

/-- Two fixed points in a plane -/
structure FixedPoints (α : Type*) [NormedAddCommGroup α] where
  F₁ : α
  F₂ : α

/-- A point P in the plane satisfying the locus condition -/
structure LocusPoint (α : Type*) [NormedAddCommGroup α] (FP : FixedPoints α) where
  P : α
  k : ℝ
  h_positive : k > 0
  h_less : k < ‖FP.F₁ - FP.F₂‖
  h_condition : ‖P - FP.F₁‖ - ‖P - FP.F₂‖ = k

/-- Definition of a hyperbola -/
def IsHyperbola (α : Type*) [NormedAddCommGroup α] (S : Set α) (FP : FixedPoints α) :=
  ∃ k : ℝ, k > 0 ∧ k < ‖FP.F₁ - FP.F₂‖ ∧
    S = {P | ‖P - FP.F₁‖ - ‖P - FP.F₂‖ = k ∨ ‖P - FP.F₂‖ - ‖P - FP.F₁‖ = k}

/-- The main theorem: The locus of points satisfying the given condition forms a hyperbola -/
theorem locus_is_hyperbola {α : Type*} [NormedAddCommGroup α] (FP : FixedPoints α) :
  IsHyperbola α {P | ∃ LP : LocusPoint α FP, LP.P = P} FP :=
sorry

end locus_is_hyperbola_l2752_275206
