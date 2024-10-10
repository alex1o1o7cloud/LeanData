import Mathlib

namespace similar_walls_length_l2042_204270

/-- Represents the work done to build a wall -/
structure WallWork where
  persons : ℕ
  days : ℕ
  length : ℝ

/-- The theorem stating the relationship between two similar walls -/
theorem similar_walls_length
  (wall1 : WallWork)
  (wall2 : WallWork)
  (h1 : wall1.persons = 18)
  (h2 : wall1.days = 42)
  (h3 : wall2.persons = 30)
  (h4 : wall2.days = 18)
  (h5 : wall2.length = 100)
  (h6 : (wall1.persons * wall1.days) / (wall2.persons * wall2.days) = wall1.length / wall2.length) :
  wall1.length = 140 := by
  sorry

#check similar_walls_length

end similar_walls_length_l2042_204270


namespace original_price_correct_l2042_204296

/-- The original selling price of a shirt before discount -/
def original_price : ℝ := 700

/-- The discount percentage offered by the shop -/
def discount_percentage : ℝ := 20

/-- The price Smith paid for the shirt after discount -/
def discounted_price : ℝ := 560

/-- Theorem stating that the original price is correct given the discount and final price -/
theorem original_price_correct : 
  original_price * (1 - discount_percentage / 100) = discounted_price :=
by sorry

end original_price_correct_l2042_204296


namespace correct_probability_open_l2042_204269

/-- Represents a three-digit combination lock -/
structure CombinationLock :=
  (digits : Fin 3 → Fin 10)

/-- The probability of opening the lock by randomly selecting the last digit -/
def probability_open (lock : CombinationLock) : ℚ :=
  1 / 10

theorem correct_probability_open (lock : CombinationLock) :
  probability_open lock = 1 / 10 := by
  sorry

end correct_probability_open_l2042_204269


namespace painter_problem_l2042_204289

/-- Calculates the total number of rooms to be painted given the painting time per room,
    number of rooms already painted, and remaining painting time. -/
def total_rooms_to_paint (time_per_room : ℕ) (rooms_painted : ℕ) (remaining_time : ℕ) : ℕ :=
  rooms_painted + remaining_time / time_per_room

/-- Proves that the total number of rooms to be painted is 10 given the specific conditions. -/
theorem painter_problem :
  let time_per_room : ℕ := 8
  let rooms_painted : ℕ := 8
  let remaining_time : ℕ := 16
  total_rooms_to_paint time_per_room rooms_painted remaining_time = 10 := by
  sorry

end painter_problem_l2042_204289


namespace perpendicular_lines_m_values_l2042_204224

theorem perpendicular_lines_m_values (m : ℝ) : 
  (∀ x y, m * x + (m + 2) * y - 1 = 0 ∧ (m - 1) * x + m * y = 0 → 
    (m * (m - 1) + (m + 2) * m = 0 ∨ m = 0)) → 
  (m = 0 ∨ m = -1/2) :=
by sorry

end perpendicular_lines_m_values_l2042_204224


namespace smallest_n_for_seven_numbers_l2042_204202

/-- Represents the sequence generation process -/
def generateSequence (n : ℕ) : List ℕ :=
  sorry

/-- Checks if a number is an even square -/
def isEvenSquare (n : ℕ) : Bool :=
  sorry

/-- Finds the largest even square less than or equal to n -/
def largestEvenSquare (n : ℕ) : ℕ :=
  sorry

theorem smallest_n_for_seven_numbers : 
  (∀ m : ℕ, m < 168 → (generateSequence m).length ≠ 7) ∧ 
  (generateSequence 168).length = 7 :=
sorry

end smallest_n_for_seven_numbers_l2042_204202


namespace quadratic_translation_problem_solution_l2042_204206

/-- Represents a horizontal and vertical translation of a quadratic function -/
structure Translation where
  horizontal : ℝ
  vertical : ℝ

/-- Applies a translation to a quadratic function -/
def apply_translation (f : ℝ → ℝ) (t : Translation) : ℝ → ℝ :=
  λ x => f (x + t.horizontal) - t.vertical

theorem quadratic_translation (a : ℝ) (t : Translation) :
  apply_translation (λ x => a * x^2) t =
  λ x => a * (x + t.horizontal)^2 - t.vertical := by
  sorry

/-- The specific translation in the problem -/
def problem_translation : Translation :=
  { horizontal := 3, vertical := 2 }

theorem problem_solution :
  apply_translation (λ x => 2 * x^2) problem_translation =
  λ x => 2 * (x + 3)^2 - 2 := by
  sorry

end quadratic_translation_problem_solution_l2042_204206


namespace remainder_sum_l2042_204233

theorem remainder_sum (n : ℤ) (h : n % 18 = 11) : (n % 2 + n % 9 = 3) := by
  sorry

end remainder_sum_l2042_204233


namespace tan_and_sin_values_l2042_204244

theorem tan_and_sin_values (α : ℝ) (h : Real.tan (α + π / 4) = -3) : 
  Real.tan α = 1 ∧ Real.sin (2 * α + π / 4) = Real.sqrt 2 / 2 := by
  sorry

end tan_and_sin_values_l2042_204244


namespace quadratic_root_property_l2042_204241

theorem quadratic_root_property (m : ℝ) : 
  m^2 - 4*m + 1 = 0 → 2023 - m^2 + 4*m = 2024 := by
  sorry

end quadratic_root_property_l2042_204241


namespace ellipse_equation_l2042_204232

/-- The equation of an ellipse with foci at (-2,0) and (2,0) passing through (2, 5/3) -/
theorem ellipse_equation (x y : ℝ) : 
  (∃ (a b : ℝ), a > 0 ∧ b > 0 ∧
    (∀ (x₀ y₀ : ℝ), x₀^2/a^2 + y₀^2/b^2 = 1 ↔ 
      (Real.sqrt ((x₀ + 2)^2 + y₀^2) + Real.sqrt ((x₀ - 2)^2 + y₀^2) = 2*a)) ∧
    (2^2/a^2 + (5/3)^2/b^2 = 1)) →
  x^2/9 + y^2/5 = 1 :=
by sorry

end ellipse_equation_l2042_204232


namespace inverse_89_mod_91_l2042_204262

theorem inverse_89_mod_91 : ∃ x : ℕ, x < 91 ∧ (89 * x) % 91 = 1 ∧ x = 45 := by
  sorry

end inverse_89_mod_91_l2042_204262


namespace negation_absolute_value_inequality_l2042_204272

theorem negation_absolute_value_inequality :
  (¬ ∀ x : ℝ, |x - 2| < 3) ↔ (∃ x : ℝ, |x - 2| ≥ 3) :=
by sorry

end negation_absolute_value_inequality_l2042_204272


namespace nested_expression_value_l2042_204227

theorem nested_expression_value : (3 * (3 * (3 * (3 * (3 * (3 + 1) + 2) + 3) + 4) + 5) + 6) = 1272 := by
  sorry

end nested_expression_value_l2042_204227


namespace total_pieces_is_11403_l2042_204279

/-- Calculates the total number of pieces in John's puzzles -/
def totalPuzzlePieces : ℕ :=
  let puzzle1 : ℕ := 1000
  let puzzle2 : ℕ := puzzle1 + (puzzle1 * 20 / 100)
  let puzzle3 : ℕ := puzzle2 + (puzzle2 * 50 / 100)
  let puzzle4 : ℕ := puzzle3 + (puzzle3 * 75 / 100)
  let puzzle5 : ℕ := puzzle4 + (puzzle4 * 35 / 100)
  puzzle1 + puzzle2 + puzzle3 + puzzle4 + puzzle5

theorem total_pieces_is_11403 : totalPuzzlePieces = 11403 := by
  sorry

end total_pieces_is_11403_l2042_204279


namespace chips_for_dinner_l2042_204294

theorem chips_for_dinner (dinner : ℕ) (after : ℕ) : 
  dinner > 0 → 
  after > 0 → 
  dinner + after = 3 → 
  dinner = 2 := by
sorry

end chips_for_dinner_l2042_204294


namespace max_airline_services_l2042_204251

theorem max_airline_services (internet_percentage : ℝ) (snack_percentage : ℝ) 
  (h1 : internet_percentage = 35) 
  (h2 : snack_percentage = 70) : 
  ∃ (max_both_percentage : ℝ), max_both_percentage ≤ 35 ∧ 
  ∀ (both_percentage : ℝ), 
    (both_percentage ≤ internet_percentage ∧ 
     both_percentage ≤ snack_percentage) → 
    both_percentage ≤ max_both_percentage :=
by sorry

end max_airline_services_l2042_204251


namespace problem_statement_l2042_204219

theorem problem_statement (m n : ℤ) : 
  (∃ k : ℤ, 56786730 * k = m * n * (m^60 - n^60)) ∧ 
  (m^5 + 3*m^4*n - 5*m^3*n^2 - 15*m^2*n^3 + 4*m*n^4 + 12*n^5 ≠ 33) := by
sorry

end problem_statement_l2042_204219


namespace total_trips_is_forty_l2042_204253

/-- The number of trips Jean makes -/
def jean_trips : ℕ := 23

/-- The difference between Jean's and Bill's trips -/
def trip_difference : ℕ := 6

/-- Calculates the total number of trips made by Bill and Jean -/
def total_trips : ℕ := jean_trips + (jean_trips - trip_difference)

/-- Proves that the total number of trips made by Bill and Jean is 40 -/
theorem total_trips_is_forty : total_trips = 40 := by
  sorry

end total_trips_is_forty_l2042_204253


namespace regular_price_is_18_l2042_204220

/-- The regular price of a medium pizza at Joe's pizzeria -/
def regular_price : ℝ := 18

/-- The cost of 3 medium pizzas with the promotion -/
def promotion_cost : ℝ := 15

/-- The total savings when taking full advantage of the promotion -/
def total_savings : ℝ := 39

/-- Theorem stating that the regular price of a medium pizza is $18 -/
theorem regular_price_is_18 :
  regular_price = (promotion_cost + total_savings) / 3 := by
  sorry

end regular_price_is_18_l2042_204220


namespace correct_graph_representation_l2042_204239

/-- Represents a car's travel characteristics -/
structure Car where
  speed : ℝ
  time : ℝ
  distance : ℝ

/-- The problem setup -/
def problem (m n : Car) : Prop :=
  m.speed > 0 ∧
  n.speed = 2 * m.speed ∧
  m.distance = n.distance ∧
  m.distance = m.speed * m.time ∧
  n.distance = n.speed * n.time

/-- The theorem to prove -/
theorem correct_graph_representation (m n : Car) 
  (h : problem m n) : n.speed = 2 * m.speed ∧ n.time = m.time / 2 := by
  sorry


end correct_graph_representation_l2042_204239


namespace sum_of_squares_and_products_l2042_204281

theorem sum_of_squares_and_products (a b c : ℝ) : 
  0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c → a^2 + b^2 + c^2 = 39 → a*b + b*c + c*a = 21 → a + b + c = 9 :=
by sorry

end sum_of_squares_and_products_l2042_204281


namespace first_number_proof_l2042_204203

theorem first_number_proof (y : ℝ) (h1 : y = 48) (h2 : ∃ x : ℝ, x + (1/4) * y = 27) : 
  ∃ x : ℝ, x + (1/4) * y = 27 ∧ x = 15 := by
  sorry

end first_number_proof_l2042_204203


namespace binomial_coefficient_n_1_l2042_204225

theorem binomial_coefficient_n_1 (n : ℕ+) : (n.val : ℕ).choose 1 = n.val := by sorry

end binomial_coefficient_n_1_l2042_204225


namespace x_value_when_y_is_14_l2042_204245

theorem x_value_when_y_is_14 (t : ℝ) (x y : ℝ) 
  (h1 : x = 3 - 2*t) 
  (h2 : y = 3*t + 5) 
  (h3 : y = 14) : 
  x = -3 := by
  sorry

end x_value_when_y_is_14_l2042_204245


namespace unique_five_digit_number_l2042_204267

def is_valid_increment (n m : ℕ) : Prop :=
  ∃ (d₁ d₂ d₃ d₄ d₅ : ℕ),
    n = 10000 * d₁ + 1000 * d₂ + 100 * d₃ + 10 * d₄ + d₅ ∧
    m = 10000 * (d₁ + 2) + 1000 * (d₂ + 4) + 100 * (d₃ + 2) + 10 * (d₄ + 4) + (d₅ + 4) ∧
    d₁ < 10 ∧ d₂ < 10 ∧ d₃ < 10 ∧ d₄ < 10 ∧ d₅ < 10

theorem unique_five_digit_number :
  ∀ n : ℕ,
    10000 ≤ n ∧ n < 100000 →
    (∃ m : ℕ, is_valid_increment n m ∧ m = 4 * n) →
    n = 14074 :=
by sorry

end unique_five_digit_number_l2042_204267


namespace pencil_sales_problem_l2042_204271

theorem pencil_sales_problem (eraser_price regular_price short_price : ℚ)
  (eraser_quantity short_quantity : ℕ) (total_revenue : ℚ)
  (h1 : eraser_price = 0.8)
  (h2 : regular_price = 0.5)
  (h3 : short_price = 0.4)
  (h4 : eraser_quantity = 200)
  (h5 : short_quantity = 35)
  (h6 : total_revenue = 194)
  (h7 : eraser_price * eraser_quantity + regular_price * x + short_price * short_quantity = total_revenue) :
  x = 40 := by
  sorry

#check pencil_sales_problem

end pencil_sales_problem_l2042_204271


namespace systematic_sampling_l2042_204255

theorem systematic_sampling (total_products : Nat) (num_samples : Nat) (sampled_second : Nat) : 
  total_products = 100 → 
  num_samples = 5 → 
  sampled_second = 24 → 
  ∃ (interval : Nat) (position : Nat),
    interval = total_products / num_samples ∧
    position = sampled_second % interval ∧
    (position + 3 * interval = 64) :=
by sorry

end systematic_sampling_l2042_204255


namespace salon_customers_l2042_204284

/-- The number of customers a salon has each day, given their hairspray usage and purchasing. -/
theorem salon_customers (total_cans : ℕ) (extra_cans : ℕ) (cans_per_customer : ℕ) : 
  total_cans = 33 →
  extra_cans = 5 →
  cans_per_customer = 2 →
  (total_cans - extra_cans) / cans_per_customer = 14 :=
by sorry

end salon_customers_l2042_204284


namespace m_range_proof_l2042_204201

/-- Proposition p: The equation x^2+mx+1=0 has two distinct negative roots -/
def p (m : ℝ) : Prop := ∃ x y : ℝ, x < 0 ∧ y < 0 ∧ x ≠ y ∧ x^2 + m*x + 1 = 0 ∧ y^2 + m*y + 1 = 0

/-- Proposition q: The domain of the function f(x)=log_2(4x^2+4(m-2)x+1) is ℝ -/
def q (m : ℝ) : Prop := ∀ x : ℝ, 4*x^2 + 4*(m-2)*x + 1 > 0

/-- The range of m given the conditions -/
def m_range : Set ℝ := {m : ℝ | m ≥ 3 ∨ (1 < m ∧ m ≤ 2)}

theorem m_range_proof (m : ℝ) : 
  (p m ∨ q m) ∧ ¬(p m ∧ q m) → m ∈ m_range :=
sorry

end m_range_proof_l2042_204201


namespace number_of_ways_to_buy_three_items_l2042_204248

/-- The number of headphones available -/
def num_headphones : ℕ := 9

/-- The number of computer mice available -/
def num_mice : ℕ := 13

/-- The number of keyboards available -/
def num_keyboards : ℕ := 5

/-- The number of "keyboard and mouse" sets available -/
def num_keyboard_mouse_sets : ℕ := 4

/-- The number of "headphones and mouse" sets available -/
def num_headphones_mouse_sets : ℕ := 5

/-- The theorem stating the number of ways to buy three items -/
theorem number_of_ways_to_buy_three_items : 
  num_keyboard_mouse_sets * num_headphones + 
  num_headphones_mouse_sets * num_keyboards + 
  num_headphones * num_mice * num_keyboards = 646 := by
  sorry


end number_of_ways_to_buy_three_items_l2042_204248


namespace fourth_root_ten_million_l2042_204213

theorem fourth_root_ten_million (x : ℝ) : x = 10 * (10 ^ (1/4)) → x^4 = 10000000 := by
  sorry

end fourth_root_ten_million_l2042_204213


namespace simplify_expression_l2042_204287

theorem simplify_expression : (256 : ℝ) ^ (1/4 : ℝ) * (125 : ℝ) ^ (1/3 : ℝ) = 20 := by
  sorry

end simplify_expression_l2042_204287


namespace translated_parabola_vertex_l2042_204228

/-- The vertex of a translated parabola -/
theorem translated_parabola_vertex :
  let f (x : ℝ) := -(x - 3)^2 - 2
  ∃! (h k : ℝ), (∀ x, f x = -(x - h)^2 + k) ∧ h = 3 ∧ k = -2 :=
sorry

end translated_parabola_vertex_l2042_204228


namespace prob_odd_divisor_15_factorial_l2042_204283

/-- The factorial function -/
def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

/-- The number of divisors of n -/
def numDivisors (n : ℕ) : ℕ := sorry

/-- The number of odd divisors of n -/
def numOddDivisors (n : ℕ) : ℕ := sorry

/-- The probability of a randomly chosen divisor of n being odd -/
def probOddDivisor (n : ℕ) : ℚ :=
  (numOddDivisors n : ℚ) / (numDivisors n : ℚ)

theorem prob_odd_divisor_15_factorial :
  probOddDivisor (factorial 15) = 1 / 12 := by sorry

end prob_odd_divisor_15_factorial_l2042_204283


namespace family_museum_cost_calculation_l2042_204260

/-- Calculates the discounted ticket price based on age --/
def discountedPrice (age : ℕ) (basePrice : ℚ) : ℚ :=
  if age ≥ 65 then basePrice * (1 - 0.2)
  else if age ≥ 12 ∧ age ≤ 18 then basePrice * (1 - 0.3)
  else if age ≥ 0 ∧ age ≤ 11 then basePrice * (1 - 0.5)
  else basePrice

/-- Calculates the total cost for a family museum trip --/
def familyMuseumCost (ages : List ℕ) (regularPrice specialPrice taxRate : ℚ) : ℚ :=
  let totalBeforeTax := (ages.map (fun age => discountedPrice age regularPrice + specialPrice)).sum
  totalBeforeTax * (1 + taxRate)

theorem family_museum_cost_calculation :
  let ages := [15, 10, 40, 42, 65]
  let regularPrice := 10
  let specialPrice := 5
  let taxRate := 0.1
  familyMuseumCost ages regularPrice specialPrice taxRate = 71.5 := by sorry

end family_museum_cost_calculation_l2042_204260


namespace engineer_is_smith_l2042_204257

-- Define the cities
inductive City
| Sheffield
| Leeds
| Halfway

-- Define the occupations
inductive Occupation
| Businessman
| Conductor
| Stoker
| Engineer

-- Define the people
structure Person where
  name : String
  occupation : Occupation
  city : City

-- Define the problem setup
def setup : Prop := ∃ (smith robinson jones : Person) 
  (conductor stoker engineer : Person),
  -- Businessmen
  smith.occupation = Occupation.Businessman ∧
  robinson.occupation = Occupation.Businessman ∧
  jones.occupation = Occupation.Businessman ∧
  -- Railroad workers
  conductor.occupation = Occupation.Conductor ∧
  stoker.occupation = Occupation.Stoker ∧
  engineer.occupation = Occupation.Engineer ∧
  -- Locations
  robinson.city = City.Sheffield ∧
  conductor.city = City.Sheffield ∧
  jones.city = City.Leeds ∧
  stoker.city = City.Leeds ∧
  smith.city = City.Halfway ∧
  engineer.city = City.Halfway ∧
  -- Salary relations
  ∃ (conductor_namesake : Person),
    conductor_namesake.name = conductor.name ∧
    conductor_namesake.occupation = Occupation.Businessman ∧
  -- Billiards game
  (∃ (smith_worker : Person),
    smith_worker.name = "Smith" ∧
    smith_worker.occupation ≠ Occupation.Businessman ∧
    smith_worker ≠ stoker) ∧
  -- Engineer's salary relation
  ∃ (closest_businessman : Person),
    closest_businessman.occupation = Occupation.Businessman ∧
    closest_businessman.city = City.Halfway

-- The theorem to prove
theorem engineer_is_smith (h : setup) : 
  ∃ (engineer : Person), engineer.occupation = Occupation.Engineer ∧ 
  engineer.name = "Smith" := by
  sorry

end engineer_is_smith_l2042_204257


namespace line_equation_l2042_204217

/-- Given a line passing through (-a, 0) and forming a triangle in the second quadrant with area T,
    prove that its equation is 2Tx - a²y + 2aT = 0 -/
theorem line_equation (a T : ℝ) (h1 : a ≠ 0) (h2 : T > 0) :
  ∃ (m b : ℝ), ∀ (x y : ℝ),
    (x = -a ∧ y = 0) ∨ (x < 0 ∧ y > 0) →
    (y = m * x + b ↔ 2 * T * x - a^2 * y + 2 * a * T = 0) :=
sorry

end line_equation_l2042_204217


namespace subset_partition_with_closure_l2042_204215

theorem subset_partition_with_closure (A B C : Set ℕ+) : 
  (A ∩ B = ∅ ∧ B ∩ C = ∅ ∧ C ∩ A = ∅) ∧ 
  (A ∪ B ∪ C = Set.univ) ∧
  (∀ a ∈ A, ∀ b ∈ B, ∀ c ∈ C, (a + c : ℕ+) ∈ A ∧ (b + c : ℕ+) ∈ B ∧ (a + b : ℕ+) ∈ C) →
  ((A = {n : ℕ+ | ∃ k : ℕ+, n = 3*k - 2} ∧ 
    B = {n : ℕ+ | ∃ k : ℕ+, n = 3*k - 1} ∧ 
    C = {n : ℕ+ | ∃ k : ℕ+, n = 3*k}) ∨
   (A = {n : ℕ+ | ∃ k : ℕ+, n = 3*k - 1} ∧ 
    B = {n : ℕ+ | ∃ k : ℕ+, n = 3*k - 2} ∧ 
    C = {n : ℕ+ | ∃ k : ℕ+, n = 3*k})) :=
by sorry

end subset_partition_with_closure_l2042_204215


namespace pencil_length_l2042_204208

/-- The length of the purple section of the pencil in centimeters -/
def purple_length : ℝ := 3.5

/-- The length of the black section of the pencil in centimeters -/
def black_length : ℝ := 2.8

/-- The length of the blue section of the pencil in centimeters -/
def blue_length : ℝ := 1.6

/-- The length of the green section of the pencil in centimeters -/
def green_length : ℝ := 0.9

/-- The length of the yellow section of the pencil in centimeters -/
def yellow_length : ℝ := 1.2

/-- The total length of the pencil is the sum of all colored sections -/
theorem pencil_length : 
  purple_length + black_length + blue_length + green_length + yellow_length = 10 := by
  sorry

end pencil_length_l2042_204208


namespace inequality_implication_l2042_204200

theorem inequality_implication (a b : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : a < b) :
  1 / (a * b^2) < 1 / (a^2 * b) := by
sorry

end inequality_implication_l2042_204200


namespace work_completion_time_l2042_204231

/-- Given workers a and b, where b completes a work in 7 days, and both a and b
    together complete the work in 4.117647058823529 days, prove that a can
    complete the work alone in 10 days. -/
theorem work_completion_time
  (total_work : ℝ)
  (rate_b : ℝ)
  (rate_combined : ℝ)
  (h1 : rate_b = total_work / 7)
  (h2 : rate_combined = total_work / 4.117647058823529)
  (h3 : rate_combined = rate_b + total_work / 10) :
  ∃ (days_a : ℝ), days_a = 10 ∧ total_work / days_a = total_work / 10 :=
sorry

end work_completion_time_l2042_204231


namespace binomial_15_4_l2042_204234

theorem binomial_15_4 : Nat.choose 15 4 = 1365 := by
  sorry

end binomial_15_4_l2042_204234


namespace stratified_sampling_problem_l2042_204247

/-- Represents the number of students to be drawn from a stratum -/
structure SampleSize (total : ℕ) (stratum : ℕ) (drawn : ℕ) where
  size : ℕ
  proportional : size * total = stratum * drawn

/-- The problem statement -/
theorem stratified_sampling_problem :
  let total_students : ℕ := 1400
  let male_students : ℕ := 800
  let female_students : ℕ := 600
  let male_drawn : ℕ := 40
  ∃ (female_sample : SampleSize total_students female_students male_drawn),
    female_sample.size = 30 := by
  sorry

end stratified_sampling_problem_l2042_204247


namespace copper_percentage_in_alloy_l2042_204273

/-- Given the following conditions:
    - 30 ounces of 20% alloy is used
    - 70 ounces of 27% alloy is used
    - Total amount of the desired alloy is 100 ounces
    Prove that the percentage of copper in the desired alloy is 24.9% -/
theorem copper_percentage_in_alloy : 
  let alloy_20_amount : ℝ := 30
  let alloy_27_amount : ℝ := 70
  let total_alloy : ℝ := 100
  let alloy_20_copper_percentage : ℝ := 20
  let alloy_27_copper_percentage : ℝ := 27
  let copper_amount : ℝ := (alloy_20_amount * alloy_20_copper_percentage / 100) + 
                           (alloy_27_amount * alloy_27_copper_percentage / 100)
  copper_amount / total_alloy * 100 = 24.9 := by
  sorry

end copper_percentage_in_alloy_l2042_204273


namespace train_speed_through_tunnel_l2042_204249

/-- Calculates the speed of a train passing through a tunnel -/
theorem train_speed_through_tunnel
  (train_length : ℝ)
  (tunnel_length : ℝ)
  (time_to_pass : ℝ)
  (h1 : train_length = 300)
  (h2 : tunnel_length = 1200)
  (h3 : time_to_pass = 100)
  : (train_length + tunnel_length) / time_to_pass * 3.6 = 54 := by
  sorry

#check train_speed_through_tunnel

end train_speed_through_tunnel_l2042_204249


namespace tesseract_triangles_l2042_204282

/-- The number of vertices in a tesseract -/
def tesseract_vertices : ℕ := 16

/-- The number of vertices required to form a triangle -/
def triangle_vertices : ℕ := 3

/-- The number of distinct triangles in a tesseract -/
def distinct_triangles : ℕ := Nat.choose tesseract_vertices triangle_vertices

theorem tesseract_triangles : distinct_triangles = 560 := by
  sorry

end tesseract_triangles_l2042_204282


namespace womens_doubles_handshakes_l2042_204258

/-- The number of handshakes in a women's doubles tennis tournament -/
theorem womens_doubles_handshakes : 
  let total_players : ℕ := 8
  let players_per_team : ℕ := 2
  let total_teams : ℕ := total_players / players_per_team
  let handshakes_per_player : ℕ := total_players - players_per_team
  total_players * handshakes_per_player / 2 = 24 := by
  sorry

end womens_doubles_handshakes_l2042_204258


namespace pentagon_perimeter_is_nine_l2042_204236

/-- Pentagon with given side lengths -/
structure Pentagon where
  AB : ℝ
  BC : ℝ
  CD : ℝ
  DE : ℝ
  EA : ℝ

/-- The perimeter of a pentagon -/
def perimeter (p : Pentagon) : ℝ := p.AB + p.BC + p.CD + p.DE + p.EA

/-- Theorem: The perimeter of the given pentagon is 9 -/
theorem pentagon_perimeter_is_nine :
  ∃ (p : Pentagon), p.AB = 2 ∧ p.BC = 2 ∧ p.CD = 1 ∧ p.DE = 1 ∧ p.EA = 3 ∧ perimeter p = 9 := by
  sorry

end pentagon_perimeter_is_nine_l2042_204236


namespace tetrahedron_centroid_intersection_sum_l2042_204238

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a tetrahedron defined by four points -/
structure Tetrahedron where
  P : Point3D
  A : Point3D
  B : Point3D
  C : Point3D

/-- Centroid of a tetrahedron -/
def centroid (t : Tetrahedron) : Point3D := sorry

/-- A line in 3D space defined by two points -/
structure Line3D where
  p1 : Point3D
  p2 : Point3D

/-- Distance between two points in 3D space -/
def distance (p1 p2 : Point3D) : ℝ := sorry

/-- Intersection point of a line and a face of the tetrahedron -/
def intersectionPoint (l : Line3D) (t : Tetrahedron) (face : Fin 4) : Point3D := sorry

theorem tetrahedron_centroid_intersection_sum (t : Tetrahedron) (l : Line3D) : 
  let G := centroid t
  let M := intersectionPoint l t 0
  let N := intersectionPoint l t 1
  let S := intersectionPoint l t 2
  let T := intersectionPoint l t 3
  1 / distance G M + 1 / distance G N + 1 / distance G S + 1 / distance G T = 0 := by sorry

end tetrahedron_centroid_intersection_sum_l2042_204238


namespace interest_rate_equivalence_l2042_204259

/-- Given an amount A that produces the same interest in 12 years as Rs 1000 produces in 2 years at 12%,
    prove that the interest rate R for amount A is 12%. -/
theorem interest_rate_equivalence (A : ℝ) (R : ℝ) : A > 0 →
  A * R * 12 = 1000 * 12 * 2 →
  R = 12 := by
  sorry

end interest_rate_equivalence_l2042_204259


namespace area_of_S_l2042_204265

-- Define the set S
def S : Set (ℝ × ℝ) := {(a, b) | ∀ x, x^2 + 2*b*x + 1 ≠ 2*a*(x + b)}

-- State the theorem
theorem area_of_S : MeasureTheory.volume S = π := by sorry

end area_of_S_l2042_204265


namespace pi_estimate_l2042_204205

theorem pi_estimate (n : ℕ) (m : ℕ) (h1 : n = 200) (h2 : m = 56) :
  let p := m / n
  let estimate := (4 * p + 2) / 1
  estimate = 78 / 25 := by
sorry

end pi_estimate_l2042_204205


namespace constant_term_g_l2042_204288

-- Define polynomials f, g, and h
variable (f g h : ℝ[X])

-- Define the conditions
axiom h_def : h = f * g
axiom f_constant : f.coeff 0 = 5
axiom h_constant : h.coeff 0 = -10
axiom g_quadratic : g.degree ≤ 2

-- Theorem to prove
theorem constant_term_g : g.coeff 0 = -2 := by sorry

end constant_term_g_l2042_204288


namespace geometric_sequence_common_ratio_l2042_204207

theorem geometric_sequence_common_ratio 
  (a : ℕ → ℝ) 
  (h_positive : ∀ n, a n > 0)
  (h_geometric : ∀ n, a (n + 1) = a n * q)
  (h_sum1 : a 1 + a 3 = 6)
  (h_sum2 : (a 1 + a 2 + a 3 + a 4) + a 2 = (a 1 + a 2 + a 3) + 3)
  : q = (1 : ℝ) / 2 := by
  sorry


end geometric_sequence_common_ratio_l2042_204207


namespace incorrect_observation_value_l2042_204214

theorem incorrect_observation_value 
  (n : ℕ) 
  (initial_mean correct_value new_mean : ℝ) 
  (h_n : n = 50)
  (h_initial : initial_mean = 36)
  (h_correct : correct_value = 46)
  (h_new : new_mean = 36.5) :
  ∃ (incorrect_value : ℝ),
    n * new_mean = (n - 1) * initial_mean + correct_value ∧
    incorrect_value = initial_mean * n - (n - 1) * initial_mean - correct_value ∧
    incorrect_value = 21 := by sorry

end incorrect_observation_value_l2042_204214


namespace total_baseball_cards_l2042_204230

/-- The number of people who have baseball cards -/
def num_people : ℕ := 4

/-- The number of baseball cards each person has -/
def cards_per_person : ℕ := 3

/-- The total number of baseball cards -/
def total_cards : ℕ := num_people * cards_per_person

theorem total_baseball_cards : total_cards = 12 := by
  sorry

end total_baseball_cards_l2042_204230


namespace rachel_colored_pictures_l2042_204204

/-- The number of pictures Rachel has colored -/
def pictures_colored (book1_pictures book2_pictures remaining_pictures : ℕ) : ℕ :=
  book1_pictures + book2_pictures - remaining_pictures

theorem rachel_colored_pictures :
  pictures_colored 23 32 11 = 44 := by
  sorry

end rachel_colored_pictures_l2042_204204


namespace load_transport_l2042_204297

theorem load_transport (total_load : ℝ) (box_weight_max : ℝ) (num_trucks : ℕ) (truck_capacity : ℝ) :
  total_load = 13.5 →
  box_weight_max ≤ 0.35 →
  num_trucks = 11 →
  truck_capacity = 1.5 →
  ∃ (n : ℕ), n ≤ num_trucks ∧ n * truck_capacity ≥ total_load :=
by sorry

end load_transport_l2042_204297


namespace buying_problem_equations_l2042_204211

theorem buying_problem_equations (x y : ℕ) : 
  x > 0 → y > 0 → (8 * x - y = 3 ∧ y - 7 * x = 4) → True := by
  sorry

end buying_problem_equations_l2042_204211


namespace marlington_orchestra_max_members_l2042_204291

theorem marlington_orchestra_max_members :
  ∀ n : ℕ,
  (∃ k : ℕ, 30 * n = 31 * k + 5) →
  30 * n < 1500 →
  30 * n ≤ 780 :=
by sorry

end marlington_orchestra_max_members_l2042_204291


namespace complex_expression_simplification_l2042_204266

theorem complex_expression_simplification :
  (0.7264 * 0.4329 * 0.5478) + (0.1235 * 0.3412 * 0.6214) - 
  (0.1289 * 0.5634 * 0.3921) / (0.3785 * 0.4979 * 0.2884) - 
  (0.2956 * 0.3412 * 0.6573) = -0.3902 := by
  sorry

end complex_expression_simplification_l2042_204266


namespace small_circle_radius_l2042_204240

/-- Given a large circle with radius 10 meters containing three smaller circles
    that touch each other and are aligned horizontally across its center,
    prove that the radius of each smaller circle is 10/3 meters. -/
theorem small_circle_radius (R : ℝ) (r : ℝ) : R = 10 →
  3 * (2 * r) = 2 * R →
  r = 10 / 3 := by sorry

end small_circle_radius_l2042_204240


namespace square_of_1031_l2042_204280

theorem square_of_1031 : (1031 : ℕ)^2 = 1062961 := by
  sorry

end square_of_1031_l2042_204280


namespace sum_absolute_value_l2042_204276

theorem sum_absolute_value (x₁ x₂ x₃ x₄ x₅ : ℝ) 
  (h : x₁ + 1 = x₂ + 2 ∧ x₂ + 2 = x₃ + 3 ∧ x₃ + 3 = x₄ + 4 ∧ x₄ + 4 = x₅ + 5 ∧ 
       x₅ + 5 = x₁ + x₂ + x₃ + x₄ + x₅ + 6) : 
  |x₁ + x₂ + x₃ + x₄ + x₅| = 3.75 := by
sorry

end sum_absolute_value_l2042_204276


namespace intersection_points_form_equilateral_triangle_l2042_204212

/-- The common points of the circle x^2 + (y - 1)^2 = 1 and the ellipse 9x^2 + (y + 1)^2 = 9 form an equilateral triangle -/
theorem intersection_points_form_equilateral_triangle :
  ∀ (A B C : ℝ × ℝ),
  (A ∈ {p : ℝ × ℝ | p.1^2 + (p.2 - 1)^2 = 1} ∩ {p : ℝ × ℝ | 9*p.1^2 + (p.2 + 1)^2 = 9}) →
  (B ∈ {p : ℝ × ℝ | p.1^2 + (p.2 - 1)^2 = 1} ∩ {p : ℝ × ℝ | 9*p.1^2 + (p.2 + 1)^2 = 9}) →
  (C ∈ {p : ℝ × ℝ | p.1^2 + (p.2 - 1)^2 = 1} ∩ {p : ℝ × ℝ | 9*p.1^2 + (p.2 + 1)^2 = 9}) →
  A ≠ B → B ≠ C → A ≠ C →
  dist A B = dist B C ∧ dist B C = dist C A :=
by sorry


end intersection_points_form_equilateral_triangle_l2042_204212


namespace correct_minus_position_l2042_204218

def numbers : List ℕ := [6, 9, 12, 15, 18, 21]

def place_signs (nums : List ℕ) (minus_pos : ℕ) : ℤ :=
  (nums.take minus_pos).sum - nums[minus_pos]! + (nums.drop (minus_pos + 1)).sum

theorem correct_minus_position (nums : List ℕ) (h : nums = numbers) :
  ∃! pos : ℕ, pos < nums.length - 1 ∧ place_signs nums pos = 45 :=
by sorry

end correct_minus_position_l2042_204218


namespace intersection_implies_a_value_l2042_204298

def A (a : ℝ) : Set ℝ := {2, 4, a^3 - 2*a^2 - a + 7}

def B (a : ℝ) : Set ℝ := {-4, a + 3, a^2 - 2*a + 2, a^3 + a^2 + 3*a + 7}

theorem intersection_implies_a_value (a : ℝ) :
  A a ∩ B a = {2, 5} → a = 2 := by
  sorry

end intersection_implies_a_value_l2042_204298


namespace existence_of_non_divisible_a_l2042_204242

theorem existence_of_non_divisible_a (p : ℕ) (hp : p.Prime) (hp_ge_5 : p ≥ 5) :
  ∃ a : ℕ, 1 ≤ a ∧ a ≤ p - 2 ∧
    ¬(p^2 ∣ a^(p-1) - 1) ∧
    ¬(p^2 ∣ (a+1)^(p-1) - 1) := by
  sorry

end existence_of_non_divisible_a_l2042_204242


namespace sum_of_reciprocals_l2042_204216

theorem sum_of_reciprocals (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) 
  (h : x + y = 5 * x * y) : 
  1 / x + 1 / y = 5 := by
sorry

end sum_of_reciprocals_l2042_204216


namespace triangle_internal_region_l2042_204261

-- Define the three lines that form the triangle
def line1 (x y : ℝ) : Prop := x + 2*y = 2
def line2 (x y : ℝ) : Prop := 2*x + y = 2
def line3 (x y : ℝ) : Prop := x - y = 3

-- Define the internal region of the triangle
def internal_region (x y : ℝ) : Prop :=
  x - y < 3 ∧ x + 2*y < 2 ∧ 2*x + y > 2

-- Theorem statement
theorem triangle_internal_region :
  ∀ x y : ℝ, 
    (∃ ε > 0, line1 (x + ε) y ∨ line2 (x + ε) y ∨ line3 (x + ε) y) →
    (∃ ε > 0, line1 (x - ε) y ∨ line2 (x - ε) y ∨ line3 (x - ε) y) →
    (∃ ε > 0, line1 x (y + ε) ∨ line2 x (y + ε) ∨ line3 x (y + ε)) →
    (∃ ε > 0, line1 x (y - ε) ∨ line2 x (y - ε) ∨ line3 x (y - ε)) →
    internal_region x y :=
sorry

end triangle_internal_region_l2042_204261


namespace jamie_ball_collection_l2042_204299

def total_balls (initial_red : ℕ) (blue_multiplier : ℕ) (lost_red : ℕ) (bought_yellow : ℕ) : ℕ :=
  (initial_red - lost_red) + (initial_red * blue_multiplier) + bought_yellow

theorem jamie_ball_collection : total_balls 16 2 6 32 = 74 := by
  sorry

end jamie_ball_collection_l2042_204299


namespace send_more_money_solution_l2042_204278

def is_valid_assignment (S E N D M O R Y : Nat) : Prop :=
  S ≠ 0 ∧ M ≠ 0 ∧
  S < 10 ∧ E < 10 ∧ N < 10 ∧ D < 10 ∧ M < 10 ∧ O < 10 ∧ R < 10 ∧ Y < 10 ∧
  S ≠ E ∧ S ≠ N ∧ S ≠ D ∧ S ≠ M ∧ S ≠ O ∧ S ≠ R ∧ S ≠ Y ∧
  E ≠ N ∧ E ≠ D ∧ E ≠ M ∧ E ≠ O ∧ E ≠ R ∧ E ≠ Y ∧
  N ≠ D ∧ N ≠ M ∧ N ≠ O ∧ N ≠ R ∧ N ≠ Y ∧
  D ≠ M ∧ D ≠ O ∧ D ≠ R ∧ D ≠ Y ∧
  M ≠ O ∧ M ≠ R ∧ M ≠ Y ∧
  O ≠ R ∧ O ≠ Y ∧
  R ≠ Y

theorem send_more_money_solution :
  ∃ (S E N D M O R Y : Nat),
    is_valid_assignment S E N D M O R Y ∧
    1000 * S + 100 * E + 10 * N + D + 1000 * M + 100 * O + 10 * R + E =
    10000 * M + 1000 * O + 100 * N + 10 * E + Y :=
by sorry

end send_more_money_solution_l2042_204278


namespace pie_chart_probability_l2042_204286

theorem pie_chart_probability (prob_D prob_E prob_F : ℚ) : 
  prob_D = 1/4 → prob_E = 1/3 → prob_D + prob_E + prob_F = 1 → prob_F = 5/12 := by
  sorry

end pie_chart_probability_l2042_204286


namespace root_in_interval_l2042_204256

open Real

-- Define the function f
def f (x : ℝ) : ℝ := 3 * x + x - 5

-- State the theorem
theorem root_in_interval (a b : ℕ+) (x₀ : ℝ) :
  b - a = 1 →
  ∃ x₀, x₀ ∈ Set.Icc a b ∧ f x₀ = 0 →
  a + b = 3 :=
by sorry

end root_in_interval_l2042_204256


namespace inequality_solution_l2042_204250

theorem inequality_solution (x : ℝ) : 
  1 / (x * (x + 1)) - 1 / ((x + 1) * (x + 2)) > 1 / 4 ↔ 
  x < -2 ∨ (0 < x ∧ x < 2) :=
sorry

end inequality_solution_l2042_204250


namespace diff_color_probability_is_three_fourths_l2042_204254

/-- The number of color choices for socks -/
def sock_colors : ℕ := 3

/-- The number of color choices for shorts -/
def short_colors : ℕ := 4

/-- The total number of possible combinations -/
def total_combinations : ℕ := sock_colors * short_colors

/-- The number of combinations where socks and shorts have the same color -/
def same_color_combinations : ℕ := 3

/-- The probability of selecting different colors for socks and shorts -/
def diff_color_probability : ℚ := (total_combinations - same_color_combinations : ℚ) / total_combinations

theorem diff_color_probability_is_three_fourths :
  diff_color_probability = 3 / 4 := by sorry

end diff_color_probability_is_three_fourths_l2042_204254


namespace merchant_articles_l2042_204293

/-- The number of articles a merchant has, given profit percentage and price relationship -/
theorem merchant_articles (N : ℕ) (profit_percentage : ℚ) : 
  profit_percentage = 25 / 400 →
  (N : ℚ) * (1 : ℚ) = 16 * (1 + profit_percentage) →
  N = 17 := by
  sorry

#check merchant_articles

end merchant_articles_l2042_204293


namespace computer_table_price_l2042_204246

/-- The selling price of an item given its cost price and markup percentage -/
def selling_price (cost : ℚ) (markup : ℚ) : ℚ :=
  cost * (1 + markup / 100)

/-- Theorem: The selling price of a computer table with cost price 6925 and 24% markup is 8587 -/
theorem computer_table_price : selling_price 6925 24 = 8587 := by
  sorry

end computer_table_price_l2042_204246


namespace sum_of_repeating_decimals_l2042_204223

/-- Represents a repeating decimal with a single repeating digit -/
def single_repeat (n : ℕ) : ℚ := n / 9

/-- Represents a repeating decimal with two repeating digits -/
def double_repeat (a b : ℕ) : ℚ := (10 * a + b) / 99

/-- The sum of 0.777... and 0.131313... is equal to 10/11 -/
theorem sum_of_repeating_decimals : 
  single_repeat 7 + double_repeat 1 3 = 10 / 11 := by sorry

end sum_of_repeating_decimals_l2042_204223


namespace choir_group_size_l2042_204226

theorem choir_group_size (total : ℕ) (group2 : ℕ) (group3 : ℕ) (h1 : total = 70) (h2 : group2 = 30) (h3 : group3 = 15) :
  total - group2 - group3 = 25 := by
  sorry

end choir_group_size_l2042_204226


namespace student_admission_price_l2042_204268

theorem student_admission_price
  (total_tickets : ℕ)
  (adult_price : ℕ)
  (total_amount : ℕ)
  (student_attendees : ℕ)
  (h1 : total_tickets = 1500)
  (h2 : adult_price = 12)
  (h3 : total_amount = 16200)
  (h4 : student_attendees = 300) :
  (total_amount - (total_tickets - student_attendees) * adult_price) / student_attendees = 6 :=
by sorry

end student_admission_price_l2042_204268


namespace odometer_puzzle_l2042_204292

theorem odometer_puzzle (a b c : ℕ) : 
  (a ≥ 1) →
  (a + b + c ≤ 9) →
  (100 * b + 10 * a + c - (100 * a + 10 * b + c)) % 60 = 0 →
  a^2 + b^2 + c^2 = 35 := by
sorry

end odometer_puzzle_l2042_204292


namespace inverse_inequality_l2042_204252

theorem inverse_inequality (a b : ℝ) (h1 : a > 0) (h2 : 0 > b) : 1 / a > 1 / b := by
  sorry

end inverse_inequality_l2042_204252


namespace medical_team_combinations_l2042_204243

theorem medical_team_combinations (n_male : Nat) (n_female : Nat) 
  (h1 : n_male = 6) (h2 : n_female = 5) : 
  (n_male.choose 2) * (n_female.choose 1) = 75 := by
  sorry

end medical_team_combinations_l2042_204243


namespace vector_subtraction_and_scalar_multiplication_l2042_204210

/-- Given vectors a and b in ℝ³, prove that a - 5b equals the expected result. -/
theorem vector_subtraction_and_scalar_multiplication (a b : ℝ × ℝ × ℝ) :
  a = (-5, 3, 2) → b = (2, -1, 4) → a - 5 • b = (-15, 8, -18) := by
  sorry

end vector_subtraction_and_scalar_multiplication_l2042_204210


namespace smallest_next_divisor_l2042_204264

theorem smallest_next_divisor (m : ℕ) : 
  m % 2 = 0 ∧ 1000 ≤ m ∧ m < 10000 ∧ m % 437 = 0 → 
  ∃ (d : ℕ), d > 437 ∧ m % d = 0 ∧ d ≥ 874 ∧ 
  ∀ (d' : ℕ), d' > 437 ∧ m % d' = 0 → d' ≥ 874 :=
by sorry

end smallest_next_divisor_l2042_204264


namespace average_rope_length_l2042_204235

theorem average_rope_length (piece1 piece2 : ℝ) (h1 : piece1 = 2) (h2 : piece2 = 6) :
  (piece1 + piece2) / 2 = 4 := by
  sorry

end average_rope_length_l2042_204235


namespace sum_floor_equals_126_l2042_204229

theorem sum_floor_equals_126 
  (a b c d : ℝ) 
  (pos_a : 0 < a) (pos_b : 0 < b) (pos_c : 0 < c) (pos_d : 0 < d)
  (sum_squares : a^2 + b^2 = 2008 ∧ c^2 + d^2 = 2008)
  (products : a*c = 1000 ∧ b*d = 1000) : 
  ⌊a + b + c + d⌋ = 126 := by
  sorry

end sum_floor_equals_126_l2042_204229


namespace traffic_light_probability_theorem_l2042_204221

/-- Represents the duration of each color in the traffic light cycle -/
structure TrafficLightCycle where
  green : ℕ
  yellow : ℕ
  red : ℕ

/-- Calculates the total cycle duration -/
def cycleDuration (cycle : TrafficLightCycle) : ℕ :=
  cycle.green + cycle.yellow + cycle.red

/-- Calculates the number of seconds where a color change can be observed -/
def changeObservationWindow (cycle : TrafficLightCycle) : ℕ :=
  3 * 5 -- 5 seconds before each color change

/-- Calculates the probability of observing a color change -/
def probabilityOfColorChange (cycle : TrafficLightCycle) (observationInterval : ℕ) : ℚ :=
  (changeObservationWindow cycle : ℚ) / (cycleDuration cycle : ℚ)

theorem traffic_light_probability_theorem (cycle : TrafficLightCycle) 
    (h1 : cycle.green = 50)
    (h2 : cycle.yellow = 5)
    (h3 : cycle.red = 40)
    (h4 : observationInterval = 5) :
    probabilityOfColorChange cycle observationInterval = 3 / 19 := by
  sorry

end traffic_light_probability_theorem_l2042_204221


namespace cubic_equation_roots_l2042_204274

theorem cubic_equation_roots (a b : ℝ) : 
  (∀ x : ℝ, 2*x^3 + a*x^2 - 13*x + b = 0 ↔ x = 2 ∨ x = -3 ∨ (∃ r : ℝ, x = r ∧ 2*(2-r)*(3+r) = 0)) →
  a = 1 ∧ b = 6 := by
sorry

end cubic_equation_roots_l2042_204274


namespace power_sum_of_i_l2042_204222

-- Define the complex number i
def i : ℂ := Complex.I

-- State the theorem
theorem power_sum_of_i : i^22 + i^222 = -2 := by sorry

end power_sum_of_i_l2042_204222


namespace finite_primes_imply_equal_bases_l2042_204237

def divides_set (a b c d : ℕ+) : Set ℕ :=
  {p : ℕ | ∃ n : ℕ, n > 0 ∧ p.Prime ∧ p ∣ (a * b^n + c * d^n)}

theorem finite_primes_imply_equal_bases (a b c d : ℕ+) :
  (Set.Finite (divides_set a b c d)) → b = d := by
  sorry

end finite_primes_imply_equal_bases_l2042_204237


namespace not_even_implies_exists_unequal_l2042_204285

-- Define a real-valued function on ℝ
variable (f : ℝ → ℝ)

-- Define what it means for f to be not even
def NotEven (f : ℝ → ℝ) : Prop :=
  ¬(∀ x : ℝ, f (-x) = f x)

-- Theorem statement
theorem not_even_implies_exists_unequal (f : ℝ → ℝ) :
  NotEven f → ∃ x₀ : ℝ, f (-x₀) ≠ f x₀ :=
by
  sorry

end not_even_implies_exists_unequal_l2042_204285


namespace complement_of_hit_at_least_once_l2042_204209

-- Define the sample space
def Ω : Type := Bool × Bool

-- Define the event of hitting the target at least once
def hit_at_least_once (ω : Ω) : Prop :=
  ω.1 ∨ ω.2

-- Define the event of missing the target both times
def miss_both_times (ω : Ω) : Prop :=
  ¬ω.1 ∧ ¬ω.2

-- Theorem stating that missing both times is the complement of hitting at least once
theorem complement_of_hit_at_least_once :
  ∀ ω : Ω, miss_both_times ω ↔ ¬(hit_at_least_once ω) :=
sorry

end complement_of_hit_at_least_once_l2042_204209


namespace max_value_of_function_l2042_204275

theorem max_value_of_function (x y z : ℝ) (h : x^2 + y^2 + z^2 ≠ 0) :
  (x*y + 2*y*z) / (x^2 + y^2 + z^2) ≤ Real.sqrt 5 / 2 := by
  sorry

end max_value_of_function_l2042_204275


namespace arithmetic_sequence_properties_l2042_204295

/-- An arithmetic sequence with specified properties -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  S : ℕ → ℝ  -- The sum sequence
  is_arithmetic : ∀ n : ℕ, a (n + 1) - a n = a (n + 2) - a (n + 1)
  sum_property : ∀ n : ℕ, S n = (n : ℝ) * (a 1 + a n) / 2
  a4_eq_10 : a 4 = 10
  S6_eq_S3_plus_39 : S 6 = S 3 + 39

/-- The theorem stating the properties of the arithmetic sequence -/
theorem arithmetic_sequence_properties (seq : ArithmeticSequence) :
  seq.a 1 = 1 ∧ ∀ n : ℕ, seq.a n = 3 * n - 2 := by
  sorry

end arithmetic_sequence_properties_l2042_204295


namespace smallest_d_correct_l2042_204277

/-- The smallest possible value of d satisfying the triangle and square perimeter conditions -/
def smallest_d : ℕ :=
  let d : ℕ := 675
  d

theorem smallest_d_correct :
  let d := smallest_d
  -- The perimeter of the equilateral triangle exceeds the perimeter of the square by 2023 cm
  ∀ s : ℝ, 3 * (s + d) - 4 * s = 2023 →
  -- The square has a perimeter greater than 0 cm
  (s > 0) →
  -- d is a multiple of 3
  (d % 3 = 0) →
  -- d is the smallest value satisfying these conditions
  ∀ d' : ℕ, d' < d →
    (∀ s : ℝ, 3 * (s + d') - 4 * s = 2023 → s > 0 → d' % 3 = 0 → False) :=
by sorry

#eval smallest_d

end smallest_d_correct_l2042_204277


namespace constant_ratio_sum_theorem_l2042_204263

theorem constant_ratio_sum_theorem (x₁ x₂ x₃ x₄ : ℝ) 
  (h_not_all_equal : ¬(x₁ = x₂ ∧ x₂ = x₃ ∧ x₃ = x₄))
  (h_constant_ratio : ∃ k : ℝ, 
    (x₁ + x₂) / (x₃ + x₄) = k ∧
    (x₁ + x₃) / (x₂ + x₄) = k ∧
    (x₁ + x₄) / (x₂ + x₃) = k) :
  (∃ k : ℝ, k = -1 ∧ 
    (x₁ + x₂) / (x₃ + x₄) = k ∧
    (x₁ + x₃) / (x₂ + x₄) = k ∧
    (x₁ + x₄) / (x₂ + x₃) = k) ∧
  x₁ + x₂ + x₃ + x₄ = 0 :=
by sorry

end constant_ratio_sum_theorem_l2042_204263


namespace circle_passes_through_points_l2042_204290

def point_A : ℝ × ℝ := (-2, 1)
def point_B : ℝ × ℝ := (9, 3)
def point_C : ℝ × ℝ := (1, 7)

def circle_equation (x y : ℝ) : Prop :=
  (x - 7/2)^2 + (y - 2)^2 = 125/4

theorem circle_passes_through_points :
  circle_equation point_A.1 point_A.2 ∧
  circle_equation point_B.1 point_B.2 ∧
  circle_equation point_C.1 point_C.2 :=
by sorry

end circle_passes_through_points_l2042_204290
