import Mathlib

namespace hike_vans_count_l4083_408361

/-- Calculates the number of vans required for a hike --/
def calculate_vans (total_people : ℕ) (cars : ℕ) (taxis : ℕ) 
  (people_per_car : ℕ) (people_per_taxi : ℕ) (people_per_van : ℕ) : ℕ :=
  let people_in_cars_and_taxis := cars * people_per_car + taxis * people_per_taxi
  let people_in_vans := total_people - people_in_cars_and_taxis
  (people_in_vans + people_per_van - 1) / people_per_van

theorem hike_vans_count : 
  calculate_vans 58 3 6 4 6 5 = 2 := by
  sorry

end hike_vans_count_l4083_408361


namespace sin_120_degrees_l4083_408363

theorem sin_120_degrees : Real.sin (120 * π / 180) = Real.sqrt 3 / 2 := by
  sorry

end sin_120_degrees_l4083_408363


namespace similar_triangles_l4083_408378

/-- Given five complex numbers representing points in a plane, if three triangles formed by these points are directly similar, then a fourth triangle is also directly similar to them. -/
theorem similar_triangles (a b c u v : ℂ) 
  (h : (v - a) / (u - a) = (u - v) / (b - v) ∧ (u - v) / (b - v) = (c - u) / (v - u)) : 
  (v - a) / (u - a) = (c - a) / (b - a) := by
  sorry

end similar_triangles_l4083_408378


namespace sum_of_coefficients_l4083_408307

/-- The polynomial P(x) = 3(x^8 - 2x^5 + 4x^3 - 7) - 5(2x^4 - 3x^2 + 8) + 6(x^6 - 3) -/
def P (x : ℝ) : ℝ := 3 * (x^8 - 2*x^5 + 4*x^3 - 7) - 5 * (2*x^4 - 3*x^2 + 8) + 6 * (x^6 - 3)

/-- The sum of the coefficients of P(x) is equal to P(1) -/
theorem sum_of_coefficients : P 1 = -59 := by
  sorry

end sum_of_coefficients_l4083_408307


namespace second_concert_attendance_l4083_408355

theorem second_concert_attendance 
  (first_concert : ℕ) 
  (additional_attendees : ℕ) 
  (h1 : first_concert = 65899) 
  (h2 : additional_attendees = 119) : 
  first_concert + additional_attendees = 66018 := by
  sorry

end second_concert_attendance_l4083_408355


namespace cards_given_away_ben_cards_given_away_l4083_408395

theorem cards_given_away (basketball_boxes : Nat) (basketball_cards_per_box : Nat)
                         (baseball_boxes : Nat) (baseball_cards_per_box : Nat)
                         (cards_left : Nat) : Nat :=
  let total_cards := basketball_boxes * basketball_cards_per_box + 
                     baseball_boxes * baseball_cards_per_box
  total_cards - cards_left

theorem ben_cards_given_away : 
  cards_given_away 4 10 5 8 22 = 58 := by
  sorry

end cards_given_away_ben_cards_given_away_l4083_408395


namespace apples_handed_out_is_19_l4083_408330

/-- Calculates the number of apples handed out to students in a cafeteria. -/
def apples_handed_out (initial_apples : ℕ) (num_pies : ℕ) (apples_per_pie : ℕ) : ℕ :=
  initial_apples - (num_pies * apples_per_pie)

/-- Proves that the number of apples handed out to students is 19. -/
theorem apples_handed_out_is_19 :
  apples_handed_out 75 7 8 = 19 := by
  sorry

#eval apples_handed_out 75 7 8

end apples_handed_out_is_19_l4083_408330


namespace children_count_l4083_408303

def number_of_children (crayons_per_child : ℕ) (total_crayons : ℕ) : ℕ :=
  total_crayons / crayons_per_child

theorem children_count : number_of_children 6 72 = 12 := by
  sorry

end children_count_l4083_408303


namespace apples_bought_l4083_408319

theorem apples_bought (initial : ℕ) (used : ℕ) (final : ℕ) : 
  initial = 17 → used = 2 → final = 38 → final - (initial - used) = 23 := by
  sorry

end apples_bought_l4083_408319


namespace fib_divisibility_l4083_408384

-- Define the Fibonacci sequence
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | n + 2 => fib (n + 1) + fib n

-- State the theorem
theorem fib_divisibility (m n : ℕ) (h : m > 0) (h' : n > 0) : 
  m ∣ n → (fib (m - 1)) ∣ (fib (n - 1)) := by
  sorry

end fib_divisibility_l4083_408384


namespace circle_center_distance_l4083_408321

theorem circle_center_distance (x y : ℝ) : 
  (x^2 + y^2 = 6*x + 8*y + 9) → 
  Real.sqrt ((11 - x)^2 + (5 - y)^2) = Real.sqrt 65 := by
sorry

end circle_center_distance_l4083_408321


namespace binomial_8_4_l4083_408352

theorem binomial_8_4 : (8 : ℕ).choose 4 = 70 := by sorry

end binomial_8_4_l4083_408352


namespace expression_value_l4083_408383

theorem expression_value :
  let x : ℤ := 3
  let y : ℤ := 2
  let z : ℤ := 4
  let w : ℤ := -1
  x^2 * y - 2 * x * y + 3 * x * z - (x + y) * (y + z) * (z + w) = -48 :=
by sorry

end expression_value_l4083_408383


namespace smallest_n_for_perfect_square_sums_l4083_408322

def isPerfectSquare (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

def validPermutation (p : List ℕ) : Prop :=
  ∀ i : ℕ, i < p.length - 1 → isPerfectSquare (p[i]! + p[i+1]!)

def consecutiveIntegers (n : ℕ) : List ℕ := List.range n

theorem smallest_n_for_perfect_square_sums : 
  ∀ n : ℕ, n > 1 →
    (∃ p : List ℕ, p.length = n ∧ p.toFinset = (consecutiveIntegers n).toFinset ∧ validPermutation p) 
    ↔ n ≥ 15 :=
sorry

end smallest_n_for_perfect_square_sums_l4083_408322


namespace angle_relations_l4083_408305

theorem angle_relations (α β : Real) (h1 : 0 < α ∧ α < π/2) (h2 : 0 < β ∧ β < π/2)
  (h3 : Real.tan (α/2) = 1/3) (h4 : Real.cos (α - β) = -4/5) :
  Real.sin α = 3/5 ∧ 2*α + β = π := by
  sorry

end angle_relations_l4083_408305


namespace total_whales_is_178_l4083_408374

/-- Represents the number of whales observed during Ishmael's monitoring trips -/
def total_whales (first_trip_male : ℕ) : ℕ :=
  let first_trip_female := 2 * first_trip_male
  let first_trip_total := first_trip_male + first_trip_female
  let second_trip_baby := 8
  let second_trip_parents := 2 * second_trip_baby
  let second_trip_total := second_trip_baby + second_trip_parents
  let third_trip_male := first_trip_male / 2
  let third_trip_female := first_trip_female
  let third_trip_total := third_trip_male + third_trip_female
  first_trip_total + second_trip_total + third_trip_total

/-- Theorem stating that the total number of whales observed is 178 -/
theorem total_whales_is_178 : total_whales 28 = 178 := by
  sorry

end total_whales_is_178_l4083_408374


namespace carls_gift_bags_l4083_408357

/-- Represents the gift bag distribution problem at Carl's open house. -/
theorem carls_gift_bags (total_visitors : ℕ) (extravagant_bags : ℕ) (additional_bags : ℕ) :
  total_visitors = 90 →
  extravagant_bags = 10 →
  additional_bags = 60 →
  total_visitors - (extravagant_bags + additional_bags) = 30 := by
  sorry

#check carls_gift_bags

end carls_gift_bags_l4083_408357


namespace trumpet_cost_l4083_408351

/-- The cost of Mike's trumpet, given the net amount spent and the amount received for selling a song book. -/
theorem trumpet_cost (net_spent : ℝ) (song_book_sold : ℝ) (h1 : net_spent = 139.32) (h2 : song_book_sold = 5.84) :
  net_spent + song_book_sold = 145.16 := by
  sorry

end trumpet_cost_l4083_408351


namespace airplane_purchase_exceeds_budget_l4083_408302

/-- Proves that the total cost of purchasing the airplane exceeds $5.00 USD -/
theorem airplane_purchase_exceeds_budget : 
  let initial_budget : ℝ := 5.00
  let airplane_cost_eur : ℝ := 3.80
  let exchange_rate : ℝ := 0.82
  let sales_tax_rate : ℝ := 0.075
  let credit_card_surcharge_rate : ℝ := 0.035
  let processing_fee_usd : ℝ := 0.25
  
  let airplane_cost_usd : ℝ := airplane_cost_eur / exchange_rate
  let sales_tax : ℝ := airplane_cost_usd * sales_tax_rate
  let credit_card_surcharge : ℝ := airplane_cost_usd * credit_card_surcharge_rate
  let total_cost : ℝ := airplane_cost_usd + sales_tax + credit_card_surcharge + processing_fee_usd
  
  total_cost > initial_budget := by
  sorry

#check airplane_purchase_exceeds_budget

end airplane_purchase_exceeds_budget_l4083_408302


namespace rotation_composition_is_translation_l4083_408399

-- Define a plane figure
def PlaneFigure : Type := sorry

-- Define a point in the plane
def Point : Type := sorry

-- Define a rotation transformation
def rotate (center : Point) (angle : ℝ) (figure : PlaneFigure) : PlaneFigure := sorry

-- Define a translation transformation
def translate (displacement : Point) (figure : PlaneFigure) : PlaneFigure := sorry

-- Define composition of transformations
def compose (t1 t2 : PlaneFigure → PlaneFigure) : PlaneFigure → PlaneFigure := sorry

theorem rotation_composition_is_translation 
  (F : PlaneFigure) (O O₁ : Point) (α : ℝ) :
  ∃ d : Point, compose (rotate O α) (rotate O₁ (-α)) F = translate d F :=
sorry

end rotation_composition_is_translation_l4083_408399


namespace smallest_satisfying_number_l4083_408342

/-- Given a two-digit number n, returns the number obtained by switching its digits -/
def switch_digits (n : ℕ) : ℕ :=
  (n % 10) * 10 + (n / 10)

/-- Checks if a number satisfies the condition: switching its digits and multiplying by 3 results in 3n -/
def satisfies_condition (n : ℕ) : Prop :=
  3 * switch_digits n = 3 * n

theorem smallest_satisfying_number :
  ∃ (n : ℕ),
    10 ≤ n ∧ n < 100 ∧
    satisfies_condition n ∧
    (∀ m : ℕ, 10 ≤ m ∧ m < n → ¬satisfies_condition m) ∧
    n = 11 := by
  sorry

end smallest_satisfying_number_l4083_408342


namespace package_cost_proof_l4083_408328

/-- The cost of a 12-roll package of paper towels -/
def package_cost : ℝ := 9

/-- The cost of one roll sold individually -/
def individual_roll_cost : ℝ := 1

/-- The number of rolls in a package -/
def rolls_per_package : ℕ := 12

/-- The percent of savings per roll for the package -/
def savings_percent : ℝ := 25

theorem package_cost_proof : 
  package_cost = rolls_per_package * (individual_roll_cost * (1 - savings_percent / 100)) :=
by sorry

end package_cost_proof_l4083_408328


namespace tv_sales_effect_l4083_408332

theorem tv_sales_effect (price_reduction : ℝ) (sales_increase : ℝ) : 
  price_reduction = 0.22 → sales_increase = 0.86 → 
  (1 - price_reduction) * (1 + sales_increase) - 1 = 0.4518 := by
  sorry

end tv_sales_effect_l4083_408332


namespace divisible_by_eleven_l4083_408381

/-- A seven-digit number in the form 8n46325 where n is a single digit -/
def sevenDigitNumber (n : ℕ) : ℕ := 8000000 + 1000000*n + 46325

/-- Predicate to check if a natural number is a single digit -/
def isSingleDigit (n : ℕ) : Prop := n < 10

theorem divisible_by_eleven (n : ℕ) : 
  isSingleDigit n → (sevenDigitNumber n) % 11 = 0 → n = 1 := by
  sorry

end divisible_by_eleven_l4083_408381


namespace hallway_width_proof_l4083_408379

/-- Proves that the width of a hallway is 4 feet given the specified conditions -/
theorem hallway_width_proof (total_area : Real) (central_length : Real) (central_width : Real) (hallway_length : Real) :
  total_area = 124 ∧ 
  central_length = 10 ∧ 
  central_width = 10 ∧ 
  hallway_length = 6 → 
  (total_area - central_length * central_width) / hallway_length = 4 := by
sorry

end hallway_width_proof_l4083_408379


namespace cos_sixty_degrees_l4083_408358

theorem cos_sixty_degrees : Real.cos (π / 3) = 1 / 2 := by
  sorry

end cos_sixty_degrees_l4083_408358


namespace pension_calculation_l4083_408392

-- Define the pension function
noncomputable def pension (k : ℝ) (x : ℝ) : ℝ := k * Real.sqrt x

-- Define the problem parameters
variable (c d r s : ℝ)

-- State the theorem
theorem pension_calculation (h1 : d ≠ c) 
                            (h2 : pension k x - pension k (x - c) = r) 
                            (h3 : pension k x - pension k (x - d) = s) : 
  pension k x = (r^2 - s^2) / (2 * (r - s)) := by
  sorry

end pension_calculation_l4083_408392


namespace boy_scouts_permission_slips_l4083_408337

theorem boy_scouts_permission_slips 
  (total_with_slips : ℝ) 
  (boy_scouts_percentage : ℝ) 
  (girl_scouts_with_slips : ℝ) :
  total_with_slips = 0.60 →
  boy_scouts_percentage = 0.45 →
  girl_scouts_with_slips = 0.6818 →
  (boy_scouts_percentage * (total_with_slips - (1 - boy_scouts_percentage) * girl_scouts_with_slips)) / 
  (boy_scouts_percentage * (1 - (1 - boy_scouts_percentage) * girl_scouts_with_slips)) = 0.50 :=
by sorry

end boy_scouts_permission_slips_l4083_408337


namespace final_alcohol_percentage_l4083_408306

/-- Calculates the final alcohol percentage of a solution after adding more alcohol and water. -/
theorem final_alcohol_percentage
  (initial_volume : ℝ)
  (initial_alcohol_percentage : ℝ)
  (added_alcohol : ℝ)
  (added_water : ℝ)
  (h1 : initial_volume = 40)
  (h2 : initial_alcohol_percentage = 0.05)
  (h3 : added_alcohol = 3.5)
  (h4 : added_water = 6.5) :
  let initial_alcohol := initial_volume * initial_alcohol_percentage
  let final_alcohol := initial_alcohol + added_alcohol
  let final_volume := initial_volume + added_alcohol + added_water
  let final_alcohol_percentage := final_alcohol / final_volume
  final_alcohol_percentage = 0.11 := by
  sorry

end final_alcohol_percentage_l4083_408306


namespace impossible_arrangement_l4083_408334

/-- Represents a domino tile with two numbers -/
structure Domino where
  first : Nat
  second : Nat
  first_range : first ≤ 6
  second_range : second ≤ 6

/-- The set of all 28 standard domino tiles -/
def StandardDominoes : Finset Domino := sorry

/-- Counts the number of even numbers on all tiles -/
def CountEvenNumbers (tiles : Finset Domino) : Nat := sorry

/-- Counts the number of odd numbers on all tiles -/
def CountOddNumbers (tiles : Finset Domino) : Nat := sorry

/-- Defines a valid arrangement of domino tiles -/
def ValidArrangement (arrangement : List Domino) : Prop := sorry

theorem impossible_arrangement :
  CountEvenNumbers StandardDominoes = 32 →
  CountOddNumbers StandardDominoes = 24 →
  ¬∃ (arrangement : List Domino), 
    (arrangement.length = StandardDominoes.card) ∧ 
    (ValidArrangement arrangement) := by
  sorry

end impossible_arrangement_l4083_408334


namespace frank_breakfast_shopping_cost_l4083_408394

/-- Calculates the total cost of Frank's breakfast shopping --/
def breakfast_shopping_cost (bun_price : ℚ) (bun_quantity : ℕ) (milk_price : ℚ) (milk_quantity : ℕ) (egg_price_multiplier : ℕ) : ℚ :=
  let bun_cost := bun_price * bun_quantity
  let milk_cost := milk_price * milk_quantity
  let egg_cost := milk_price * egg_price_multiplier
  bun_cost + milk_cost + egg_cost

/-- Theorem: The total cost of Frank's breakfast shopping is $11.00 --/
theorem frank_breakfast_shopping_cost :
  breakfast_shopping_cost 0.1 10 2 2 3 = 11 := by
  sorry

end frank_breakfast_shopping_cost_l4083_408394


namespace inequality_proof_l4083_408345

theorem inequality_proof (x y z : ℝ) 
  (h_pos_x : x > 0) (h_pos_y : y > 0) (h_pos_z : z > 0)
  (h_xyz : x * y * z ≥ 1) :
  (x^4 + y) * (y^4 + z) * (z^4 + x) ≥ (x + y^2) * (y + z^2) * (z + x^2) :=
sorry

end inequality_proof_l4083_408345


namespace integer_points_in_triangle_DEF_l4083_408354

/-- Represents a right triangle with integer side lengths -/
structure RightTriangle where
  leg1 : ℕ
  leg2 : ℕ

/-- Counts the number of integer coordinate points in and on a right triangle -/
def count_integer_points (t : RightTriangle) : ℕ :=
  sorry

/-- The specific right triangle with legs 15 and 20 -/
def triangle_DEF : RightTriangle :=
  { leg1 := 15, leg2 := 20 }

/-- Theorem stating that the number of integer coordinate points in triangle_DEF is 181 -/
theorem integer_points_in_triangle_DEF : 
  count_integer_points triangle_DEF = 181 := by
  sorry

end integer_points_in_triangle_DEF_l4083_408354


namespace m_range_proof_l4083_408375

-- Define the propositions p and q
def p (m : ℝ) : Prop := ∃ x : ℝ, x^2 - 2*x + m = 0

def q (m : ℝ) : Prop := m ∈ Set.Icc (-1 : ℝ) 5

-- Define the range of m
def m_range : Set ℝ := Set.Ioi (-1 : ℝ) ∪ Set.Ioc 1 5

-- Theorem statement
theorem m_range_proof :
  (∀ m : ℝ, (p m ∧ q m → False) ∧ (p m ∨ q m)) →
  (∀ m : ℝ, m ∈ m_range ↔ (p m ∨ q m) ∧ ¬(p m ∧ q m)) :=
by sorry

end m_range_proof_l4083_408375


namespace fiftieth_term_of_sequence_l4083_408377

def arithmetic_sequence (a₁ : ℕ) (d : ℕ) (n : ℕ) : ℕ :=
  a₁ + (n - 1) * d

theorem fiftieth_term_of_sequence : arithmetic_sequence 2 4 50 = 198 := by
  sorry

end fiftieth_term_of_sequence_l4083_408377


namespace polynomial_simplification_l4083_408324

theorem polynomial_simplification (p : ℝ) :
  (2 * p^4 + 5 * p^3 - 3 * p + 4) + (-p^4 + 2 * p^3 - 7 * p^2 + 4 * p - 2) =
  p^4 + 7 * p^3 - 7 * p^2 + p + 2 := by
  sorry

end polynomial_simplification_l4083_408324


namespace square_of_binomial_formula_l4083_408326

theorem square_of_binomial_formula (a b : ℝ) :
  (a - b) * (b + a) = a^2 - b^2 ∧
  (4*a + b) * (4*a - 2*b) ≠ (2*a + b)^2 - (2*a - b)^2 ∧
  (a - 2*b) * (2*b - a) ≠ (a + b)^2 - (a - b)^2 ∧
  (2*a - b) * (-2*a + b) ≠ (a + b)^2 - (a - b)^2 :=
by sorry

#check square_of_binomial_formula

end square_of_binomial_formula_l4083_408326


namespace factorization_p1_factorization_p2_l4083_408380

-- Define the polynomials
def p1 (x : ℝ) : ℝ := (x^2 - 2*x - 1) * (x^2 - 2*x + 3) + 4
def p2 (x : ℝ) : ℝ := (x^2 + 6*x) * (x^2 + 6*x + 18) + 81

-- State the theorems
theorem factorization_p1 : ∀ x : ℝ, p1 x = (x - 1)^4 := by sorry

theorem factorization_p2 : ∀ x : ℝ, p2 x = (x + 3)^4 := by sorry

end factorization_p1_factorization_p2_l4083_408380


namespace winnie_balloons_distribution_l4083_408360

/-- Represents the number of balloons Winnie has left after distribution -/
def balloonsLeft (red white green chartreuse friends : ℕ) : ℕ :=
  (red + white + green + chartreuse) % friends

/-- Proves that Winnie has no balloons left after distribution -/
theorem winnie_balloons_distribution 
  (red : ℕ) (white : ℕ) (green : ℕ) (chartreuse : ℕ) (friends : ℕ)
  (h_red : red = 24)
  (h_white : white = 36)
  (h_green : green = 70)
  (h_chartreuse : chartreuse = 90)
  (h_friends : friends = 10) :
  balloonsLeft red white green chartreuse friends = 0 := by
  sorry

#eval balloonsLeft 24 36 70 90 10

end winnie_balloons_distribution_l4083_408360


namespace distance_AB_is_336_l4083_408311

/-- The distance between two points A and B, given the conditions of the problem. -/
def distance_AB : ℝ :=
  let t_total := 3.5  -- Total time in hours
  let t_car3 := 3     -- Time for Car 3 to reach A
  let d_car1_left := 84  -- Distance left for Car 1 at 10:30 AM
  let d_car2_fraction := 3/8  -- Fraction of total distance Car 2 has traveled when Car 1 and 3 meet
  336

/-- The theorem stating that the distance between A and B is 336 km. -/
theorem distance_AB_is_336 :
  let d := distance_AB
  let v1 := d / 3.5 - 24  -- Speed of Car 1
  let v2 := d / 3.5       -- Speed of Car 2
  let v3 := d / 6         -- Speed of Car 3
  (v1 + v3 = 8/3 * v2) ∧  -- Condition when Car 1 and 3 meet
  (v3 * 3 = d / 2) ∧      -- Car 3 reaches A at 10:00 AM
  (v2 * 3.5 = d) ∧        -- Car 2 reaches A at 10:30 AM
  (d - v1 * 3.5 = 84) →   -- Car 1 is 84 km from B at 10:30 AM
  d = 336 := by
  sorry


end distance_AB_is_336_l4083_408311


namespace arithmetic_sequence_problem_l4083_408382

theorem arithmetic_sequence_problem (a : ℕ → ℝ) (d : ℝ) :
  (∀ n, a (n + 1) = a n + d) →  -- arithmetic sequence condition
  a 1 + a 4 + a 10 + a 16 + a 19 = 150 →
  a 20 - a 26 + a 16 = 30 := by
sorry

end arithmetic_sequence_problem_l4083_408382


namespace no_real_solutions_existence_of_zero_product_ellipses_same_foci_l4083_408388

-- Statement 1
theorem no_real_solutions : ∀ x : ℝ, x^2 - 3*x + 3 ≠ 0 := by sorry

-- Statement 2
theorem existence_of_zero_product : ∃ x y : ℝ, x * y = 0 ∧ x ≠ 0 ∧ y ≠ 0 := by sorry

-- Statement 3
def ellipse1 (x y : ℝ) : Prop := x^2/25 + y^2/9 = 1

def ellipse2 (k x y : ℝ) : Prop := x^2/(25-k) + y^2/(9-k) = 1

def has_same_foci (k : ℝ) : Prop :=
  ∃ a b : ℝ, (∀ x y : ℝ, ellipse1 x y ↔ (x-a)^2/25 + (x+a)^2/25 + y^2/9 = 1) ∧
             (∀ x y : ℝ, ellipse2 k x y ↔ (x-b)^2/(25-k) + (x+b)^2/(25-k) + y^2/(9-k) = 1) ∧
             a = b

theorem ellipses_same_foci : ∀ k : ℝ, 9 < k → k < 25 → has_same_foci k := by sorry

end no_real_solutions_existence_of_zero_product_ellipses_same_foci_l4083_408388


namespace line_through_points_with_xintercept_l4083_408366

/-- A line in a 2D plane -/
structure Line where
  slope : ℚ
  yIntercept : ℚ

/-- Create a line from two points -/
def Line.fromPoints (x1 y1 x2 y2 : ℚ) : Line :=
  let slope := (y2 - y1) / (x2 - x1)
  let yIntercept := y1 - slope * x1
  { slope := slope, yIntercept := yIntercept }

/-- Get the x-coordinate for a given y-coordinate on a line -/
def Line.xCoordinate (l : Line) (y : ℚ) : ℚ :=
  (y - l.yIntercept) / l.slope

theorem line_through_points_with_xintercept
  (line : Line)
  (h1 : line = Line.fromPoints 4 0 10 3)
  (h2 : line.xCoordinate (-6) = -8) : True :=
by sorry

end line_through_points_with_xintercept_l4083_408366


namespace sphere_surface_area_l4083_408362

theorem sphere_surface_area (r : ℝ) (h : r = 4) : 4 * Real.pi * r^2 = 64 * Real.pi := by
  sorry

end sphere_surface_area_l4083_408362


namespace inequality_not_always_true_l4083_408349

theorem inequality_not_always_true (a b c : ℝ) (h1 : c < b) (h2 : b < a) (h3 : a * c < 0) :
  ¬ (∀ a b c, c < b ∧ b < a ∧ a * c < 0 → c * b < a * b) := by
sorry

end inequality_not_always_true_l4083_408349


namespace total_theme_parks_eq_395_l4083_408301

/-- The number of theme parks in four towns -/
def total_theme_parks (jamestown venice marina_del_ray newport_beach : ℕ) : ℕ :=
  jamestown + venice + marina_del_ray + newport_beach

/-- Theorem: The total number of theme parks in four towns is 395 -/
theorem total_theme_parks_eq_395 :
  ∃ (jamestown venice marina_del_ray newport_beach : ℕ),
    jamestown = 35 ∧
    venice = jamestown + 40 ∧
    marina_del_ray = jamestown + 60 ∧
    newport_beach = 2 * marina_del_ray ∧
    total_theme_parks jamestown venice marina_del_ray newport_beach = 395 :=
by sorry

end total_theme_parks_eq_395_l4083_408301


namespace temperature_is_dependent_l4083_408320

/-- Represents the variables in the solar water heater scenario -/
inductive SolarHeaterVariable
  | IntensityOfSunlight
  | TemperatureOfWater
  | DurationOfExposure
  | CapacityOfHeater

/-- Represents the relationship between two variables -/
structure Relationship where
  independent : SolarHeaterVariable
  dependent : SolarHeaterVariable

/-- Defines the relationship in the solar water heater scenario -/
def solarHeaterRelationship : Relationship :=
  { independent := SolarHeaterVariable.DurationOfExposure,
    dependent := SolarHeaterVariable.TemperatureOfWater }

/-- Theorem: The temperature of water is the dependent variable in the solar water heater scenario -/
theorem temperature_is_dependent :
  solarHeaterRelationship.dependent = SolarHeaterVariable.TemperatureOfWater :=
by sorry

end temperature_is_dependent_l4083_408320


namespace smallest_perimeter_isosceles_triangle_l4083_408356

/-- Triangle with positive integer side lengths --/
structure Triangle :=
  (side1 : ℕ+) (side2 : ℕ+) (side3 : ℕ+)

/-- Isosceles triangle with two equal sides --/
def IsoscelesTriangle (t : Triangle) : Prop :=
  t.side1 = t.side2

/-- Point J is the intersection of angle bisectors of ∠Q and ∠R --/
def HasIntersectionJ (t : Triangle) : Prop :=
  ∃ j : ℝ × ℝ, true  -- We don't need to specify the exact conditions for J

/-- Length of QJ is 10 --/
def QJLength (t : Triangle) : Prop :=
  ∃ qj : ℝ, qj = 10

/-- Perimeter of a triangle --/
def Perimeter (t : Triangle) : ℕ :=
  t.side1.val + t.side2.val + t.side3.val

/-- The main theorem --/
theorem smallest_perimeter_isosceles_triangle :
  ∀ t : Triangle,
    IsoscelesTriangle t →
    HasIntersectionJ t →
    QJLength t →
    (∀ t' : Triangle,
      IsoscelesTriangle t' →
      HasIntersectionJ t' →
      QJLength t' →
      Perimeter t ≤ Perimeter t') →
    Perimeter t = 120 :=
sorry

end smallest_perimeter_isosceles_triangle_l4083_408356


namespace f_increasing_on_interval_l4083_408343

noncomputable def f (x : ℝ) : ℝ := -2/3 * x^3 + 3/2 * x^2 - x

theorem f_increasing_on_interval :
  ∀ x ∈ Set.Icc (1/2 : ℝ) 1, 
    (∀ y ∈ Set.Icc (1/2 : ℝ) 1, x ≤ y → f x ≤ f y) :=
by
  sorry

end f_increasing_on_interval_l4083_408343


namespace z_value_l4083_408359

theorem z_value (x y z : ℚ) 
  (eq1 : 3 * x^2 + 2 * x * y * z - y^3 + 11 = z)
  (eq2 : x = 2)
  (eq3 : y = 3) : 
  z = 4 / 11 := by
  sorry

end z_value_l4083_408359


namespace homework_problem_distribution_l4083_408371

theorem homework_problem_distribution (total : ℕ) (true_false : ℕ) : 
  total = 45 → true_false = 6 → ∃ (multiple_choice free_response : ℕ),
    multiple_choice = 2 * free_response ∧
    free_response > true_false ∧
    multiple_choice + free_response + true_false = total ∧
    free_response - true_false = 7 :=
by sorry

end homework_problem_distribution_l4083_408371


namespace square_perimeter_relation_l4083_408373

-- Define the perimeter of square A
def perimeterA : ℝ := 36

-- Define the relationship between areas of square A and B
def areaRelation (areaA areaB : ℝ) : Prop := areaB = areaA / 3

-- State the theorem
theorem square_perimeter_relation (sideA sideB : ℝ) 
  (h1 : sideA * 4 = perimeterA)
  (h2 : areaRelation (sideA * sideA) (sideB * sideB)) :
  4 * sideB = 12 * Real.sqrt 3 := by
  sorry

end square_perimeter_relation_l4083_408373


namespace businessmen_drink_neither_l4083_408391

theorem businessmen_drink_neither (total : ℕ) (coffee : ℕ) (tea : ℕ) (both : ℕ)
  (h_total : total = 30)
  (h_coffee : coffee = 15)
  (h_tea : tea = 13)
  (h_both : both = 7) :
  total - ((coffee + tea) - both) = 9 :=
by sorry

end businessmen_drink_neither_l4083_408391


namespace hyperbola_eccentricity_l4083_408323

/-- Given a hyperbola E with equation x²/a² - y²/b² = 1 (where a > 0 and b > 0),
    if one of its asymptotes has a slope of 30°, then its eccentricity is 2√3/3. -/
theorem hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (b / a = Real.tan (π / 6)) →
  Real.sqrt (1 + (b / a)^2) = 2 * Real.sqrt 3 / 3 := by
  sorry

end hyperbola_eccentricity_l4083_408323


namespace sculpture_cost_theorem_l4083_408389

/-- Calculates the total cost of John's custom sculpture --/
def calculate_sculpture_cost (base_price : ℝ) (standard_discount : ℝ) (marble_increase : ℝ) 
  (glass_increase : ℝ) (shipping_cost : ℝ) (tax_rate : ℝ) : ℝ :=
  let discounted_price := base_price * (1 - standard_discount)
  let marble_price := discounted_price * (1 + marble_increase)
  let glass_price := marble_price * (1 + glass_increase)
  let pre_tax_price := glass_price
  let tax := pre_tax_price * tax_rate
  pre_tax_price + tax + shipping_cost

/-- The total cost of John's sculpture is $1058.18 --/
theorem sculpture_cost_theorem : 
  calculate_sculpture_cost 450 0.15 0.70 0.35 75 0.12 = 1058.18 := by
  sorry

end sculpture_cost_theorem_l4083_408389


namespace problem_solution_l4083_408336

theorem problem_solution (P Q : ℚ) : 
  (4 / 7 : ℚ) = P / 63 ∧ (4 / 7 : ℚ) = 98 / (Q - 14) → P + Q = 221.5 := by
  sorry

end problem_solution_l4083_408336


namespace waiter_customers_l4083_408376

/-- The total number of customers a waiter has after new arrivals -/
def total_customers (initial : ℕ) (new_arrivals : ℕ) : ℕ :=
  initial + new_arrivals

/-- Theorem stating that with 3 initial customers and 5 new arrivals, the total is 8 -/
theorem waiter_customers : total_customers 3 5 = 8 := by
  sorry

end waiter_customers_l4083_408376


namespace regular_polygon_sides_l4083_408325

theorem regular_polygon_sides (exterior_angle : ℝ) : 
  exterior_angle = 30 → (360 / exterior_angle : ℝ) = 12 :=
by sorry

end regular_polygon_sides_l4083_408325


namespace inequality_xyz_l4083_408368

theorem inequality_xyz (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  x^4 + y^4 + z^2 ≥ x*y*z*Real.sqrt 8 := by
  sorry

end inequality_xyz_l4083_408368


namespace binomial_10_2_l4083_408353

theorem binomial_10_2 : (Nat.choose 10 2) = 45 := by
  sorry

end binomial_10_2_l4083_408353


namespace problem_solution_l4083_408393

-- Define the set M
def M : Set ℝ := {x | x^2 - x - 2 < 0}

-- Define the set N
def N (a b : ℝ) : Set ℝ := {x | a < x ∧ x < b}

theorem problem_solution :
  -- Part 1: M = (-1, 2)
  M = Set.Ioo (-1) 2 ∧
  -- Part 2: If M ⊇ N, then the minimum value of a is -1
  (∀ a b : ℝ, M ⊇ N a b → a ≥ -1) ∧
  (∃ a₀ : ℝ, a₀ = -1 ∧ ∃ b : ℝ, M ⊇ N a₀ b) ∧
  -- Part 3: If M ∩ N = M, then b ∈ [2, +∞)
  (∀ a b : ℝ, M ∩ N a b = M → b ≥ 2) :=
by sorry

end problem_solution_l4083_408393


namespace min_value_trigonometric_expression_l4083_408317

theorem min_value_trigonometric_expression (α β : ℝ) :
  (3 * Real.cos α + 4 * Real.sin β - 7)^2 + (3 * Real.sin α + 4 * Real.cos β - 12)^2 ≥ 48 :=
by sorry

end min_value_trigonometric_expression_l4083_408317


namespace sophomore_freshman_difference_l4083_408314

/-- Represents the number of students in each grade -/
structure GradeDistribution where
  freshman : ℕ
  sophomore : ℕ
  junior : ℕ

/-- Represents the sample size for each grade -/
structure SampleDistribution where
  freshman : ℕ
  sophomore : ℕ
  junior : ℕ

/-- Calculates the stratified sample distribution based on the grade distribution and total sample size -/
def stratifiedSample (grades : GradeDistribution) (totalSample : ℕ) : SampleDistribution :=
  let total := grades.freshman + grades.sophomore + grades.junior
  let freshmanSample := (grades.freshman * totalSample) / total
  let sophomoreSample := (grades.sophomore * totalSample) / total
  let juniorSample := totalSample - freshmanSample - sophomoreSample
  { freshman := freshmanSample
  , sophomore := sophomoreSample
  , junior := juniorSample }

/-- The main theorem to be proved -/
theorem sophomore_freshman_difference
  (grades : GradeDistribution)
  (h1 : grades.freshman = 1000)
  (h2 : grades.sophomore = 1050)
  (h3 : grades.junior = 1200)
  (totalSample : ℕ)
  (h4 : totalSample = 65) :
  let sample := stratifiedSample grades totalSample
  sample.sophomore = sample.freshman + 1 := by
  sorry

end sophomore_freshman_difference_l4083_408314


namespace intersection_implies_a_value_a_geq_2_sufficient_not_necessary_l4083_408300

-- Define the sets A and B
def A : Set ℝ := {x | -1 < x ∧ x < 1}
def B (a : ℝ) : Set ℝ := {x | -1-a ≤ x ∧ x ≤ 1-a}

-- Theorem 1: If A ∩ B = {x | 1/2 ≤ x < 1}, then a = -3/2
theorem intersection_implies_a_value (a : ℝ) : 
  A ∩ B a = {x | 1/2 ≤ x ∧ x < 1} → a = -3/2 := by sorry

-- Theorem 2: a ≥ 2 is a sufficient but not necessary condition for A ∩ B = ∅
theorem a_geq_2_sufficient_not_necessary (a : ℝ) :
  (a ≥ 2 → A ∩ B a = ∅) ∧ ¬(A ∩ B a = ∅ → a ≥ 2) := by sorry

end intersection_implies_a_value_a_geq_2_sufficient_not_necessary_l4083_408300


namespace even_function_implies_a_equals_one_l4083_408348

def f (a x : ℝ) : ℝ := (x + 1) * (x - a)

theorem even_function_implies_a_equals_one (a : ℝ) :
  (∀ x, f a x = f a (-x)) → a = 1 := by
  sorry

end even_function_implies_a_equals_one_l4083_408348


namespace repeating_decimal_as_fraction_l4083_408346

/-- The decimal representation of 0.6̄03 as a rational number -/
def repeating_decimal : ℚ := 0.6 + (3 : ℚ) / 100 / (1 - 1/100)

/-- Theorem stating that 0.6̄03 is equal to 104/165 -/
theorem repeating_decimal_as_fraction : repeating_decimal = 104 / 165 := by
  sorry

end repeating_decimal_as_fraction_l4083_408346


namespace linear_equation_condition_l4083_408318

theorem linear_equation_condition (a : ℝ) : 
  (|a| - 1 = 1 ∧ a - 2 ≠ 0) ↔ a = -2 := by sorry

end linear_equation_condition_l4083_408318


namespace sum_of_squares_impossible_l4083_408308

theorem sum_of_squares_impossible (n : ℤ) :
  (n % 4 = 3 → ¬∃ (a b : ℤ), n = a^2 + b^2) ∧
  (n % 8 = 7 → ¬∃ (a b c : ℤ), n = a^2 + b^2 + c^2) :=
by sorry

end sum_of_squares_impossible_l4083_408308


namespace total_toys_count_l4083_408331

/-- The number of toys Mandy has -/
def mandy_toys : ℕ := 20

/-- The number of toys Anna has -/
def anna_toys : ℕ := 3 * mandy_toys

/-- The number of toys Amanda has -/
def amanda_toys : ℕ := anna_toys + 2

/-- The total number of toys -/
def total_toys : ℕ := mandy_toys + anna_toys + amanda_toys

theorem total_toys_count : total_toys = 142 := by sorry

end total_toys_count_l4083_408331


namespace last_three_positions_l4083_408327

/-- Represents the position of a person in the line after a certain number of rounds -/
def position (round : ℕ) : ℕ :=
  match round with
  | 0 => 3
  | n + 1 =>
    let prev := position n
    if prev % 2 = 1 then (3 * prev - 1) / 2 else (3 * prev - 2) / 2

/-- The theorem stating the initial positions of the last three people remaining -/
theorem last_three_positions (initial_count : ℕ) (h : initial_count = 2009) :
  ∃ (rounds : ℕ), position rounds = 1600 ∧ 
    (∀ k, k > rounds → position k < 1600) ∧
    (∀ n, n ≤ initial_count → n ≠ 1 → n ≠ 2 → n ≠ 1600 → 
      ∃ m, m ≤ rounds ∧ (3 * (position m)) % n = 0) :=
sorry

end last_three_positions_l4083_408327


namespace roots_of_cubic_polynomials_l4083_408341

theorem roots_of_cubic_polynomials (a b : ℝ) (r s : ℝ) :
  (∃ t, r + s + t = 0 ∧ r * s + r * t + s * t = a) →
  (∃ t', r + 4 + s - 3 + t' = 0 ∧ (r + 4) * (s - 3) + (r + 4) * t' + (s - 3) * t' = a) →
  b = -330 ∨ b = 90 :=
by sorry

end roots_of_cubic_polynomials_l4083_408341


namespace sum_of_roots_l4083_408309

theorem sum_of_roots (a b : ℝ) 
  (ha : a^3 - 3*a^2 - 4*a + 12 = 0)
  (hb : 3*b^3 + 9*b^2 - 11*b - 3 = 0) : 
  a + b = -2 := by
sorry

end sum_of_roots_l4083_408309


namespace water_park_admission_charge_l4083_408350

/-- Calculates the total admission charge for an adult and accompanying children in a water park. -/
def totalAdmissionCharge (adultCharge childCharge : ℚ) (numChildren : ℕ) : ℚ :=
  adultCharge + childCharge * numChildren

/-- Proves that the total admission charge for an adult and 3 children is $3.25 -/
theorem water_park_admission_charge :
  let adultCharge : ℚ := 1
  let childCharge : ℚ := 3/4
  let numChildren : ℕ := 3
  totalAdmissionCharge adultCharge childCharge numChildren = 13/4 := by
sorry

#eval totalAdmissionCharge 1 (3/4) 3

end water_park_admission_charge_l4083_408350


namespace more_stable_performance_l4083_408315

/-- Represents a student's performance in throwing solid balls -/
structure StudentPerformance where
  average_score : ℝ
  variance : ℝ

/-- Determines if a student's performance is more stable than another's -/
def more_stable (a b : StudentPerformance) : Prop :=
  a.variance < b.variance

/-- Theorem: Given two students with equal average scores, the one with smaller variance has more stable performance -/
theorem more_stable_performance (student_A student_B : StudentPerformance)
  (h_equal_average : student_A.average_score = student_B.average_score)
  (h_A_variance : student_A.variance = 0.1)
  (h_B_variance : student_B.variance = 0.02) :
  more_stable student_B student_A :=
by sorry

end more_stable_performance_l4083_408315


namespace min_y_value_l4083_408339

theorem min_y_value (x y : ℝ) (h : x^2 + y^2 = 16*x + 36*y) :
  ∃ (y_min : ℝ), y_min = 18 - Real.sqrt 388 ∧ ∀ (y' : ℝ), ∃ (x' : ℝ), x'^2 + y'^2 = 16*x' + 36*y' → y' ≥ y_min :=
by sorry

end min_y_value_l4083_408339


namespace line_equation_equivalence_slope_intercept_parameters_l4083_408365

/-- Given a line equation in vector form, prove its slope-intercept form and parameters -/
theorem line_equation_equivalence :
  ∀ (x y : ℝ),
  (2 : ℝ) * (x - 4) + (-1 : ℝ) * (y + 3) = 0 ↔ y = 2 * x - 11 :=
by sorry

/-- Prove the slope and y-intercept of the line -/
theorem slope_intercept_parameters :
  ∃ (m b : ℝ), (∀ (x y : ℝ), (2 : ℝ) * (x - 4) + (-1 : ℝ) * (y + 3) = 0 ↔ y = m * x + b) ∧ m = 2 ∧ b = -11 :=
by sorry

end line_equation_equivalence_slope_intercept_parameters_l4083_408365


namespace quadratic_inequality_solution_l4083_408397

-- Define the quadratic function
def f (c : ℝ) (x : ℝ) : ℝ := x^2 - 8*x + c

-- State the theorem
theorem quadratic_inequality_solution (c : ℝ) :
  (c > 0) → (∃ x, f c x < 0) ↔ (c > 0 ∧ c < 16) :=
sorry

end quadratic_inequality_solution_l4083_408397


namespace five_letter_words_count_l4083_408367

def word_count : ℕ := 26

theorem five_letter_words_count :
  (Finset.sum (Finset.range 4) (λ k => Nat.choose 5 k)) = word_count := by
  sorry

end five_letter_words_count_l4083_408367


namespace min_value_problem_l4083_408369

theorem min_value_problem (a b c d e f g h : ℝ) 
  (h1 : a * b * c * d = 16) 
  (h2 : e * f * g * h = 1) : 
  (a * f)^2 + (b * e)^2 + (c * h)^2 + (d * g)^2 ≥ 16 := by
  sorry

end min_value_problem_l4083_408369


namespace item_price_ratio_l4083_408347

theorem item_price_ratio (x y c : ℝ) (hx : x = 0.8 * c) (hy : y = 1.25 * c) :
  y / x = 25 / 16 := by
  sorry

end item_price_ratio_l4083_408347


namespace least_period_is_30_l4083_408310

/-- A function satisfying the given condition -/
def SatisfyingFunction (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (x + 5) + f (x - 5) = f x

/-- The period of a function -/
def IsPeriod (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x : ℝ, f (x + p) = f x

/-- The least positive period of a function -/
def IsLeastPeriod (f : ℝ → ℝ) (p : ℝ) : Prop :=
  p > 0 ∧ IsPeriod f p ∧ ∀ q, 0 < q ∧ q < p → ¬IsPeriod f q

/-- The main theorem -/
theorem least_period_is_30 :
  ∀ f : ℝ → ℝ, SatisfyingFunction f → IsLeastPeriod f 30 :=
sorry

end least_period_is_30_l4083_408310


namespace mixed_fraction_division_l4083_408390

theorem mixed_fraction_division :
  (7 + 1/3) / (2 + 1/2) = 44/15 := by sorry

end mixed_fraction_division_l4083_408390


namespace existence_of_sequence_with_divisors_l4083_408338

theorem existence_of_sequence_with_divisors :
  ∃ f : ℕ → ℕ, ∀ k : ℕ, ∃ d : Finset ℕ,
    d.card ≥ k ∧
    ∀ x ∈ d, x > 0 ∧ (f k)^2 + f k + 2023 ≡ 0 [MOD x] :=
by sorry

end existence_of_sequence_with_divisors_l4083_408338


namespace least_possible_average_speed_l4083_408398

/-- Represents a palindromic number -/
def isPalindrome (n : ℕ) : Prop := sorry

/-- The drive duration in hours -/
def driveDuration : ℕ := 5

/-- The speed limit in miles per hour -/
def speedLimit : ℕ := 65

/-- The initial odometer reading -/
def initialReading : ℕ := 123321

/-- Theorem: The least possible average speed is 20 miles per hour -/
theorem least_possible_average_speed :
  ∃ (finalReading : ℕ),
    isPalindrome initialReading ∧
    isPalindrome finalReading ∧
    finalReading > initialReading ∧
    finalReading - initialReading ≤ driveDuration * speedLimit ∧
    (finalReading - initialReading) / driveDuration = 20 ∧
    ∀ (otherFinalReading : ℕ),
      isPalindrome otherFinalReading →
      otherFinalReading > initialReading →
      otherFinalReading - initialReading ≤ driveDuration * speedLimit →
      (otherFinalReading - initialReading) / driveDuration ≥ 20 :=
sorry

end least_possible_average_speed_l4083_408398


namespace total_peanuts_l4083_408340

def jose_peanuts : ℕ := 85
def kenya_peanuts : ℕ := jose_peanuts + 48
def malachi_peanuts : ℕ := kenya_peanuts + 35

theorem total_peanuts : jose_peanuts + kenya_peanuts + malachi_peanuts = 386 := by
  sorry

end total_peanuts_l4083_408340


namespace gcd_lcm_product_30_45_l4083_408344

theorem gcd_lcm_product_30_45 : Nat.gcd 30 45 * Nat.lcm 30 45 = 1350 := by
  sorry

end gcd_lcm_product_30_45_l4083_408344


namespace wall_volume_calculation_l4083_408316

/-- Proves that the volume of a wall is 345 cubic meters given specific brick dimensions and quantity --/
theorem wall_volume_calculation (brick_length : ℝ) (brick_width : ℝ) (brick_height : ℝ) 
  (brick_count : ℕ) (h1 : brick_length = 20) (h2 : brick_width = 10) (h3 : brick_height = 7.5) 
  (h4 : brick_count = 23000) : 
  (brick_length * brick_width * brick_height * brick_count) / 1000000 = 345 := by
  sorry

end wall_volume_calculation_l4083_408316


namespace complex_number_in_first_quadrant_l4083_408396

def i : ℂ := Complex.I

theorem complex_number_in_first_quadrant :
  let z : ℂ := i * (1 - i) * i
  (z.re > 0) ∧ (z.im > 0) :=
by sorry

end complex_number_in_first_quadrant_l4083_408396


namespace specific_pyramid_volume_l4083_408313

/-- Represents a pyramid with a square base and specified face areas -/
structure Pyramid where
  base_area : ℝ
  face_area1 : ℝ
  face_area2 : ℝ

/-- Calculates the volume of a pyramid given its properties -/
noncomputable def pyramid_volume (p : Pyramid) : ℝ :=
  let base_side := Real.sqrt p.base_area
  let height1 := 2 * p.face_area1 / base_side
  let height2 := 2 * p.face_area2 / base_side
  let a := (height1^2 - height2^2 + base_side^2) / (2 * base_side)
  let h := Real.sqrt (height1^2 - (base_side - a)^2)
  (1/3) * p.base_area * h

/-- The theorem stating the volume of the specific pyramid -/
theorem specific_pyramid_volume :
  let p := Pyramid.mk 256 128 112
  ∃ ε > 0, |pyramid_volume p - 1230.83| < ε :=
sorry

end specific_pyramid_volume_l4083_408313


namespace root_difference_implies_k_value_l4083_408385

theorem root_difference_implies_k_value (k : ℝ) :
  (∀ x y : ℝ, x^2 + k*x + 10 = 0 ∧ y^2 - k*y + 10 = 0 ∧ y = x + 3) →
  k = 3 := by
sorry

end root_difference_implies_k_value_l4083_408385


namespace race_length_race_length_is_165_l4083_408304

theorem race_length : ℝ → Prop :=
  fun x =>
    ∀ (speed_a speed_b speed_c : ℝ),
      speed_a > 0 ∧ speed_b > 0 ∧ speed_c > 0 →
      x > 35 →
      speed_b * x = speed_a * (x - 15) →
      speed_c * x = speed_a * (x - 35) →
      speed_c * (x - 15) = speed_b * (x - 22) →
      x = 165

theorem race_length_is_165 : race_length 165 := by
  sorry

end race_length_race_length_is_165_l4083_408304


namespace p_plus_q_equals_27_over_2_l4083_408329

theorem p_plus_q_equals_27_over_2 (p q : ℝ) 
  (hp : p^3 - 18*p^2 + 27*p - 135 = 0)
  (hq : 12*q^3 - 90*q^2 - 450*q + 4950 = 0) :
  p + q = 27/2 := by
  sorry

end p_plus_q_equals_27_over_2_l4083_408329


namespace equation_solution_l4083_408364

theorem equation_solution : ∃! x : ℝ, (2 / (x + 3) + 3 * x / (x + 3) - 4 / (x + 3) = 4) ∧ x = -14 := by
  sorry

end equation_solution_l4083_408364


namespace rhombus_area_l4083_408386

/-- The area of a rhombus with side length 2 and an angle of 45 degrees between adjacent sides is 2√2 -/
theorem rhombus_area (s : ℝ) (θ : ℝ) (h1 : s = 2) (h2 : θ = π / 4) :
  s * s * Real.sin θ = 2 * Real.sqrt 2 := by
  sorry

end rhombus_area_l4083_408386


namespace set_intersection_example_l4083_408372

theorem set_intersection_example : 
  let A : Set ℕ := {1, 2, 3}
  let B : Set ℕ := {2, 3, 4}
  A ∩ B = {2, 3} := by
sorry

end set_intersection_example_l4083_408372


namespace ball_picking_problem_l4083_408312

/-- Represents a bag with black and white balls -/
structure BallBag where
  total_balls : ℕ
  white_balls : ℕ
  black_balls : ℕ
  h_total : total_balls = white_balls + black_balls

/-- The probability of picking a white ball -/
def prob_white (bag : BallBag) : ℚ :=
  bag.white_balls / bag.total_balls

theorem ball_picking_problem (bag : BallBag) 
  (h_total : bag.total_balls = 4)
  (h_prob : prob_white bag = 1/2) :
  (bag.white_balls = 2) ∧ 
  (1/3 : ℚ) = (bag.white_balls * (bag.white_balls - 1) + bag.black_balls * (bag.black_balls - 1)) / 
               (bag.total_balls * (bag.total_balls - 1)) := by
  sorry


end ball_picking_problem_l4083_408312


namespace polynomial_nonzero_coeffs_l4083_408370

/-- A polynomial has at least n+1 nonzero coefficients if its degree is at least n -/
def HasAtLeastNPlusOneNonzeroCoeffs (p : Polynomial ℝ) (n : ℕ) : Prop :=
  (Finset.filter (· ≠ 0) p.support).card ≥ n + 1

/-- The main theorem statement -/
theorem polynomial_nonzero_coeffs
  (a : ℝ) (k : ℕ) (Q : Polynomial ℝ) 
  (ha : a ≠ 0) (hQ : Q ≠ 0) :
  let W := (Polynomial.X - Polynomial.C a)^k * Q
  HasAtLeastNPlusOneNonzeroCoeffs W k := by
sorry

end polynomial_nonzero_coeffs_l4083_408370


namespace factorization_theorem_1_factorization_theorem_2_l4083_408333

-- Define variables
variable (a b x y p : ℝ)

-- Theorem for the first expression
theorem factorization_theorem_1 : 
  8*a*x - b*x + 8*a*y - b*y = (x + y)*(8*a - b) := by sorry

-- Theorem for the second expression
theorem factorization_theorem_2 : 
  a*p + a*x - 2*b*x - 2*b*p = (p + x)*(a - 2*b) := by sorry

end factorization_theorem_1_factorization_theorem_2_l4083_408333


namespace six_paths_from_M_to_N_l4083_408335

/- Define a directed graph with vertices and edges -/
def Graph : Type := List (Char × Char)

/- Define the graph structure for our problem -/
def problemGraph : Graph := [
  ('M', 'A'), ('M', 'B'),
  ('A', 'C'), ('A', 'D'),
  ('B', 'C'), ('B', 'N'),
  ('C', 'N'), ('D', 'N')
]

/- Function to count paths between two vertices -/
def countPaths (g : Graph) (start finish : Char) : Nat :=
  sorry

/- Theorem stating that there are 6 paths from M to N -/
theorem six_paths_from_M_to_N :
  countPaths problemGraph 'M' 'N' = 6 :=
sorry

end six_paths_from_M_to_N_l4083_408335


namespace no_triangle_condition_l4083_408387

-- Define the lines
def line1 (x y : ℝ) : Prop := 2 * x - 3 * y + 1 = 0
def line2 (x y : ℝ) : Prop := 4 * x + 3 * y + 5 = 0
def line3 (m x y : ℝ) : Prop := m * x - y - 1 = 0

-- Define when three lines form a triangle
def form_triangle (l1 l2 l3 : ℝ → ℝ → Prop) : Prop :=
  ∃ (x1 y1 x2 y2 x3 y3 : ℝ),
    l1 x1 y1 ∧ l2 x2 y2 ∧ l3 x3 y3 ∧
    (x1 ≠ x2 ∨ y1 ≠ y2) ∧ (x2 ≠ x3 ∨ y2 ≠ y3) ∧ (x3 ≠ x1 ∨ y3 ≠ y1)

-- Theorem statement
theorem no_triangle_condition (m : ℝ) :
  ¬(form_triangle line1 line2 (line3 m)) ↔ m ∈ ({-4/3, 2/3, 4/3} : Set ℝ) :=
sorry

end no_triangle_condition_l4083_408387
