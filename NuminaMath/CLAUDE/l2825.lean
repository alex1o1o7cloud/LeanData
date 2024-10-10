import Mathlib

namespace toy_sales_earnings_difference_l2825_282556

theorem toy_sales_earnings_difference :
  let bert_initial_price : ℝ := 18
  let bert_initial_quantity : ℕ := 10
  let bert_discount_percentage : ℝ := 0.15
  let bert_discounted_quantity : ℕ := 3

  let tory_initial_price : ℝ := 20
  let tory_initial_quantity : ℕ := 15
  let tory_discount_percentage : ℝ := 0.10
  let tory_discounted_quantity : ℕ := 7

  let tax_rate : ℝ := 0.05

  let bert_earnings : ℝ := 
    (bert_initial_price * bert_initial_quantity - 
     bert_discount_percentage * bert_initial_price * bert_discounted_quantity) * 
    (1 + tax_rate)

  let tory_earnings : ℝ := 
    (tory_initial_price * tory_initial_quantity - 
     tory_discount_percentage * tory_initial_price * tory_discounted_quantity) * 
    (1 + tax_rate)

  tory_earnings - bert_earnings = 119.805 :=
by sorry

end toy_sales_earnings_difference_l2825_282556


namespace ice_cream_price_l2825_282567

theorem ice_cream_price (game_cost : ℚ) (num_ice_creams : ℕ) (h1 : game_cost = 60) (h2 : num_ice_creams = 24) :
  game_cost / num_ice_creams = 2.5 := by
  sorry

end ice_cream_price_l2825_282567


namespace x_range_l2825_282590

theorem x_range (x : ℝ) (h1 : 1/x < 4) (h2 : 1/x > -6) (h3 : x < 0) :
  -1/6 < x ∧ x < 0 := by
  sorry

end x_range_l2825_282590


namespace polynomial_root_product_l2825_282522

theorem polynomial_root_product (b c : ℤ) : 
  (∀ r : ℝ, r^2 - r - 2 = 0 → r^4 - b*r - c = 0) → b*c = 30 := by
  sorry

end polynomial_root_product_l2825_282522


namespace cycling_time_difference_l2825_282534

-- Define the distances and speeds for each day
def monday_distance : ℝ := 3
def monday_speed : ℝ := 6
def tuesday_distance : ℝ := 4
def tuesday_speed : ℝ := 4
def thursday_distance : ℝ := 3
def thursday_speed : ℝ := 3
def saturday_distance : ℝ := 2
def saturday_speed : ℝ := 8

-- Define the constant speed
def constant_speed : ℝ := 5

-- Define the total distance
def total_distance : ℝ := monday_distance + tuesday_distance + thursday_distance + saturday_distance

-- Theorem statement
theorem cycling_time_difference : 
  let actual_time := (monday_distance / monday_speed) + 
                     (tuesday_distance / tuesday_speed) + 
                     (thursday_distance / thursday_speed) + 
                     (saturday_distance / saturday_speed)
  let constant_time := total_distance / constant_speed
  ((actual_time - constant_time) * 60) = 21 := by
  sorry

end cycling_time_difference_l2825_282534


namespace farm_acreage_difference_l2825_282509

theorem farm_acreage_difference (total_acres flax_acres : ℕ) 
  (h1 : total_acres = 240)
  (h2 : flax_acres = 80)
  (h3 : flax_acres < total_acres - flax_acres) : 
  total_acres - flax_acres - flax_acres = 80 := by
sorry

end farm_acreage_difference_l2825_282509


namespace representation_625_ends_with_1_l2825_282540

def base_count : ℕ := 4

theorem representation_625_ends_with_1 :
  (∃ (S : Finset ℕ), (∀ b ∈ S, 3 ≤ b ∧ b ≤ 10) ∧
   (∀ b ∈ S, (625 : ℕ) % b = 1) ∧
   S.card = base_count) :=
by sorry

end representation_625_ends_with_1_l2825_282540


namespace symmetric_points_sum_l2825_282587

/-- 
Given two points A and B that are symmetric with respect to the origin,
prove that the sum of their x and y coordinates is -2.
-/
theorem symmetric_points_sum (m n : ℝ) : 
  (3 : ℝ) = -(-m) → n = -(5 : ℝ) → m + n = -2 := by sorry

end symmetric_points_sum_l2825_282587


namespace quadratic_transformation_l2825_282586

theorem quadratic_transformation (m : ℝ) : 
  (∀ x : ℝ, x^2 - 2*(m+1)*x + 16 = (x-4)^2) → m = 3 := by
  sorry

end quadratic_transformation_l2825_282586


namespace inequality_proof_l2825_282588

theorem inequality_proof (a b c : ℝ) 
  (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0)
  (h_sum : a + b + c = 1) : 
  (Real.sqrt (3 * a + 1) + Real.sqrt (3 * b + 1) + Real.sqrt (3 * c + 1) ≤ 3 * Real.sqrt 2) ∧
  (2 * (a^3 + b^3 + c^3) ≥ a*b + b*c + c*a - 3*a*b*c) := by
  sorry

end inequality_proof_l2825_282588


namespace increasing_decreasing_behavior_l2825_282512

theorem increasing_decreasing_behavior 
  (f : ℝ → ℝ) (a : ℝ) (n : ℕ) 
  (h_f : ∀ x, f x = a * x ^ n) 
  (h_a : a ≠ 0) :
  (n % 2 = 0 ∧ a > 0 → ∀ x ≠ 0, deriv f x > 0) ∧
  (n % 2 = 0 ∧ a < 0 → ∀ x ≠ 0, deriv f x < 0) ∧
  (n % 2 = 1 ∧ a > 0 → (∀ x > 0, deriv f x > 0) ∧ (∀ x < 0, deriv f x < 0)) ∧
  (n % 2 = 1 ∧ a < 0 → (∀ x > 0, deriv f x < 0) ∧ (∀ x < 0, deriv f x > 0)) :=
by sorry

end increasing_decreasing_behavior_l2825_282512


namespace equation_undefined_at_five_l2825_282533

theorem equation_undefined_at_five :
  ¬∃ (y : ℝ), (1 / (5 + 5) + 1 / (5 - 5) : ℝ) = y :=
sorry

end equation_undefined_at_five_l2825_282533


namespace chess_game_probability_l2825_282525

theorem chess_game_probability (draw_prob win_b_prob : ℚ) :
  draw_prob = 1/2 →
  win_b_prob = 1/3 →
  1 - win_b_prob = 2/3 :=
by
  sorry

end chess_game_probability_l2825_282525


namespace line_parallel_to_countless_lines_l2825_282521

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallel relation between lines
variable (parallel : Line → Line → Prop)

-- Define the parallel relation between a line and a plane
variable (parallel_to_plane : Line → Plane → Prop)

-- Define the containment relation of a line in a plane
variable (contained_in : Line → Plane → Prop)

-- Define a property for a line being parallel to countless lines in a plane
variable (parallel_to_countless_lines : Line → Plane → Prop)

-- State the theorem
theorem line_parallel_to_countless_lines 
  (a b : Line) (α : Plane) :
  parallel a b → contained_in b α → 
  parallel_to_countless_lines a α :=
sorry

end line_parallel_to_countless_lines_l2825_282521


namespace total_revenue_is_4586_80_l2825_282566

structure PhoneModel where
  name : String
  initialInventory : ℕ
  price : ℚ
  discountRate : ℚ
  taxRate : ℚ
  damaged : ℕ
  finalInventory : ℕ

def calculateRevenue (model : PhoneModel) : ℚ :=
  let discountedPrice := model.price * (1 - model.discountRate)
  let priceAfterTax := discountedPrice * (1 + model.taxRate)
  let soldUnits := model.initialInventory - model.finalInventory - model.damaged
  soldUnits * priceAfterTax

def totalRevenue (models : List PhoneModel) : ℚ :=
  models.map calculateRevenue |>.sum

def phoneModels : List PhoneModel := [
  { name := "Samsung Galaxy S20", initialInventory := 14, price := 800, discountRate := 0.1, taxRate := 0.12, damaged := 2, finalInventory := 10 },
  { name := "iPhone 12", initialInventory := 8, price := 1000, discountRate := 0.15, taxRate := 0.1, damaged := 1, finalInventory := 5 },
  { name := "Google Pixel 5", initialInventory := 7, price := 700, discountRate := 0.05, taxRate := 0.08, damaged := 0, finalInventory := 8 },
  { name := "OnePlus 8T", initialInventory := 6, price := 600, discountRate := 0.2, taxRate := 0.15, damaged := 1, finalInventory := 3 }
]

theorem total_revenue_is_4586_80 :
  totalRevenue phoneModels = 4586.8 := by
  sorry

end total_revenue_is_4586_80_l2825_282566


namespace oil_drop_probability_l2825_282589

theorem oil_drop_probability (c : ℝ) (h : c > 0) : 
  (0.5 * c)^2 / (π * (c/2)^2) = 0.25 / π := by
  sorry

end oil_drop_probability_l2825_282589


namespace function_value_solution_l2825_282550

theorem function_value_solution (x : ℝ) :
  (x^2 + x - 1 = 5) ↔ (x = 2 ∨ x = -3) := by sorry

end function_value_solution_l2825_282550


namespace binomial_20_10_l2825_282574

theorem binomial_20_10 (h1 : Nat.choose 18 9 = 48620) 
                       (h2 : Nat.choose 18 10 = 43758) 
                       (h3 : Nat.choose 18 11 = 24310) : 
  Nat.choose 20 10 = 97240 := by
  sorry

end binomial_20_10_l2825_282574


namespace acute_triangle_angles_l2825_282568

-- Define an acute triangle
def is_acute_triangle (a b c : ℝ) : Prop :=
  0 < a ∧ a < 90 ∧
  0 < b ∧ b < 90 ∧
  0 < c ∧ c < 90 ∧
  a + b + c = 180

-- Theorem statement
theorem acute_triangle_angles (a b c : ℝ) :
  is_acute_triangle a b c →
  ∃ (x y z : ℝ), is_acute_triangle x y z ∧ x > 45 ∧ y > 45 ∧ z > 45 :=
by
  sorry

end acute_triangle_angles_l2825_282568


namespace bens_gross_income_l2825_282551

theorem bens_gross_income (car_payment insurance maintenance fuel : ℝ)
  (h1 : car_payment = 400)
  (h2 : insurance = 150)
  (h3 : maintenance = 75)
  (h4 : fuel = 50)
  (h5 : ∀ after_tax_income : ℝ, 
    0.2 * after_tax_income = car_payment + insurance + maintenance + fuel)
  (h6 : ∀ gross_income : ℝ, 
    (2/3) * gross_income = after_tax_income) :
  ∃ gross_income : ℝ, gross_income = 5062.50 := by
sorry

end bens_gross_income_l2825_282551


namespace work_completion_time_l2825_282558

/-- Proves that if A is thrice as fast as B and together they can do a work in 15 days, 
    then A alone can do the work in 20 days. -/
theorem work_completion_time 
  (a b : ℝ) -- Work rates of A and B
  (h1 : a = 3 * b) -- A is thrice as fast as B
  (h2 : (a + b) * 15 = 1) -- Together, A and B can do the work in 15 days
  : a * 20 = 1 := by -- A alone can do the work in 20 days
sorry


end work_completion_time_l2825_282558


namespace product_234_75_in_base5_l2825_282514

/-- Converts a decimal number to its base 5 representation -/
def toBase5 (n : ℕ) : List ℕ :=
  sorry

/-- Converts a list of base 5 digits to a decimal number -/
def fromBase5 (digits : List ℕ) : ℕ :=
  sorry

/-- Multiplies two numbers in base 5 representation -/
def multiplyBase5 (a b : List ℕ) : List ℕ :=
  sorry

theorem product_234_75_in_base5 :
  let a := toBase5 234
  let b := toBase5 75
  multiplyBase5 a b = [4, 5, 0, 6, 2, 0] :=
sorry

end product_234_75_in_base5_l2825_282514


namespace tangent_line_equation_l2825_282584

-- Define the curve
def f (x : ℝ) : ℝ := x^3

-- Define the point through which the tangent line passes
def point : ℝ × ℝ := (1, 1)

-- Define the two possible tangent line equations
def tangent1 (x y : ℝ) : Prop := 3*x - y - 2 = 0
def tangent2 (x y : ℝ) : Prop := 3*x - 4*y + 1 = 0

-- Theorem statement
theorem tangent_line_equation :
  ∃ (x₀ y₀ : ℝ), 
    (y₀ = f x₀) ∧ 
    ((tangent1 x₀ y₀ ∧ (∀ x : ℝ, tangent1 x (f x) → x = x₀)) ∨
     (tangent2 x₀ y₀ ∧ (∀ x : ℝ, tangent2 x (f x) → x = x₀))) ∧
    (point.1 = 1 ∧ point.2 = 1) :=
sorry

end tangent_line_equation_l2825_282584


namespace remainder_divisibility_l2825_282527

theorem remainder_divisibility (N : ℤ) : 
  N % 342 = 47 → N % 19 = 9 := by
  sorry

end remainder_divisibility_l2825_282527


namespace sequence_sign_change_l2825_282578

theorem sequence_sign_change (a₀ c : ℝ) (h₁ : a₀ > 0) (h₂ : c > 0) : 
  ∃ (a : ℕ → ℝ), a 0 = a₀ ∧ 
  (∀ n, a (n + 1) = (a n + c) / (1 - a n * c)) ∧
  (∀ n, n < 1990 → a n > 0) ∧
  a 1990 < 0 := by
sorry

end sequence_sign_change_l2825_282578


namespace binomial_expansion_coefficients_l2825_282594

theorem binomial_expansion_coefficients :
  ∀ (a₀ a₁ a₂ a₃ a₄ a₅ : ℝ),
  (∀ x : ℝ, (1 + 2*x)^5 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5) →
  (a₄ = 80 ∧ a₁ + a₂ + a₃ = 130) := by
  sorry

end binomial_expansion_coefficients_l2825_282594


namespace series_solution_l2825_282532

/-- The sum of the infinite geometric series with first term a and common ratio r -/
noncomputable def geometric_sum (a : ℝ) (r : ℝ) : ℝ := a / (1 - r)

/-- The series in question -/
noncomputable def series (k : ℝ) : ℝ :=
  5 + geometric_sum ((5 + k) / 3) (1 / 3)

theorem series_solution :
  ∃ k : ℝ, series k = 15 ∧ k = 7.5 := by sorry

end series_solution_l2825_282532


namespace pineapple_sweets_count_l2825_282515

/-- Proves the number of initial pineapple-flavored sweets in a candy packet --/
theorem pineapple_sweets_count (cherry : ℕ) (strawberry : ℕ) (remaining : ℕ) : 
  cherry = 30 → 
  strawberry = 40 → 
  remaining = 55 → 
  ∃ (pineapple : ℕ), 
    pineapple + cherry + strawberry = 
    2 * remaining + 5 + (cherry / 2) + (strawberry / 2) ∧ 
    pineapple = 50 := by
  sorry

#check pineapple_sweets_count

end pineapple_sweets_count_l2825_282515


namespace sets_and_conditions_l2825_282563

def A : Set ℝ := {x | -2 < x ∧ x < 5}
def B (a : ℝ) : Set ℝ := {x | 2 - a < x ∧ x < 1 + 2*a}

theorem sets_and_conditions :
  (∀ x, x ∈ (A ∪ B 3) ↔ -2 < x ∧ x < 7) ∧
  (∀ x, x ∈ (A ∩ B 3) ↔ -1 < x ∧ x < 5) ∧
  (∀ a, (∀ x, x ∈ B a → x ∈ A) ↔ a ≤ 2) :=
by sorry

end sets_and_conditions_l2825_282563


namespace seven_eighths_of_48_l2825_282528

theorem seven_eighths_of_48 : (7 / 8 : ℚ) * 48 = 42 := by
  sorry

end seven_eighths_of_48_l2825_282528


namespace problem_solution_l2825_282585

-- Define the solution set for x(x-2) < 0
def solution_set := {x : ℝ | x * (x - 2) < 0}

-- Define the proposed incorrect solution set
def incorrect_set := {x : ℝ | x < 0 ∨ x > 2}

-- Define a triangle ABC
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ

-- Theorem statement
theorem problem_solution :
  (solution_set ≠ incorrect_set) ∧
  (∀ t : Triangle, t.A > t.B ↔ Real.sin t.A > Real.sin t.B) :=
sorry

end problem_solution_l2825_282585


namespace unique_zero_implies_a_range_l2825_282571

/-- A cubic function with a parameter a -/
def f (a : ℝ) (x : ℝ) : ℝ := a * x^3 - 6 * x^2 + 1

/-- The derivative of f with respect to x -/
def f' (a : ℝ) (x : ℝ) : ℝ := 3 * a * x^2 - 12 * x

theorem unique_zero_implies_a_range (a : ℝ) :
  (∃! x₀ : ℝ, f a x₀ = 0 ∧ x₀ > 0) → a < -4 * Real.sqrt 2 := by
  sorry

end unique_zero_implies_a_range_l2825_282571


namespace wage_decrease_percentage_l2825_282506

theorem wage_decrease_percentage (W : ℝ) (P : ℝ) : 
  W > 0 →  -- Wages are positive
  0.20 * (W - (P / 100) * W) = 0.50 * (0.30 * W) → 
  P = 25 :=
by sorry

end wage_decrease_percentage_l2825_282506


namespace remainder_problem_l2825_282592

theorem remainder_problem (N : ℤ) (h : N % 899 = 63) : N % 29 = 5 := by
  sorry

end remainder_problem_l2825_282592


namespace greatest_q_minus_r_l2825_282545

theorem greatest_q_minus_r : ∃ (q r : ℕ), 
  945 = 21 * q + r ∧ 
  q > 0 ∧ 
  r > 0 ∧ 
  ∀ (q' r' : ℕ), 945 = 21 * q' + r' ∧ q' > 0 ∧ r' > 0 → q - r ≥ q' - r' :=
by sorry

end greatest_q_minus_r_l2825_282545


namespace charlie_original_price_l2825_282582

-- Define the given quantities
def alice_acorns : ℕ := 3600
def bob_acorns : ℕ := 2400
def charlie_acorns : ℕ := 4500
def bob_total_price : ℚ := 6000
def discount_rate : ℚ := 0.1

-- Define the relationships
def bob_price_per_acorn : ℚ := bob_total_price / bob_acorns
def alice_price_per_acorn : ℚ := 9 * bob_price_per_acorn
def average_price_per_acorn : ℚ := (alice_price_per_acorn * alice_acorns + bob_price_per_acorn * bob_acorns) / (alice_acorns + bob_acorns)
def charlie_discounted_price_per_acorn : ℚ := average_price_per_acorn * (1 - discount_rate)

-- State the theorem
theorem charlie_original_price : 
  charlie_acorns * average_price_per_acorn = 65250 := by sorry

end charlie_original_price_l2825_282582


namespace angle_d_measure_l2825_282599

/-- A scalene triangle with specific angle relationships -/
structure ScaleneTriangle where
  /-- Measure of angle D in degrees -/
  angleD : ℝ
  /-- Measure of angle E in degrees -/
  angleE : ℝ
  /-- Measure of angle F in degrees -/
  angleF : ℝ
  /-- Triangle is scalene -/
  scalene : angleD ≠ angleE ∧ angleE ≠ angleF ∧ angleD ≠ angleF
  /-- Angle E is twice angle D -/
  e_twice_d : angleE = 2 * angleD
  /-- Angle F is 40 degrees -/
  f_is_40 : angleF = 40
  /-- Sum of angles in a triangle is 180 degrees -/
  angle_sum : angleD + angleE + angleF = 180

/-- Theorem: In a scalene triangle DEF with the given conditions, angle D measures 140/3 degrees -/
theorem angle_d_measure (t : ScaleneTriangle) : t.angleD = 140 / 3 := by
  sorry

end angle_d_measure_l2825_282599


namespace piglets_count_l2825_282541

/-- Calculates the number of piglets given the total number of straws and straws per piglet -/
def number_of_piglets (total_straws : ℕ) (straws_per_piglet : ℕ) : ℕ :=
  let straws_for_adult_pigs := (3 * total_straws) / 5
  let straws_for_piglets := straws_for_adult_pigs
  straws_for_piglets / straws_per_piglet

/-- Proves that the number of piglets is 30 given the problem conditions -/
theorem piglets_count : number_of_piglets 300 6 = 30 := by
  sorry

end piglets_count_l2825_282541


namespace medicine_types_count_l2825_282546

/-- The number of medical boxes -/
def num_boxes : ℕ := 5

/-- The number of boxes each medicine appears in -/
def boxes_per_medicine : ℕ := 2

/-- Calculates the binomial coefficient (n choose k) -/
def binomial (n k : ℕ) : ℕ :=
  if k > n then 0
  else Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

/-- The number of types of medicine -/
def num_medicine_types : ℕ := binomial num_boxes boxes_per_medicine

theorem medicine_types_count : num_medicine_types = 10 := by
  sorry

end medicine_types_count_l2825_282546


namespace triangle_count_is_twenty_l2825_282529

/-- Represents a point in 2D space -/
structure Point :=
  (x : ℝ) (y : ℝ)

/-- Represents a square with diagonals and midpoint segments -/
structure SquareWithDiagonalsAndMidpoints :=
  (vertices : Fin 4 → Point)
  (diagonals : Fin 2 → Point × Point)
  (midpoints : Fin 4 → Point)
  (cross : Point × Point)

/-- Counts the number of triangles in the figure -/
def countTriangles (square : SquareWithDiagonalsAndMidpoints) : ℕ :=
  sorry

/-- Theorem stating that the number of triangles in the figure is 20 -/
theorem triangle_count_is_twenty (square : SquareWithDiagonalsAndMidpoints) :
  countTriangles square = 20 :=
sorry

end triangle_count_is_twenty_l2825_282529


namespace floor_ceiling_sum_l2825_282535

theorem floor_ceiling_sum : ⌊(-3.67 : ℝ)⌋ + ⌈(30.3 : ℝ)⌉ = 27 := by
  sorry

end floor_ceiling_sum_l2825_282535


namespace door_height_calculation_l2825_282502

/-- Calculates the height of a door in a room given the room dimensions, door width, window dimensions, number of windows, cost of white washing per square foot, and total cost of white washing. -/
theorem door_height_calculation (room_length room_width room_height : ℝ)
                                (door_width : ℝ)
                                (window_length window_width : ℝ)
                                (num_windows : ℕ)
                                (cost_per_sqft : ℝ)
                                (total_cost : ℝ) :
  room_length = 25 ∧ room_width = 15 ∧ room_height = 12 ∧
  door_width = 3 ∧
  window_length = 4 ∧ window_width = 3 ∧
  num_windows = 3 ∧
  cost_per_sqft = 3 ∧
  total_cost = 2718 →
  ∃ (door_height : ℝ),
    door_height = 6 ∧
    total_cost = (2 * (room_length * room_height + room_width * room_height) -
                  (door_height * door_width + ↑num_windows * window_length * window_width)) * cost_per_sqft :=
by sorry

end door_height_calculation_l2825_282502


namespace inequality_proof_l2825_282517

theorem inequality_proof (x y z : ℝ) 
  (h_pos_x : 0 < x) (h_pos_y : 0 < y) (h_pos_z : 0 < z)
  (h_sum : x + 2*y + 3*z = 11/12) :
  6*(3*x*y + 4*x*z + 2*y*z) + 6*x + 3*y + 4*z + 72*x*y*z ≤ 107/18 := by
sorry

end inequality_proof_l2825_282517


namespace solution_set_part1_range_of_a_part2_l2825_282559

-- Define the quadratic function f(x)
def f (a b : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + 2

-- Part 1
theorem solution_set_part1 :
  {x : ℝ | f (-12) (-2) x > 0} = {x : ℝ | -1/2 < x ∧ x < 1/3} := by sorry

-- Part 2
theorem range_of_a_part2 :
  ∀ a : ℝ, (∀ x : ℝ, f a (-1) x ≥ 0) ↔ a ≥ 1/8 := by sorry

end solution_set_part1_range_of_a_part2_l2825_282559


namespace first_fabulous_friday_is_oct31_l2825_282543

/-- Represents a date with a day, month, and day of the week -/
structure Date where
  day : Nat
  month : Nat
  dayOfWeek : Nat
  deriving Repr

/-- Represents a school calendar -/
structure SchoolCalendar where
  startDate : Date
  deriving Repr

/-- Determines if a given date is a Fabulous Friday -/
def isFabulousFriday (d : Date) : Bool :=
  sorry

/-- Finds the first Fabulous Friday after the school start date -/
def firstFabulousFriday (sc : SchoolCalendar) : Date :=
  sorry

/-- Theorem stating that the first Fabulous Friday after school starts on Tuesday, October 3 is October 31 -/
theorem first_fabulous_friday_is_oct31 (sc : SchoolCalendar) :
  sc.startDate = Date.mk 3 10 2 →  -- October 3 is a Tuesday (day 2 of the week)
  firstFabulousFriday sc = Date.mk 31 10 5 :=  -- October 31 is a Friday (day 5 of the week)
  sorry

end first_fabulous_friday_is_oct31_l2825_282543


namespace derivative_odd_implies_a_eq_neg_one_l2825_282501

/-- Given a real number a and a function f(x) = e^x - ae^(-x), 
    if the derivative of f is an odd function, then a = -1. -/
theorem derivative_odd_implies_a_eq_neg_one (a : ℝ) : 
  let f : ℝ → ℝ := λ x ↦ Real.exp x - a * Real.exp (-x)
  let f' : ℝ → ℝ := λ x ↦ Real.exp x + a * Real.exp (-x)
  (∀ x, f' x = -f' (-x)) → a = -1 := by
  sorry

end derivative_odd_implies_a_eq_neg_one_l2825_282501


namespace symmetric_point_coordinates_l2825_282547

/-- A point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Symmetry with respect to the origin -/
def symmetricPoint (p : Point2D) : Point2D :=
  ⟨-p.x, -p.y⟩

theorem symmetric_point_coordinates :
  let p : Point2D := ⟨1, -2⟩
  let q : Point2D := symmetricPoint p
  q.x = -1 ∧ q.y = 2 := by
  sorry

end symmetric_point_coordinates_l2825_282547


namespace carly_running_ratio_l2825_282564

def week1_distance : ℝ := 2
def week2_distance : ℝ := 2 * week1_distance + 3
def week4_distance : ℝ := 4
def week3_distance : ℝ := week4_distance + 5

theorem carly_running_ratio :
  week3_distance / week2_distance = 9 / 7 := by
  sorry

end carly_running_ratio_l2825_282564


namespace board_covering_l2825_282526

def can_cover (m n : ℕ) : Prop :=
  ∃ (a b : ℕ), m * n = 3 * a + 10 * b

def excluded_pairs : Set (ℕ × ℕ) :=
  {(4,4), (2,2), (2,4), (2,7)}

def excluded_1xn (n : ℕ) : Prop :=
  ∃ (k : ℕ), n = 3*k + 1 ∨ n = 3*k + 2

theorem board_covering (m n : ℕ) :
  can_cover m n ↔ (m, n) ∉ excluded_pairs ∧ (m ≠ 1 ∨ ¬excluded_1xn n) :=
sorry

end board_covering_l2825_282526


namespace strawberry_jelly_amount_l2825_282513

theorem strawberry_jelly_amount (total_jelly : ℕ) (blueberry_jelly : ℕ) 
  (h1 : total_jelly = 6310)
  (h2 : blueberry_jelly = 4518) :
  total_jelly - blueberry_jelly = 1792 := by
  sorry

end strawberry_jelly_amount_l2825_282513


namespace largest_divisor_of_n_l2825_282565

theorem largest_divisor_of_n (n : ℕ) (h1 : n > 0) (h2 : 72 ∣ n^2) :
  ∃ w : ℕ, w > 0 ∧ w ∣ n ∧ ∀ k : ℕ, k > 0 ∧ k ∣ n → k ≤ w ∧ w = 12 :=
sorry

end largest_divisor_of_n_l2825_282565


namespace no_special_polynomials_l2825_282570

/-- Represents a polynomial of the form x^4 + ax^3 + bx^2 + cx + 2048 -/
def SpecialPolynomial (a b c : ℝ) (x : ℂ) : ℂ :=
  x^4 + a*x^3 + b*x^2 + c*x + 2048

/-- Predicate to check if a complex number is a root of the polynomial -/
def IsRoot (a b c : ℝ) (s : ℂ) : Prop :=
  SpecialPolynomial a b c s = 0

/-- Predicate to check if the polynomial satisfies the special root property -/
def HasSpecialRootProperty (a b c : ℝ) : Prop :=
  ∀ s : ℂ, IsRoot a b c s → IsRoot a b c (s^2) ∧ IsRoot a b c (s⁻¹)

theorem no_special_polynomials :
  ¬∃ a b c : ℝ, HasSpecialRootProperty a b c :=
sorry

end no_special_polynomials_l2825_282570


namespace equation_solution_l2825_282549

theorem equation_solution : ∃ x : ℝ, x ≠ 0 ∧ (1 / x + (3 / x) / (6 / x) - 5 / x = 0.5) ∧ x = 8 := by
  sorry

end equation_solution_l2825_282549


namespace proposition_implications_l2825_282577

def p (a : ℝ) : Prop := 1 ∈ {x : ℝ | x^2 < a}
def q (a : ℝ) : Prop := 2 ∈ {x : ℝ | x^2 < a}

theorem proposition_implications (a : ℝ) :
  ((p a ∨ q a) → a > 1) ∧ ((p a ∧ q a) → a > 4) := by sorry

end proposition_implications_l2825_282577


namespace min_sum_of_product_1806_l2825_282511

theorem min_sum_of_product_1806 (x y z : ℕ+) (h : x * y * z = 1806) :
  ∃ (a b c : ℕ+), a * b * c = 1806 ∧ a + b + c ≤ x + y + z ∧ a + b + c = 72 :=
sorry

end min_sum_of_product_1806_l2825_282511


namespace maria_assembly_time_l2825_282555

/-- Represents the time taken to assemble furniture items -/
structure AssemblyTime where
  chairs : Nat
  tables : Nat
  bookshelf : Nat
  tv_stand : Nat

/-- Calculates the total assembly time for all furniture items -/
def total_assembly_time (time : AssemblyTime) (num_chairs num_tables : Nat) : Nat :=
  num_chairs * time.chairs + num_tables * time.tables + time.bookshelf + time.tv_stand

/-- Theorem: The total assembly time for Maria's furniture is 100 minutes -/
theorem maria_assembly_time :
  let time : AssemblyTime := { chairs := 8, tables := 12, bookshelf := 25, tv_stand := 35 }
  total_assembly_time time 2 2 = 100 := by
  sorry

end maria_assembly_time_l2825_282555


namespace intersection_count_l2825_282539

/-- Represents a lattice point in the coordinate plane -/
structure LatticePoint where
  x : ℤ
  y : ℤ

/-- Represents a circle centered at a lattice point -/
structure Circle where
  center : LatticePoint
  radius : ℚ

/-- Represents a square centered at a lattice point -/
structure Square where
  center : LatticePoint
  sideLength : ℚ

/-- Represents a line segment from (0,0) to (703, 299) -/
def lineSegment : Set (ℚ × ℚ) :=
  {p | ∃ t : ℚ, 0 ≤ t ∧ t ≤ 1 ∧ p = (703 * t, 299 * t)}

/-- Counts the number of intersections with squares and circles -/
def countIntersections (line : Set (ℚ × ℚ)) (squares : Set Square) (circles : Set Circle) : ℕ :=
  sorry

/-- Main theorem statement -/
theorem intersection_count :
  ∀ (squares : Set Square) (circles : Set Circle),
    (∀ p : LatticePoint, ∃ s ∈ squares, s.center = p ∧ s.sideLength = 2/5) →
    (∀ p : LatticePoint, ∃ c ∈ circles, c.center = p ∧ c.radius = 1/5) →
    countIntersections lineSegment squares circles = 2109 := by
  sorry

end intersection_count_l2825_282539


namespace number_square_equation_l2825_282500

theorem number_square_equation : ∃ x : ℝ, x^2 + 100 = (x - 20)^2 ∧ x = 7.5 := by
  sorry

end number_square_equation_l2825_282500


namespace mikis_sandcastle_height_l2825_282581

/-- The height of Miki's sister's sandcastle in feet -/
def sisters_height : ℝ := 0.5

/-- The difference in height between Miki's and her sister's sandcastles in feet -/
def height_difference : ℝ := 0.33

/-- The height of Miki's sandcastle in feet -/
def mikis_height : ℝ := sisters_height + height_difference

theorem mikis_sandcastle_height : mikis_height = 0.83 := by
  sorry

end mikis_sandcastle_height_l2825_282581


namespace certain_number_exists_and_unique_l2825_282542

theorem certain_number_exists_and_unique : 
  ∃! x : ℕ, 220050 = (x + 445) * (2 * (x - 445)) + 50 :=
sorry

end certain_number_exists_and_unique_l2825_282542


namespace ellipse_major_axis_length_l2825_282516

-- Define the foci of the ellipse
def F1 : ℝ × ℝ := (3, 15)
def F2 : ℝ × ℝ := (28, 45)

-- Define the reflection of F1 over the y-axis
def F1_reflected : ℝ × ℝ := (-3, 15)

-- Define the ellipse
def is_on_ellipse (P : ℝ × ℝ) (k : ℝ) : Prop :=
  dist P F1 + dist P F2 = k

-- Define the tangency condition
def is_tangent_to_y_axis (k : ℝ) : Prop :=
  ∃ y : ℝ, is_on_ellipse (0, y) k ∧
    ∀ y' : ℝ, is_on_ellipse (0, y') k → y = y'

-- State the theorem
theorem ellipse_major_axis_length :
  ∃ k : ℝ, is_tangent_to_y_axis k ∧ k = dist F1_reflected F2 :=
sorry

end ellipse_major_axis_length_l2825_282516


namespace pie_sugar_percentage_l2825_282560

/-- Given a pie weighing 200 grams with 50 grams of sugar, 
    prove that 75% of the pie is not sugar. -/
theorem pie_sugar_percentage 
  (total_weight : ℝ) 
  (sugar_weight : ℝ) 
  (h1 : total_weight = 200) 
  (h2 : sugar_weight = 50) : 
  (total_weight - sugar_weight) / total_weight * 100 = 75 := by
sorry

end pie_sugar_percentage_l2825_282560


namespace more_even_products_l2825_282575

def S : Finset Nat := {1, 2, 3, 4, 5}

def pairs : Finset (Nat × Nat) :=
  S.product S |>.filter (λ (a, b) => a ≤ b)

def products : Finset Nat :=
  pairs.image (λ (a, b) => a * b)

def evenProducts : Finset Nat :=
  products.filter (λ x => x % 2 = 0)

def oddProducts : Finset Nat :=
  products.filter (λ x => x % 2 ≠ 0)

theorem more_even_products :
  Finset.card evenProducts > Finset.card oddProducts :=
by sorry

end more_even_products_l2825_282575


namespace office_episodes_l2825_282504

theorem office_episodes (total_episodes : ℕ) (weeks : ℕ) (wednesday_episodes : ℕ) 
  (h1 : total_episodes = 201)
  (h2 : weeks = 67)
  (h3 : wednesday_episodes = 2) :
  ∃ monday_episodes : ℕ, 
    weeks * (monday_episodes + wednesday_episodes) = total_episodes ∧ 
    monday_episodes = 1 := by
  sorry

end office_episodes_l2825_282504


namespace max_plates_buyable_l2825_282597

/-- The cost of a pan -/
def pan_cost : ℕ := 3

/-- The cost of a pot -/
def pot_cost : ℕ := 5

/-- The cost of a plate -/
def plate_cost : ℕ := 10

/-- The total budget -/
def total_budget : ℕ := 100

/-- The minimum number of each item to buy -/
def min_items : ℕ := 2

/-- A function to calculate the total cost of the purchase -/
def total_cost (pans pots plates : ℕ) : ℕ :=
  pan_cost * pans + pot_cost * pots + plate_cost * plates

/-- The main theorem stating the maximum number of plates that can be bought -/
theorem max_plates_buyable :
  ∃ (pans pots plates : ℕ),
    pans ≥ min_items ∧
    pots ≥ min_items ∧
    plates ≥ min_items ∧
    total_cost pans pots plates = total_budget ∧
    plates = 8 ∧
    ∀ (p : ℕ), p > plates →
      ∀ (x y : ℕ), x ≥ min_items → y ≥ min_items →
        total_cost x y p ≠ total_budget :=
by sorry

end max_plates_buyable_l2825_282597


namespace average_weight_problem_l2825_282580

/-- Given three weights a, b, and c, prove that their average weights satisfy the given conditions and the average of a and b is 40. -/
theorem average_weight_problem (a b c : ℝ) : 
  (a + b + c) / 3 = 45 →
  (b + c) / 2 = 41 →
  b = 27 →
  (a + b) / 2 = 40 :=
by sorry

end average_weight_problem_l2825_282580


namespace toms_age_difference_l2825_282510

theorem toms_age_difference (sister_age : ℕ) : 
  sister_age + 9 = 14 →
  2 * sister_age - 9 = 1 := by
sorry

end toms_age_difference_l2825_282510


namespace touch_point_theorem_l2825_282572

/-- A right triangle with an inscribed circle -/
structure RightTriangleWithInscribedCircle where
  -- The length of the hypotenuse
  hypotenuse : ℝ
  -- The radius of the inscribed circle
  radius : ℝ
  -- Assumption that the hypotenuse is positive
  hypotenuse_pos : hypotenuse > 0
  -- Assumption that the radius is positive
  radius_pos : radius > 0

/-- The length from one vertex to where the circle touches the hypotenuse -/
def touchPoint (t : RightTriangleWithInscribedCircle) : Set ℝ :=
  {x : ℝ | x = t.hypotenuse / 2 - t.radius ∨ x = t.hypotenuse / 2 + t.radius}

theorem touch_point_theorem (t : RightTriangleWithInscribedCircle) 
    (h1 : t.hypotenuse = 10) (h2 : t.radius = 2) : 
    touchPoint t = {4, 6} := by
  sorry

end touch_point_theorem_l2825_282572


namespace adjacent_edge_angle_is_45_degrees_l2825_282520

/-- A regular tetrahedron with coinciding centers of inscribed and circumscribed spheres -/
structure RegularTetrahedron where
  -- The tetrahedron is regular
  is_regular : Bool
  -- The center of the circumscribed sphere coincides with the center of the inscribed sphere
  centers_coincide : Bool

/-- The angle between two adjacent edges of a regular tetrahedron -/
def adjacent_edge_angle (t : RegularTetrahedron) : ℝ := sorry

/-- Theorem: The angle between two adjacent edges of a regular tetrahedron 
    with coinciding sphere centers is 45 degrees -/
theorem adjacent_edge_angle_is_45_degrees (t : RegularTetrahedron) 
  (h1 : t.is_regular = true) 
  (h2 : t.centers_coincide = true) : 
  adjacent_edge_angle t = 45 * (π / 180) := by sorry

end adjacent_edge_angle_is_45_degrees_l2825_282520


namespace range_of_m_l2825_282503

theorem range_of_m (a b m : ℝ) (ha : a > 0) (hb : b > 0) 
  (h : ∀ (a b : ℝ), a > 0 → b > 0 → (1/a + 1/b) * Real.sqrt (a^2 + b^2) ≥ 2*m - 4) : 
  m ≤ 2 + Real.sqrt 2 := by
  sorry

end range_of_m_l2825_282503


namespace pages_left_to_read_l2825_282553

theorem pages_left_to_read (total_pages read_pages : ℕ) 
  (h1 : total_pages = 563)
  (h2 : read_pages = 147) :
  total_pages - read_pages = 416 := by
  sorry

end pages_left_to_read_l2825_282553


namespace distinct_configurations_eq_seven_l2825_282530

/-- The group of 2D rotations and flips for a 2x3 rectangle -/
inductive SymmetryGroup
| identity
| rotation180
| flipVertical
| flipHorizontal

/-- A configuration of red and yellow cubes in a 2x3 rectangle -/
def Configuration := Fin 6 → Bool

/-- The number of elements in the symmetry group -/
def symmetryGroupSize : ℕ := 4

/-- The total number of configurations -/
def totalConfigurations : ℕ := Nat.choose 6 3

/-- Function to count fixed points for each symmetry operation -/
noncomputable def fixedPoints (g : SymmetryGroup) : ℕ :=
  match g with
  | SymmetryGroup.identity => totalConfigurations
  | _ => 3  -- For rotation180, flipVertical, and flipHorizontal

/-- The sum of fixed points for all symmetry operations -/
noncomputable def totalFixedPoints : ℕ :=
  (fixedPoints SymmetryGroup.identity) +
  (fixedPoints SymmetryGroup.rotation180) +
  (fixedPoints SymmetryGroup.flipVertical) +
  (fixedPoints SymmetryGroup.flipHorizontal)

/-- The number of distinct configurations -/
noncomputable def distinctConfigurations : ℕ :=
  totalFixedPoints / symmetryGroupSize

theorem distinct_configurations_eq_seven :
  distinctConfigurations = 7 := by sorry

end distinct_configurations_eq_seven_l2825_282530


namespace stock_price_calculation_l2825_282591

/-- Proves that the original stock price is 100 given the conditions --/
theorem stock_price_calculation (X : ℝ) : 
  X * 0.95 + 0.001 * (X * 0.95) = 95.2 → X = 100 := by
  sorry

end stock_price_calculation_l2825_282591


namespace convex_polyhedron_properties_l2825_282538

/-- A convex polyhedron. -/
structure ConvexPolyhedron where
  -- Add necessary fields here
  convex : Bool

/-- The number of sides of a face in a polyhedron. -/
def face_sides (p : ConvexPolyhedron) (f : Nat) : Nat :=
  sorry

/-- The number of edges meeting at a vertex in a polyhedron. -/
def vertex_edges (p : ConvexPolyhedron) (v : Nat) : Nat :=
  sorry

/-- The set of all faces in a polyhedron. -/
def faces (p : ConvexPolyhedron) : Set Nat :=
  sorry

/-- The set of all vertices in a polyhedron. -/
def vertices (p : ConvexPolyhedron) : Set Nat :=
  sorry

theorem convex_polyhedron_properties (p : ConvexPolyhedron) (h : p.convex) :
  (∃ f ∈ faces p, face_sides p f ≤ 5) ∧
  (∃ v ∈ vertices p, vertex_edges p v ≤ 5) :=
sorry

end convex_polyhedron_properties_l2825_282538


namespace shopping_trip_expenses_l2825_282573

theorem shopping_trip_expenses (T : ℝ) (h_positive : T > 0) : 
  let clothing_percent : ℝ := 0.50
  let other_percent : ℝ := 0.30
  let clothing_tax : ℝ := 0.05
  let other_tax : ℝ := 0.10
  let total_tax_percent : ℝ := 0.055
  let food_percent : ℝ := 1 - clothing_percent - other_percent

  clothing_tax * clothing_percent * T + other_tax * other_percent * T = total_tax_percent * T →
  food_percent = 0.20 := by
sorry

end shopping_trip_expenses_l2825_282573


namespace complex_magnitude_product_l2825_282552

theorem complex_magnitude_product : Complex.abs (3 - 5 * Complex.I) * Complex.abs (3 + 5 * Complex.I) = 34 := by
  sorry

end complex_magnitude_product_l2825_282552


namespace pure_imaginary_product_imaginary_part_quotient_l2825_282505

-- Define complex numbers z₁ and z₂
def z₁ (m : ℝ) : ℂ := m + Complex.I
def z₂ (m : ℝ) : ℂ := 2 + m * Complex.I

-- Part 1
theorem pure_imaginary_product (m : ℝ) :
  (z₁ m * z₂ m).re = 0 → m = 0 :=
sorry

-- Part 2
theorem imaginary_part_quotient (m : ℝ) :
  z₁ m ^ 2 - 2 * z₁ m + 2 = 0 →
  (z₂ m / z₁ m).im = -1/2 :=
sorry

end pure_imaginary_product_imaginary_part_quotient_l2825_282505


namespace power_sum_zero_l2825_282519

theorem power_sum_zero : (-2 : ℤ)^(3^2) + 2^(3^2) = 0 := by
  sorry

end power_sum_zero_l2825_282519


namespace mike_percentage_l2825_282557

def phone_cost : ℝ := 1300
def additional_needed : ℝ := 780

theorem mike_percentage : 
  (phone_cost - additional_needed) / phone_cost * 100 = 40 := by
  sorry

end mike_percentage_l2825_282557


namespace smallest_x_multiple_of_53_l2825_282537

theorem smallest_x_multiple_of_53 : 
  ∃ (x : ℕ), x > 0 ∧ 
  (∀ (y : ℕ), y > 0 → y < x → ¬(53 ∣ ((3*y)^2 + 3*41*(3*y) + 41^2))) ∧
  (53 ∣ ((3*x)^2 + 3*41*(3*x) + 41^2)) ∧
  x = 4 := by
sorry

end smallest_x_multiple_of_53_l2825_282537


namespace cylinder_surface_area_l2825_282561

theorem cylinder_surface_area (h r : ℝ) (h_height : h = 12) (h_radius : r = 5) :
  2 * π * r^2 + 2 * π * r * h = 170 * π := by
  sorry

end cylinder_surface_area_l2825_282561


namespace grid_sum_theorem_l2825_282544

/-- Represents a 3x3 grid of numbers -/
def Grid := Fin 3 → Fin 3 → Nat

/-- Check if all numbers in the grid are unique and between 1 and 9 -/
def valid_grid (g : Grid) : Prop :=
  (∀ i j, g i j ∈ Finset.range 9) ∧
  (∀ i j k l, g i j = g k l → (i = k ∧ j = l))

/-- Sum of the right column -/
def right_column_sum (g : Grid) : Nat :=
  g 0 2 + g 1 2 + g 2 2

/-- Sum of the bottom row -/
def bottom_row_sum (g : Grid) : Nat :=
  g 2 0 + g 2 1 + g 2 2

theorem grid_sum_theorem (g : Grid) 
  (h_valid : valid_grid g) 
  (h_right_sum : right_column_sum g = 32) 
  (h_corner : g 2 2 = 7) : 
  bottom_row_sum g = 18 :=
sorry

end grid_sum_theorem_l2825_282544


namespace carla_tile_counting_l2825_282548

theorem carla_tile_counting (tiles : ℕ) (books : ℕ) (book_counts : ℕ) (total_counts : ℕ)
  (h1 : tiles = 38)
  (h2 : books = 75)
  (h3 : book_counts = 3)
  (h4 : total_counts = 301)
  : ∃ (tile_counts : ℕ), tile_counts * tiles + book_counts * books = total_counts ∧ tile_counts = 2 := by
  sorry

end carla_tile_counting_l2825_282548


namespace year_2049_is_jisi_l2825_282554

/-- Represents the Heavenly Stems -/
inductive HeavenlyStem
| Jia | Yi | Bing | Ding | Wu | Ji | Geng | Xin | Ren | Gui

/-- Represents the Earthly Branches -/
inductive EarthlyBranch
| Zi | Chou | Yin | Mao | Chen | Si | Wu | Wei | Shen | You | Xu | Hai

/-- Represents a year in the Heavenly Stems and Earthly Branches system -/
structure StemBranchYear :=
  (stem : HeavenlyStem)
  (branch : EarthlyBranch)

def next_stem (s : HeavenlyStem) : HeavenlyStem := sorry
def next_branch (b : EarthlyBranch) : EarthlyBranch := sorry

def advance_year (y : StemBranchYear) (n : ℕ) : StemBranchYear := sorry

theorem year_2049_is_jisi (year_2017 : StemBranchYear) 
  (h2017 : year_2017 = ⟨HeavenlyStem.Ding, EarthlyBranch.You⟩) :
  advance_year year_2017 32 = ⟨HeavenlyStem.Ji, EarthlyBranch.Si⟩ := by
  sorry

end year_2049_is_jisi_l2825_282554


namespace smallest_class_number_l2825_282583

theorem smallest_class_number (total_classes : Nat) (selected_classes : Nat) (sum_selected : Nat) : 
  total_classes = 24 →
  selected_classes = 4 →
  sum_selected = 48 →
  ∃ x : Nat, 
    x > 0 ∧ 
    x ≤ total_classes ∧
    x + (x + (total_classes / selected_classes)) + 
    (x + 2 * (total_classes / selected_classes)) + 
    (x + 3 * (total_classes / selected_classes)) = sum_selected ∧
    x = 3 := by
  sorry

end smallest_class_number_l2825_282583


namespace marble_fraction_after_tripling_l2825_282569

theorem marble_fraction_after_tripling (total : ℝ) (h_total_pos : total > 0) :
  let initial_blue := (2/3) * total
  let initial_red := total - initial_blue
  let new_red := 3 * initial_red
  let new_total := initial_blue + new_red
  new_red / new_total = 3/5 := by
sorry

end marble_fraction_after_tripling_l2825_282569


namespace predictor_accuracy_two_thirds_l2825_282531

/-- Represents a match between two teams -/
structure Match where
  team_a_win_prob : ℝ
  team_b_win_prob : ℝ
  (prob_sum_one : team_a_win_prob + team_b_win_prob = 1)

/-- Represents a predictor who chooses winners with the same probability as the team's chance of winning -/
def predictor_correct_prob (m : Match) : ℝ :=
  m.team_a_win_prob * m.team_a_win_prob + m.team_b_win_prob * m.team_b_win_prob

/-- Theorem stating that for a match where one team has 2/3 probability of winning,
    the probability of the predictor correctly choosing the winner is 5/9 -/
theorem predictor_accuracy_two_thirds :
  ∀ m : Match, m.team_a_win_prob = 2/3 → predictor_correct_prob m = 5/9 := by
  sorry

end predictor_accuracy_two_thirds_l2825_282531


namespace set_equality_proof_all_sets_satisfying_condition_l2825_282579

def solution_set : Set (Set Nat) :=
  {{3}, {1, 3}, {2, 3}, {1, 2, 3}}

theorem set_equality_proof (B : Set Nat) :
  ({1, 2} ∪ B = {1, 2, 3}) ↔ (B ∈ solution_set) := by
  sorry

theorem all_sets_satisfying_condition :
  {B : Set Nat | {1, 2} ∪ B = {1, 2, 3}} = solution_set := by
  sorry

end set_equality_proof_all_sets_satisfying_condition_l2825_282579


namespace sqrt_72_div_sqrt_8_minus_abs_neg_2_equals_1_l2825_282507

theorem sqrt_72_div_sqrt_8_minus_abs_neg_2_equals_1 :
  Real.sqrt 72 / Real.sqrt 8 - |(-2)| = 1 := by sorry

end sqrt_72_div_sqrt_8_minus_abs_neg_2_equals_1_l2825_282507


namespace triangle_side_equality_l2825_282536

theorem triangle_side_equality (A B C : Real) (a b c : Real) :
  (0 < A) ∧ (A < π) ∧ (0 < B) ∧ (B < π) ∧ (0 < C) ∧ (C < π) →  -- angles are positive and less than π
  (A + B + C = π) →  -- sum of angles in a triangle
  (a > 0) ∧ (b > 0) ∧ (c > 0) →  -- sides are positive
  (a / Real.sin A = b / Real.sin B) →  -- Law of Sines
  (a / Real.sin A = c / Real.sin C) →  -- Law of Sines
  (3 * b * Real.cos C + 3 * c * Real.cos B = a^2) →  -- given condition
  a = 3 := by
sorry

end triangle_side_equality_l2825_282536


namespace range_of_a_l2825_282524

-- Define the function f
def f (x : ℝ) : ℝ := -x^2 + 2*x + 3

-- Define the theorem
theorem range_of_a (a : ℝ) :
  (∀ x ∈ Set.Icc (-2 : ℝ) a, f x ∈ Set.Icc (-5 : ℝ) 4) ∧
  (∃ x₁ ∈ Set.Icc (-2 : ℝ) a, f x₁ = -5) ∧
  (∃ x₂ ∈ Set.Icc (-2 : ℝ) a, f x₂ = 4) →
  a ∈ Set.Icc (1 : ℝ) 4 :=
by sorry

end range_of_a_l2825_282524


namespace limit_of_a_l2825_282508

def a (n : ℕ) : ℚ := (3 * n - 1) / (5 * n + 1)

theorem limit_of_a : ∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, |a n - 3/5| < ε := by sorry

end limit_of_a_l2825_282508


namespace f_negative_a_eq_zero_l2825_282518

noncomputable def f (x : ℝ) : ℝ := x^3 + Real.sin x + 1

theorem f_negative_a_eq_zero (a : ℝ) (h : f a = 2) : f (-a) = 0 := by
  sorry

end f_negative_a_eq_zero_l2825_282518


namespace unique_complex_solution_l2825_282576

theorem unique_complex_solution :
  ∃! z : ℂ, Complex.abs z < 20 ∧ Complex.exp z = 1 - z / 2 := by
  sorry

end unique_complex_solution_l2825_282576


namespace volume_sphere_minus_cylinder_l2825_282595

/-- The volume of space inside a sphere and outside an inscribed right cylinder -/
theorem volume_sphere_minus_cylinder (R : ℝ) (r : ℝ) (h : ℝ) :
  R = 7 →
  r = 4 →
  h = 2 * Real.sqrt 33 →
  (4 / 3 * π * R^3 - π * r^2 * h) = ((1372 / 3 : ℝ) - 32 * Real.sqrt 33) * π :=
by sorry

end volume_sphere_minus_cylinder_l2825_282595


namespace parallelogram_not_symmetrical_l2825_282562

-- Define a type for shapes
inductive Shape
  | Circle
  | Rectangle
  | IsoscelesTrapezoid
  | Parallelogram

-- Define a property for symmetry
def is_symmetrical (s : Shape) : Prop :=
  match s with
  | Shape.Circle => True
  | Shape.Rectangle => True
  | Shape.IsoscelesTrapezoid => True
  | Shape.Parallelogram => False

-- Theorem statement
theorem parallelogram_not_symmetrical :
  ∃ (s : Shape), ¬(is_symmetrical s) ∧ s = Shape.Parallelogram :=
sorry

end parallelogram_not_symmetrical_l2825_282562


namespace chessboard_probability_l2825_282598

theorem chessboard_probability (k : ℕ) : k ≥ 5 →
  (((k - 4)^2 - 1) / (2 * (k - 4)^2 : ℚ) = 48 / 100) ↔ k = 9 := by
  sorry

end chessboard_probability_l2825_282598


namespace roof_dimension_difference_l2825_282593

theorem roof_dimension_difference (area : ℝ) (length_width_ratio : ℝ) :
  area = 676 ∧ length_width_ratio = 4 →
  ∃ (length width : ℝ),
    length = length_width_ratio * width ∧
    area = length * width ∧
    length - width = 39 :=
by sorry

end roof_dimension_difference_l2825_282593


namespace tan_non_intersection_l2825_282596

theorem tan_non_intersection :
  ∀ y : ℝ, ∃ k : ℤ, (2 * (π/8) + π/4) = k * π + π/2 :=
by sorry

end tan_non_intersection_l2825_282596


namespace unique_cube_prime_factor_l2825_282523

def greatest_prime_factor (n : ℕ) : ℕ := sorry

theorem unique_cube_prime_factor : 
  ∃! n : ℕ, n > 1 ∧ 
    (greatest_prime_factor n = n^(1/3)) ∧ 
    (greatest_prime_factor (n + 200) = (n + 200)^(1/3)) := by
  sorry

end unique_cube_prime_factor_l2825_282523
