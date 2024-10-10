import Mathlib

namespace range_of_f_l3727_372762

def f (x : ℝ) : ℝ := (x - 2)^2 - 1

theorem range_of_f :
  ∀ x ∈ Set.Icc (-1 : ℝ) 3,
  ∃ y ∈ Set.Icc (-1 : ℝ) 8,
  f x = y ∧
  ∀ z, f x = z → z ∈ Set.Icc (-1 : ℝ) 8 :=
sorry

end range_of_f_l3727_372762


namespace grocery_value_proof_l3727_372716

def car_cost : ℝ := 14600
def initial_savings : ℝ := 14500
def trips : ℕ := 40
def fixed_charge : ℝ := 1.5
def grocery_charge_rate : ℝ := 0.05

theorem grocery_value_proof (grocery_value : ℝ) : 
  car_cost - initial_savings = trips * fixed_charge + grocery_charge_rate * grocery_value →
  grocery_value = 800 := by
  sorry

end grocery_value_proof_l3727_372716


namespace jeremy_stroll_distance_l3727_372782

theorem jeremy_stroll_distance (speed : ℝ) (time : ℝ) (distance : ℝ) : 
  speed = 2 → time = 10 → distance = speed * time → distance = 20 := by
  sorry

end jeremy_stroll_distance_l3727_372782


namespace circle_polar_to_cartesian_l3727_372777

/-- Given a circle with polar equation ρ = 2cos θ, its Cartesian equation is (x-1)^2 + y^2 = 1 -/
theorem circle_polar_to_cartesian :
  ∀ (x y ρ θ : ℝ),
  (ρ = 2 * Real.cos θ) →
  (x = ρ * Real.cos θ) →
  (y = ρ * Real.sin θ) →
  ((x - 1)^2 + y^2 = 1) :=
by sorry

end circle_polar_to_cartesian_l3727_372777


namespace value_of_expression_l3727_372700

theorem value_of_expression (x y : ℝ) 
  (h1 : x^2 + x*y = 3) 
  (h2 : x*y + y^2 = -2) : 
  2*x^2 - x*y - 3*y^2 = 12 := by
sorry

end value_of_expression_l3727_372700


namespace even_function_negative_domain_l3727_372746

/-- A function f: ℝ → ℝ is even if f(-x) = f(x) for all x ∈ ℝ -/
def EvenFunction (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

theorem even_function_negative_domain
  (f : ℝ → ℝ)
  (h_even : EvenFunction f)
  (h_positive : ∀ x ≥ 0, f x = 2 * x + 1) :
  ∀ x < 0, f x = -2 * x + 1 := by
  sorry

end even_function_negative_domain_l3727_372746


namespace jake_second_test_difference_l3727_372775

def jake_test_scores (test1 test2 test3 test4 : ℕ) : Prop :=
  test1 = 80 ∧ 
  test3 = 65 ∧ 
  test3 = test4 ∧ 
  (test1 + test2 + test3 + test4) / 4 = 75

theorem jake_second_test_difference :
  ∀ test1 test2 test3 test4 : ℕ,
    jake_test_scores test1 test2 test3 test4 →
    test2 - test1 = 10 := by
  sorry

end jake_second_test_difference_l3727_372775


namespace quadratic_function_properties_l3727_372721

-- Define the quadratic function f
def f (x : ℝ) := 2 * x^2 - 4 * x + 3

-- State the theorem
theorem quadratic_function_properties :
  (f 1 = 1) ∧
  (∀ x, f (x + 1) - f x = 4 * x - 2) ∧
  (∀ a, (∃ x y, 2 * a ≤ x ∧ x < y ∧ y ≤ a + 1 ∧ f x > f y ∧ ∃ z, x < z ∧ z < y ∧ f z > f x)
    ↔ (0 < a ∧ a < 1/2)) :=
by sorry

end quadratic_function_properties_l3727_372721


namespace beatrice_tv_shopping_l3727_372715

theorem beatrice_tv_shopping (x : ℕ) 
  (h1 : x > 0)  -- Beatrice looked at some TVs in the first store
  (h2 : 42 = x + 3*x + 10) : -- Total TVs = First store + Online store + Auction site
  x = 8 := by
sorry

end beatrice_tv_shopping_l3727_372715


namespace output_increase_l3727_372797

theorem output_increase (production_increase : Real) (hours_decrease : Real) : 
  production_increase = 0.8 →
  hours_decrease = 0.1 →
  ((1 + production_increase) / (1 - hours_decrease) - 1) * 100 = 100 := by
sorry

end output_increase_l3727_372797


namespace triangle_max_area_l3727_372736

theorem triangle_max_area (a b c : ℝ) (A : ℝ) (h_a : a = 4) (h_A : A = π/3) :
  (∃ (S : ℝ), S = (1/2) * b * c * Real.sin A ∧ 
    ∀ (S' : ℝ), S' = (1/2) * b * c * Real.sin A → S' ≤ 4 * Real.sqrt 3) :=
by sorry

end triangle_max_area_l3727_372736


namespace newspaper_conference_max_overlap_l3727_372778

theorem newspaper_conference_max_overlap (total : ℕ) (writers : ℕ) (editors : ℕ) 
  (h_total : total = 100)
  (h_writers : writers = 45)
  (h_editors : editors > 36)
  (x : ℕ)
  (h_both : x = writers + editors - total + (total - writers - editors) / 2) :
  x ≤ 18 ∧ ∃ (e : ℕ), e > 36 ∧ x = writers + e - total + (total - writers - e) / 2 ∧ x = 18 :=
by sorry

end newspaper_conference_max_overlap_l3727_372778


namespace problem_1_problem_2_l3727_372757

-- Problem 1
theorem problem_1 : Real.sqrt 27 - 6 * Real.sqrt (1/3) + Real.sqrt ((-2)^2) = Real.sqrt 3 + 2 := by
  sorry

-- Problem 2
theorem problem_2 (x y : ℝ) (hx : x = 2 + Real.sqrt 3) (hy : y = 2 - Real.sqrt 3) :
  x^2 * y + x * y^2 = 4 := by
  sorry

end problem_1_problem_2_l3727_372757


namespace binomial_30_choose_3_l3727_372770

theorem binomial_30_choose_3 : Nat.choose 30 3 = 4060 := by
  sorry

end binomial_30_choose_3_l3727_372770


namespace rectangle_area_stage_8_l3727_372738

/-- The area of a rectangle formed by adding n squares of side length s --/
def rectangleArea (n : ℕ) (s : ℝ) : ℝ := n * (s * s)

/-- Theorem: The area of a rectangle formed by adding 8 squares, each 4 inches by 4 inches, is 128 square inches --/
theorem rectangle_area_stage_8 : rectangleArea 8 4 = 128 := by
  sorry

end rectangle_area_stage_8_l3727_372738


namespace seven_zero_three_six_repeating_equals_fraction_l3727_372731

/-- Represents a repeating decimal with an integer part and a repeating fractional part -/
structure RepeatingDecimal where
  integerPart : Int
  repeatingPart : Nat
  repeatingLength : Nat

/-- The value of 7.036̄ as a RepeatingDecimal -/
def seven_zero_three_six_repeating : RepeatingDecimal :=
  { integerPart := 7
  , repeatingPart := 36
  , repeatingLength := 3 }

/-- Converts a RepeatingDecimal to a rational number -/
def toRational (d : RepeatingDecimal) : Rat :=
  sorry

theorem seven_zero_three_six_repeating_equals_fraction :
  toRational seven_zero_three_six_repeating = 781 / 111 := by
  sorry

end seven_zero_three_six_repeating_equals_fraction_l3727_372731


namespace classroom_puzzle_l3727_372791

theorem classroom_puzzle (initial_boys initial_girls : ℕ) : 
  initial_boys = initial_girls →
  initial_boys = 2 * (initial_girls - 8) →
  initial_boys + initial_girls = 32 := by
sorry

end classroom_puzzle_l3727_372791


namespace mechanics_total_charge_l3727_372755

/-- Calculates the total amount charged by two mechanics working on a car. -/
theorem mechanics_total_charge
  (hours1 : ℕ)  -- Hours worked by the first mechanic
  (hours2 : ℕ)  -- Hours worked by the second mechanic
  (rate : ℕ)    -- Combined hourly rate in dollars
  (h1 : hours1 = 10)  -- First mechanic worked for 10 hours
  (h2 : hours2 = 5)   -- Second mechanic worked for 5 hours
  (h3 : rate = 160)   -- Combined hourly rate is $160
  : (hours1 + hours2) * rate = 2400 := by
  sorry


end mechanics_total_charge_l3727_372755


namespace rectangle_area_with_inscribed_circle_l3727_372710

theorem rectangle_area_with_inscribed_circle (r : ℝ) (ratio : ℝ) : 
  r = 7 → ratio = 3 → 2 * r * ratio * 2 * r = 588 := by
  sorry

end rectangle_area_with_inscribed_circle_l3727_372710


namespace min_value_expression_l3727_372744

/-- Given two positive real numbers m and n, and two vectors a and b that are perpendicular,
    prove that the minimum value of 1/m + 2/n is 3 + 2√2 -/
theorem min_value_expression (m n : ℝ) (a b : ℝ × ℝ) 
  (hm : m > 0) (hn : n > 0)
  (ha : a = (m, 1)) (hb : b = (1, n - 1))
  (h_perp : a.1 * b.1 + a.2 * b.2 = 0) :
  (∀ x y, x > 0 → y > 0 → 1/x + 2/y ≥ 1/m + 2/n) → 1/m + 2/n = 3 + 2 * Real.sqrt 2 :=
sorry

end min_value_expression_l3727_372744


namespace solution_set_min_value_equality_condition_l3727_372704

-- Define the function f(x)
def f (x : ℝ) : ℝ := 2 * abs (x - 1) + abs (x + 2)

-- Part 1: Solution set of f(x) ≤ 9
theorem solution_set (x : ℝ) : f x ≤ 9 ↔ x ∈ Set.Icc (-3) 3 := by sorry

-- Part 2: Minimum value of 4a² + b² + c²
theorem min_value (a b c : ℝ) (h : a + b + c = 3) :
  4 * a^2 + b^2 + c^2 ≥ 4 := by sorry

-- Equality condition
theorem equality_condition (a b c : ℝ) (h : a + b + c = 3) :
  4 * a^2 + b^2 + c^2 = 4 ↔ a = 1/3 ∧ b = 4/3 ∧ c = 4/3 := by sorry

end solution_set_min_value_equality_condition_l3727_372704


namespace chord_length_l3727_372727

theorem chord_length (r d : ℝ) (hr : r = 5) (hd : d = 4) :
  let chord_length := 2 * Real.sqrt (r^2 - d^2)
  chord_length = 6 := by sorry

end chord_length_l3727_372727


namespace contrapositive_equivalence_l3727_372799

theorem contrapositive_equivalence (a b : ℝ) : 
  (¬(a = 0 ∧ b = 0) → a^2 + b^2 ≠ 0) ↔ (a^2 + b^2 = 0 → a = 0 ∧ b = 0) :=
by sorry

end contrapositive_equivalence_l3727_372799


namespace arctan_of_tan_difference_l3727_372729

theorem arctan_of_tan_difference (θ : Real) : 
  θ ∈ Set.Icc 0 180 → 
  Real.arctan (Real.tan (80 * π / 180) - 3 * Real.tan (30 * π / 180)) = 50 * π / 180 := by
  sorry

end arctan_of_tan_difference_l3727_372729


namespace square_overlap_area_l3727_372726

/-- The area of overlapping regions in a rectangle with four squares -/
theorem square_overlap_area (total_square_area sum_individual_areas uncovered_area : ℝ) :
  total_square_area = 27.5 ∧ 
  sum_individual_areas = 30 ∧ 
  uncovered_area = 1.5 →
  sum_individual_areas - total_square_area + uncovered_area = 4 := by
  sorry

end square_overlap_area_l3727_372726


namespace parallelogram_area_example_l3727_372740

/-- Represents a parallelogram with given base and height -/
structure Parallelogram where
  base : ℝ
  height : ℝ

/-- Calculates the area of a parallelogram -/
def area (p : Parallelogram) : ℝ := p.base * p.height

/-- Theorem: The area of a parallelogram with base 22 cm and height 14 cm is 308 square centimeters -/
theorem parallelogram_area_example : 
  let p : Parallelogram := { base := 22, height := 14 }
  area p = 308 := by sorry

end parallelogram_area_example_l3727_372740


namespace price_reduction_percentage_l3727_372783

theorem price_reduction_percentage (initial_price final_price : ℝ) 
  (h1 : initial_price = 2000)
  (h2 : final_price = 1620)
  (h3 : initial_price > 0)
  (h4 : final_price > 0)
  (h5 : final_price < initial_price) :
  ∃ (x : ℝ), x > 0 ∧ x < 1 ∧ initial_price * (1 - x)^2 = final_price ∧ x = 0.1 := by
  sorry

end price_reduction_percentage_l3727_372783


namespace polynomial_simplification_l3727_372769

theorem polynomial_simplification (x : ℝ) :
  2 * x^2 * (4 * x^3 - 3 * x + 1) - 7 * (x^3 - 3 * x^2 + 2 * x - 8) =
  8 * x^5 - 13 * x^3 + 23 * x^2 - 14 * x + 56 := by
  sorry

end polynomial_simplification_l3727_372769


namespace reading_homework_pages_l3727_372772

theorem reading_homework_pages (total_pages math_pages : ℕ) 
  (h1 : total_pages = 7) 
  (h2 : math_pages = 5) : 
  total_pages - math_pages = 2 := by
sorry

end reading_homework_pages_l3727_372772


namespace hyperbola_sum_l3727_372737

/-- Represents a hyperbola with center (h, k), focus (h + c, k), and vertex (h + a, k) --/
structure Hyperbola where
  h : ℝ
  k : ℝ
  a : ℝ
  c : ℝ

/-- The theorem that for a specific hyperbola, h + k + a + b = 7 --/
theorem hyperbola_sum (H : Hyperbola) (h_center : H.h = 1) (k_center : H.k = -3)
  (h_vertex : H.h + H.a = 4) (h_focus : H.h + H.c = 1 + 3 * Real.sqrt 5) :
  H.h + H.k + H.a + Real.sqrt (H.c^2 - H.a^2) = 7 := by
  sorry

end hyperbola_sum_l3727_372737


namespace parallel_vectors_k_value_l3727_372781

/-- Given two vectors a and b in ℝ², prove that if 2a + b is parallel to (1/2)a + kb, then k = 1/4. -/
theorem parallel_vectors_k_value (a b : ℝ × ℝ) (k : ℝ) 
    (h1 : a = (2, 1)) 
    (h2 : b = (1, 2)) 
    (h_parallel : ∃ (t : ℝ), t • (2 • a + b) = (1/2 • a + k • b)) : 
  k = 1/4 := by
sorry

end parallel_vectors_k_value_l3727_372781


namespace masking_tape_wall_width_l3727_372750

theorem masking_tape_wall_width (total_tape : ℝ) (known_wall_width : ℝ) (known_wall_count : ℕ) (unknown_wall_count : ℕ) :
  total_tape = 20 →
  known_wall_width = 6 →
  known_wall_count = 2 →
  unknown_wall_count = 2 →
  (unknown_wall_count : ℝ) * (total_tape - known_wall_count * known_wall_width) / unknown_wall_count = 4 := by
sorry

end masking_tape_wall_width_l3727_372750


namespace smallest_absolute_value_of_z_l3727_372724

theorem smallest_absolute_value_of_z (z : ℂ) (h : Complex.abs (z - 15) + Complex.abs (z - Complex.I * 7) = 17) :
  ∃ (w : ℂ), Complex.abs (z - 15) + Complex.abs (z - Complex.I * 7) = 17 ∧ Complex.abs w ≤ Complex.abs z ∧ Complex.abs w = 105 / 17 :=
sorry

end smallest_absolute_value_of_z_l3727_372724


namespace complex_sum_equals_z_l3727_372732

theorem complex_sum_equals_z (z : ℂ) (h : z^2 - z + 1 = 0) : 
  z^107 + z^108 + z^109 + z^110 + z^111 = z := by
  sorry

end complex_sum_equals_z_l3727_372732


namespace min_questions_required_l3727_372713

/-- Represents the color of a ball -/
inductive Color
| White
| Black

/-- Represents a box containing a ball -/
structure Box where
  ball : Color

/-- Represents the state of the boxes -/
structure BoxState where
  boxes : Vector Box 2004
  white_count : Nat
  white_count_even : Even white_count

/-- Represents a question about two boxes -/
structure Question where
  box1 : Fin 2004
  box2 : Fin 2004
  box1_ne_box2 : box1 ≠ box2

/-- The result of asking a question -/
def ask_question (state : BoxState) (q : Question) : Bool :=
  match state.boxes[q.box1].ball, state.boxes[q.box2].ball with
  | Color.White, _ => true
  | _, Color.White => true
  | _, _ => false

/-- A strategy for asking questions -/
def Strategy := Nat → Question

/-- Checks if a strategy is successful for a given state -/
def strategy_successful (state : BoxState) (strategy : Strategy) : Prop :=
  ∃ n : Nat, ∃ i j : Fin 2004,
    i ≠ j ∧
    state.boxes[i].ball = Color.White ∧
    state.boxes[j].ball = Color.White ∧
    (∀ k < n, ask_question state (strategy k) = true)

/-- The main theorem stating the minimum number of questions required -/
theorem min_questions_required :
  ∀ (strategy : Strategy),
  (∀ state : BoxState, strategy_successful state strategy) →
  (∃ n : Nat, ∀ k, strategy k = strategy n → k ≥ 4005) :=
sorry

end min_questions_required_l3727_372713


namespace circle_diameter_relation_l3727_372776

theorem circle_diameter_relation (R S : Real) (h : R > 0 ∧ S > 0) :
  (R * R) / (S * S) = 0.16 → R / S = 0.4 := by
  sorry

end circle_diameter_relation_l3727_372776


namespace cell_plan_comparison_l3727_372795

/-- Represents a cell phone plan with a flat fee and per-minute rate -/
structure CellPlan where
  flatFee : ℕ  -- Flat fee in cents
  perMinRate : ℕ  -- Per-minute rate in cents
  
/-- Calculates the cost of a plan for a given number of minutes -/
def planCost (plan : CellPlan) (minutes : ℕ) : ℕ :=
  plan.flatFee + plan.perMinRate * minutes

/-- The three cell phone plans -/
def planX : CellPlan := { flatFee := 0, perMinRate := 15 }
def planY : CellPlan := { flatFee := 2500, perMinRate := 7 }
def planZ : CellPlan := { flatFee := 3000, perMinRate := 6 }

theorem cell_plan_comparison :
  (∀ m : ℕ, m < 313 → planCost planX m ≤ planCost planY m) ∧
  (planCost planY 313 < planCost planX 313) ∧
  (∀ m : ℕ, m < 334 → planCost planX m ≤ planCost planZ m) ∧
  (planCost planZ 334 < planCost planX 334) :=
by sorry


end cell_plan_comparison_l3727_372795


namespace sum_of_cubes_equation_l3727_372779

theorem sum_of_cubes_equation (x y : ℝ) :
  x^3 + 21*x*y + y^3 = 343 → x + y = 7 ∨ x + y = -14 := by
  sorry

end sum_of_cubes_equation_l3727_372779


namespace max_product_sum_l3727_372793

theorem max_product_sum (f g h j : ℕ) : 
  f ∈ ({7, 8, 9, 10} : Set ℕ) →
  g ∈ ({7, 8, 9, 10} : Set ℕ) →
  h ∈ ({7, 8, 9, 10} : Set ℕ) →
  j ∈ ({7, 8, 9, 10} : Set ℕ) →
  f ≠ g → f ≠ h → f ≠ j → g ≠ h → g ≠ j → h ≠ j →
  (f * g + g * h + h * j + f * j) ≤ 289 :=
by sorry

end max_product_sum_l3727_372793


namespace smallest_fourth_number_l3727_372774

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n < 100

def digits_sum (n : ℕ) : ℕ :=
  let d₁ := n / 10
  let d₂ := n % 10
  d₁ + d₂

def strictly_increasing_digits (n : ℕ) : Prop :=
  let d₁ := n / 10
  let d₂ := n % 10
  d₁ < d₂

theorem smallest_fourth_number :
  ∃ (n : ℕ),
    is_two_digit n ∧
    strictly_increasing_digits n ∧
    (∀ m, is_two_digit m → strictly_increasing_digits m →
      digits_sum 34 + digits_sum 18 + digits_sum 73 + digits_sum n +
      digits_sum m = (34 + 18 + 73 + n + m) / 6 →
      n ≤ m) ∧
    digits_sum 34 + digits_sum 18 + digits_sum 73 + digits_sum n =
      (34 + 18 + 73 + n) / 6 ∧
    n = 29 :=
by sorry

end smallest_fourth_number_l3727_372774


namespace sqrt_26_is_7th_term_l3727_372773

theorem sqrt_26_is_7th_term (a : ℕ → ℝ) :
  (∀ n : ℕ, n > 0 → a n = Real.sqrt (4 * n - 2)) →
  a 7 = Real.sqrt 26 :=
by
  sorry

end sqrt_26_is_7th_term_l3727_372773


namespace book_pages_and_reading_schedule_l3727_372761

-- Define the total number of pages in the book
variable (P : ℕ)

-- Define the number of pages read on the 4th day
variable (x : ℕ)

-- Theorem statement
theorem book_pages_and_reading_schedule :
  -- Conditions
  (2 / 3 : ℚ) * P = ((2 / 3 : ℚ) * P - (1 / 3 : ℚ) * P) + 90 ∧
  (1 / 3 : ℚ) * P = x + (x - 10) ∧
  x > 10 →
  -- Conclusions
  P = 270 ∧ x = 50 ∧ x - 10 = 40 := by
sorry

end book_pages_and_reading_schedule_l3727_372761


namespace monday_pages_proof_l3727_372787

def total_pages : ℕ := 158
def tuesday_pages : ℕ := 38
def wednesday_pages : ℕ := 61
def thursday_pages : ℕ := 12
def friday_pages : ℕ := 2 * thursday_pages

theorem monday_pages_proof :
  total_pages - (tuesday_pages + wednesday_pages + thursday_pages + friday_pages) = 23 := by
  sorry

end monday_pages_proof_l3727_372787


namespace conference_handshakes_l3727_372786

theorem conference_handshakes (n : ℕ) (h : n = 30) :
  (n * (n - 1)) / 2 = 435 := by
  sorry

end conference_handshakes_l3727_372786


namespace ceiling_square_count_l3727_372766

theorem ceiling_square_count (x : ℝ) (h : ⌈x⌉ = 15) : 
  ∃ (S : Finset ℕ), (∀ n ∈ S, ∃ y : ℝ, ⌈y⌉ = 15 ∧ ⌈y^2⌉ = n) ∧ S.card = 29 :=
sorry

end ceiling_square_count_l3727_372766


namespace intersection_A_B_intersection_complement_A_B_intersection_A_complement_B_l3727_372708

-- Define the universal set U as ℝ
def U := ℝ

-- Define set A
def A : Set ℝ := {x | -1 < x ∧ x < 3}

-- Define set B
def B : Set ℝ := {x | 0 < x ∧ x < 5}

-- Theorem for A ∩ B
theorem intersection_A_B : A ∩ B = {x | 0 < x ∧ x < 3} := by sorry

-- Theorem for (∁ᵤA) ∩ B
theorem intersection_complement_A_B : (Aᶜ : Set ℝ) ∩ B = {x | 3 ≤ x ∧ x < 5} := by sorry

-- Theorem for A ∩ (∁ᵤB)
theorem intersection_A_complement_B : A ∩ (Bᶜ : Set ℝ) = {x | -1 < x ∧ x ≤ 0} := by sorry

end intersection_A_B_intersection_complement_A_B_intersection_A_complement_B_l3727_372708


namespace mixed_number_calculation_l3727_372768

theorem mixed_number_calculation : 
  23 * ((1 + 2/3) + (2 + 1/4)) / ((1 + 1/2) + (1 + 1/5)) = 3 + 43/108 := by
  sorry

end mixed_number_calculation_l3727_372768


namespace min_value_sum_reciprocals_l3727_372749

theorem min_value_sum_reciprocals (a b c : ℝ) 
  (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) 
  (h4 : a + b + c = 3) : 
  ∀ x y z, x > 0 → y > 0 → z > 0 → x + y + z = 3 → 
  1 / (a + b) + 1 / c ≤ 1 / (x + y) + 1 / z :=
by sorry

end min_value_sum_reciprocals_l3727_372749


namespace number_division_problem_l3727_372703

theorem number_division_problem : ∃ x : ℝ, (x / 5 = 70 + x / 6) ∧ x = 2100 := by
  sorry

end number_division_problem_l3727_372703


namespace specific_child_group_size_l3727_372733

/-- Represents a group of children with specific age characteristics -/
structure ChildGroup where
  sum_of_ages : ℕ
  age_difference : ℕ
  eldest_age : ℕ

/-- Calculates the number of children in a ChildGroup -/
def number_of_children (group : ChildGroup) : ℕ :=
  sorry

/-- Theorem stating that for a specific ChildGroup, the number of children is 10 -/
theorem specific_child_group_size :
  let group : ChildGroup := {
    sum_of_ages := 50,
    age_difference := 2,
    eldest_age := 14
  }
  number_of_children group = 10 := by
  sorry

end specific_child_group_size_l3727_372733


namespace square_area_to_cube_volume_ratio_l3727_372784

theorem square_area_to_cube_volume_ratio 
  (cube : Real → Real) 
  (square : Real → Real) 
  (h : ∀ s : Real, s > 0 → s * Real.sqrt 3 = 4 * square s) :
  ∀ s : Real, s > 0 → (square s)^2 / (cube s) = 3/16 := by
  sorry

end square_area_to_cube_volume_ratio_l3727_372784


namespace construct_from_blocks_l3727_372717

/-- A building block consists of 7 unit cubes in a 2x2x2 shape with one corner unit cube missing. -/
structure BuildingBlock :=
  (size : Nat)
  (unit_cubes : Nat)

/-- Definition of our specific building block -/
def specific_block : BuildingBlock :=
  { size := 2,
    unit_cubes := 7 }

/-- A cube with one unit removed -/
structure CubeWithUnitRemoved :=
  (edge_length : Nat)
  (total_units : Nat)

/-- Function to check if a cube with a unit removed can be constructed from building blocks -/
def can_construct (c : CubeWithUnitRemoved) (b : BuildingBlock) : Prop :=
  ∃ (num_blocks : Nat), c.total_units = num_blocks * b.unit_cubes

/-- Main theorem -/
theorem construct_from_blocks (n : Nat) (h : n ≥ 2) :
  let c := CubeWithUnitRemoved.mk (2^n) ((2^n)^3 - 1)
  can_construct c specific_block :=
by sorry

end construct_from_blocks_l3727_372717


namespace max_type_c_tubes_exists_solution_with_73_type_c_l3727_372764

/-- Represents the types of test tubes -/
inductive TubeType
  | A
  | B
  | C

/-- Represents a solution of test tubes -/
structure Solution where
  a : ℕ  -- number of type A tubes
  b : ℕ  -- number of type B tubes
  c : ℕ  -- number of type C tubes

/-- The concentration of the solution in each type of tube -/
def concentration : TubeType → ℚ
  | TubeType.A => 1/10
  | TubeType.B => 1/5
  | TubeType.C => 9/10

/-- The total number of tubes used -/
def Solution.total (s : Solution) : ℕ := s.a + s.b + s.c

/-- The average concentration of the final solution -/
def Solution.averageConcentration (s : Solution) : ℚ :=
  (s.a * concentration TubeType.A + s.b * concentration TubeType.B + s.c * concentration TubeType.C) / s.total

/-- Predicate to check if the solution satisfies the conditions -/
def Solution.isValid (s : Solution) : Prop :=
  s.averageConcentration = 20.17/100 ∧
  s.total ≥ 3 ∧
  s.a > 0 ∧ s.b > 0 ∧ s.c > 0

theorem max_type_c_tubes (s : Solution) (h : s.isValid) :
  s.c ≤ 73 :=
sorry

theorem exists_solution_with_73_type_c :
  ∃ s : Solution, s.isValid ∧ s.c = 73 :=
sorry

end max_type_c_tubes_exists_solution_with_73_type_c_l3727_372764


namespace income_comparison_l3727_372759

/-- Represents the problem of calculating incomes relative to Juan's base income -/
theorem income_comparison (J : ℝ) (J_pos : J > 0) : 
  let tim_base := 0.7 * J
  let mary_total := 1.12 * J * 1.1
  let lisa_base := 0.63 * J
  let lisa_total := lisa_base * 1.03
  let alan_base := lisa_base / 1.15
  let nina_base := 1.25 * J
  let nina_total := nina_base * 1.07
  (mary_total + lisa_total + nina_total) / J = 3.2184 := by
sorry

end income_comparison_l3727_372759


namespace proposition_p_and_not_q_l3727_372741

theorem proposition_p_and_not_q :
  (∃ x : ℝ, x - 2 > Real.log x) ∧ ¬(∀ x : ℝ, Real.sin x < x) := by sorry

end proposition_p_and_not_q_l3727_372741


namespace remainder_of_product_with_modular_inverse_l3727_372754

theorem remainder_of_product_with_modular_inverse (n a b : ℤ) : 
  n > 0 → (a * b) % n = 1 % n → (a * b) % n = 1 :=
by sorry

end remainder_of_product_with_modular_inverse_l3727_372754


namespace remaining_average_l3727_372785

theorem remaining_average (total : ℝ) (group1 : ℝ) (group2 : ℝ) :
  total = 6 * 2.8 ∧ group1 = 2 * 2.4 ∧ group2 = 2 * 2.3 →
  (total - group1 - group2) / 2 = 3.7 := by
sorry

end remaining_average_l3727_372785


namespace class_item_distribution_l3727_372758

/-- Calculates the total number of items distributed in a class --/
def calculate_total_items (num_children : ℕ) 
                          (initial_pencils : ℕ) 
                          (initial_erasers : ℕ) 
                          (initial_crayons : ℕ) 
                          (extra_pencils : ℕ) 
                          (extra_crayons : ℕ) 
                          (extra_erasers : ℕ) 
                          (num_children_extra_pencils_crayons : ℕ) : ℕ × ℕ × ℕ :=
  let total_pencils := num_children * initial_pencils + num_children_extra_pencils_crayons * extra_pencils
  let total_erasers := num_children * initial_erasers + (num_children - num_children_extra_pencils_crayons) * extra_erasers
  let total_crayons := num_children * initial_crayons + num_children_extra_pencils_crayons * extra_crayons
  (total_pencils, total_erasers, total_crayons)

theorem class_item_distribution :
  let num_children : ℕ := 18
  let initial_pencils : ℕ := 6
  let initial_erasers : ℕ := 3
  let initial_crayons : ℕ := 12
  let extra_pencils : ℕ := 5
  let extra_crayons : ℕ := 8
  let extra_erasers : ℕ := 2
  let num_children_extra_pencils_crayons : ℕ := 10
  
  calculate_total_items num_children initial_pencils initial_erasers initial_crayons
                        extra_pencils extra_crayons extra_erasers
                        num_children_extra_pencils_crayons = (158, 70, 296) := by
  sorry

end class_item_distribution_l3727_372758


namespace cubic_roots_sum_of_cubes_l3727_372760

theorem cubic_roots_sum_of_cubes (a b c : ℂ) : 
  (a^3 - 2*a^2 + 3*a - 4 = 0) → 
  (b^3 - 2*b^2 + 3*b - 4 = 0) → 
  (c^3 - 2*c^2 + 3*c - 4 = 0) → 
  a^3 + b^3 + c^3 = 2 := by
sorry

end cubic_roots_sum_of_cubes_l3727_372760


namespace school_trip_photos_l3727_372707

theorem school_trip_photos (claire_photos : ℕ) (lisa_photos : ℕ) (robert_photos : ℕ) :
  claire_photos = 10 →
  lisa_photos = 3 * claire_photos →
  robert_photos = claire_photos + 20 →
  lisa_photos + robert_photos = 60 := by
  sorry

end school_trip_photos_l3727_372707


namespace coefficient_of_x_l3727_372771

theorem coefficient_of_x (x : ℝ) : 
  let expression := 5*(x - 6) + 3*(9 - 3*x^2 + 2*x) - 10*(3*x - 2)
  ∃ (a b c : ℝ), expression = a*x^2 + (-19)*x + c :=
sorry

end coefficient_of_x_l3727_372771


namespace multiply_37_23_l3727_372742

theorem multiply_37_23 : 37 * 23 = 851 := by
  sorry

end multiply_37_23_l3727_372742


namespace fred_weekend_earnings_l3727_372751

/-- Fred's initial amount of money in dollars -/
def fred_initial : ℕ := 19

/-- Fred's final amount of money in dollars -/
def fred_final : ℕ := 40

/-- Fred's earnings over the weekend in dollars -/
def fred_earnings : ℕ := fred_final - fred_initial

theorem fred_weekend_earnings : fred_earnings = 21 := by
  sorry

end fred_weekend_earnings_l3727_372751


namespace find_defective_box_l3727_372756

/-- Represents the number of boxes -/
def num_boxes : ℕ := 9

/-- Represents the number of standard parts per box -/
def standard_parts_per_box : ℕ := 10

/-- Represents the number of defective parts in one box -/
def defective_parts : ℕ := 10

/-- Represents the weight of a standard part in grams -/
def standard_weight : ℕ := 100

/-- Represents the weight of a defective part in grams -/
def defective_weight : ℕ := 101

/-- Represents the total number of parts selected for weighing -/
def total_selected : ℕ := (num_boxes + 1) * num_boxes / 2

/-- Represents the expected weight if all selected parts were standard -/
def expected_weight : ℕ := total_selected * standard_weight

theorem find_defective_box (actual_weight : ℕ) :
  actual_weight > expected_weight →
  ∃ (box_number : ℕ), 
    box_number ≤ num_boxes ∧
    box_number = actual_weight - expected_weight ∧
    box_number * defective_parts = (defective_weight - standard_weight) * total_selected :=
by sorry

end find_defective_box_l3727_372756


namespace unique_pairs_count_l3727_372709

theorem unique_pairs_count (num_teenagers num_adults : ℕ) : 
  num_teenagers = 12 → num_adults = 8 → 
  (num_teenagers.choose 2) + (num_adults.choose 2) + (num_teenagers * num_adults) = 190 := by
  sorry

end unique_pairs_count_l3727_372709


namespace sales_tax_calculation_l3727_372711

-- Define the total spent
def total_spent : ℝ := 40

-- Define the tax rate
def tax_rate : ℝ := 0.06

-- Define the cost of tax-free items
def tax_free_cost : ℝ := 34.7

-- Theorem to prove
theorem sales_tax_calculation :
  let taxable_cost := total_spent - tax_free_cost
  let sales_tax := taxable_cost * tax_rate / (1 + tax_rate)
  sales_tax = 0.3 := by sorry

end sales_tax_calculation_l3727_372711


namespace carols_rectangle_length_l3727_372796

theorem carols_rectangle_length (carol_width jordan_length jordan_width : ℕ) 
  (h1 : carol_width = 15)
  (h2 : jordan_length = 6)
  (h3 : jordan_width = 30)
  (h4 : carol_width * carol_length = jordan_length * jordan_width) :
  carol_length = 12 := by
  sorry

#check carols_rectangle_length

end carols_rectangle_length_l3727_372796


namespace total_tiles_is_183_l3727_372747

/-- Calculates the number of tiles needed for a room with given dimensions and tile specifications. -/
def calculate_tiles (room_length room_width border_width : ℕ) 
  (border_tile_size inner_tile_size : ℕ) : ℕ :=
  let inner_length := room_length - 2 * border_width
  let inner_width := room_width - 2 * border_width
  let border_tiles := 2 * (room_length + room_width) * (border_width / border_tile_size) +
                      4 * (border_width / border_tile_size) ^ 2
  let inner_tiles := (inner_length * inner_width) / (inner_tile_size ^ 2)
  border_tiles + inner_tiles

/-- Theorem stating that the total number of tiles for the given room specifications is 183. -/
theorem total_tiles_is_183 :
  calculate_tiles 24 18 2 1 3 = 183 := by sorry

end total_tiles_is_183_l3727_372747


namespace second_concert_attendance_l3727_372798

theorem second_concert_attendance 
  (first_concert : ℕ) 
  (additional_attendees : ℕ) 
  (h1 : first_concert = 65899)
  (h2 : additional_attendees = 119) :
  first_concert + additional_attendees = 66018 := by
sorry

end second_concert_attendance_l3727_372798


namespace square_side_length_l3727_372718

theorem square_side_length (area : ℝ) (side : ℝ) :
  area = 1 / 9 ∧ area = side ^ 2 → side = 1 / 3 := by
  sorry

end square_side_length_l3727_372718


namespace unique_a_value_l3727_372748

-- Define the sets A, B, and C
def A (a : ℝ) : Set ℝ := {x | x^2 - a*x + a^2 - 19 = 0}
def B : Set ℝ := {x | x^2 - 5*x + 6 = 0}
def C : Set ℝ := {x | x^2 + 2*x - 8 = 0}

-- State the theorem
theorem unique_a_value : ∃! a : ℝ, (A a ∩ B).Nonempty ∧ (A a ∩ C = ∅) ∧ a = -2 := by
  sorry

end unique_a_value_l3727_372748


namespace remainder_after_adding_2025_l3727_372763

theorem remainder_after_adding_2025 (m : ℤ) : 
  m % 9 = 4 → (m + 2025) % 9 = 4 := by
sorry

end remainder_after_adding_2025_l3727_372763


namespace savings_exceed_500_on_sunday_l3727_372743

/-- The day of the week, starting from Sunday as 0 -/
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- Calculate the total savings after n days -/
def totalSavings (n : ℕ) : ℚ :=
  (3^n - 1) / 2

/-- Convert number of days to day of the week -/
def toDayOfWeek (n : ℕ) : DayOfWeek :=
  match n % 7 with
  | 0 => DayOfWeek.Sunday
  | 1 => DayOfWeek.Monday
  | 2 => DayOfWeek.Tuesday
  | 3 => DayOfWeek.Wednesday
  | 4 => DayOfWeek.Thursday
  | 5 => DayOfWeek.Friday
  | _ => DayOfWeek.Saturday

theorem savings_exceed_500_on_sunday :
  ∃ n : ℕ, totalSavings n > 500 ∧
    ∀ m : ℕ, m < n → totalSavings m ≤ 500 ∧
    toDayOfWeek n = DayOfWeek.Sunday :=
by sorry

end savings_exceed_500_on_sunday_l3727_372743


namespace stationery_cost_is_52_66_l3727_372790

/-- Represents the cost calculation for a set of stationery items -/
def stationery_cost (usd_to_cad_rate : ℝ) : ℝ := by
  -- Define the base costs
  let pencil_cost : ℝ := 2
  let pen_cost : ℝ := pencil_cost + 9
  let notebook_cost : ℝ := 2 * pen_cost

  -- Apply discounts
  let discounted_notebook_cost : ℝ := notebook_cost * 0.85
  let discounted_pen_cost : ℝ := pen_cost * 0.8

  -- Calculate total cost in USD before tax
  let total_usd_before_tax : ℝ := pencil_cost + 2 * discounted_pen_cost + discounted_notebook_cost

  -- Apply tax
  let total_usd_with_tax : ℝ := total_usd_before_tax * 1.1

  -- Convert to CAD
  exact total_usd_with_tax * usd_to_cad_rate

/-- Theorem stating that the total cost of the stationery items is $52.66 CAD -/
theorem stationery_cost_is_52_66 :
  stationery_cost 1.25 = 52.66 := by
  sorry

end stationery_cost_is_52_66_l3727_372790


namespace least_number_divisible_l3727_372780

theorem least_number_divisible (n : ℕ) : n = 861 ↔ 
  (∀ m : ℕ, m < n → ¬(∃ k : ℕ, (m + 3) = 24 * k)) ∧
  (∀ m : ℕ, m < n → ¬(∃ k : ℕ, (m + 3) = 32 * k)) ∧
  (∀ m : ℕ, m < n → ¬(∃ k : ℕ, (m + 3) = 36 * k)) ∧
  (∀ m : ℕ, m < n → ¬(∃ k : ℕ, (m + 3) = 54 * k)) ∧
  (∃ k1 k2 k3 k4 : ℕ, (n + 3) = 24 * k1 ∧ (n + 3) = 32 * k2 ∧ (n + 3) = 36 * k3 ∧ (n + 3) = 54 * k4) :=
by sorry

#check least_number_divisible

end least_number_divisible_l3727_372780


namespace original_equals_scientific_l3727_372702

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  is_valid : 1 ≤ coefficient ∧ coefficient < 10

/-- The number to be expressed in scientific notation -/
def original_number : ℕ := 1570000000

/-- The scientific notation representation of the original number -/
def scientific_form : ScientificNotation :=
  { coefficient := 1.57
    exponent := 9
    is_valid := by sorry }

/-- Theorem stating that the original number is equal to its scientific notation representation -/
theorem original_equals_scientific :
  (original_number : ℝ) = scientific_form.coefficient * (10 : ℝ) ^ scientific_form.exponent := by
  sorry

end original_equals_scientific_l3727_372702


namespace min_value_of_function_l3727_372734

theorem min_value_of_function (x : ℝ) (h : x > 0) :
  x + 3 / (4 * x) ≥ Real.sqrt 3 ∧ ∃ y > 0, y + 3 / (4 * y) = Real.sqrt 3 := by
  sorry

end min_value_of_function_l3727_372734


namespace expanded_dining_area_total_l3727_372722

/-- The total area of an expanded outdoor dining area consisting of a rectangular section
    with an area of 35 square feet and a semi-circular section with a radius of 4 feet
    is equal to 35 + 8π square feet. -/
theorem expanded_dining_area_total (rectangular_area : ℝ) (semi_circle_radius : ℝ) :
  rectangular_area = 35 ∧ semi_circle_radius = 4 →
  rectangular_area + (1/2 * π * semi_circle_radius^2) = 35 + 8*π := by
  sorry

end expanded_dining_area_total_l3727_372722


namespace sqrt_x_minus_one_real_implies_x_geq_one_l3727_372789

theorem sqrt_x_minus_one_real_implies_x_geq_one (x : ℝ) :
  (∃ y : ℝ, y ^ 2 = x - 1) → x ≥ 1 := by
  sorry

end sqrt_x_minus_one_real_implies_x_geq_one_l3727_372789


namespace complex_power_result_l3727_372719

theorem complex_power_result : (3 * Complex.cos (π / 4) + 3 * Complex.I * Complex.sin (π / 4)) ^ 4 = (-81 : ℂ) := by
  sorry

end complex_power_result_l3727_372719


namespace fuel_left_in_tank_l3727_372765

/-- Calculates the remaining fuel in a plane's tank given the fuel consumption rate and remaining flight time. -/
def remaining_fuel (fuel_rate : ℝ) (flight_time : ℝ) : ℝ :=
  fuel_rate * flight_time

/-- Proves that given a plane using fuel at a rate of 9.5 gallons per hour and can continue flying for 0.6667 hours, the amount of fuel left in the tank is approximately 6.33365 gallons. -/
theorem fuel_left_in_tank : 
  let fuel_rate := 9.5
  let flight_time := 0.6667
  abs (remaining_fuel fuel_rate flight_time - 6.33365) < 0.00001 := by
sorry

end fuel_left_in_tank_l3727_372765


namespace program_output_l3727_372792

theorem program_output : ∃ i : ℕ, (∀ j < i, 2^j ≤ 2000) ∧ (2^i > 2000) ∧ (i - 1 = 10) := by
  sorry

end program_output_l3727_372792


namespace election_votes_l3727_372753

theorem election_votes (votes_A : ℕ) (ratio_A ratio_B : ℕ) : 
  votes_A = 14 → ratio_A = 2 → ratio_B = 1 → 
  votes_A + (votes_A * ratio_B / ratio_A) = 21 := by
  sorry

end election_votes_l3727_372753


namespace workshop_workers_l3727_372714

theorem workshop_workers (total_average : ℕ) (tech_count : ℕ) (tech_average : ℕ) (non_tech_average : ℕ) :
  total_average = 8000 →
  tech_count = 7 →
  tech_average = 12000 →
  non_tech_average = 6000 →
  ∃ (total_workers : ℕ), 
    total_workers * total_average = tech_count * tech_average + (total_workers - tech_count) * non_tech_average ∧
    total_workers = 21 :=
by sorry

end workshop_workers_l3727_372714


namespace intersection_of_A_and_B_l3727_372730

def A : Set ℝ := {-2, -1, 0, 1}
def B : Set ℝ := {x : ℝ | x^2 - 1 ≤ 0}

theorem intersection_of_A_and_B : A ∩ B = {-1, 0, 1} := by sorry

end intersection_of_A_and_B_l3727_372730


namespace sum_of_powers_lower_bound_l3727_372739

theorem sum_of_powers_lower_bound 
  (x y z : ℝ) 
  (n : ℕ) 
  (pos_x : 0 < x) 
  (pos_y : 0 < y) 
  (pos_z : 0 < z) 
  (sum_one : x + y + z = 1) 
  (pos_n : 0 < n) : 
  x^n + y^n + z^n ≥ 1 / (3^(n-1)) := by
sorry

end sum_of_powers_lower_bound_l3727_372739


namespace quadratic_one_solution_positive_n_for_one_solution_l3727_372712

theorem quadratic_one_solution (n : ℝ) : 
  (∃! x : ℝ, 4 * x^2 + n * x + 25 = 0) ↔ n = 20 ∨ n = -20 := by
  sorry

theorem positive_n_for_one_solution (n : ℝ) :
  n > 0 ∧ (∃! x : ℝ, 4 * x^2 + n * x + 25 = 0) → n = 20 := by
  sorry

end quadratic_one_solution_positive_n_for_one_solution_l3727_372712


namespace smallest_sum_xyz_l3727_372725

theorem smallest_sum_xyz (x y z : ℕ+) 
  (eq1 : (x.val + y.val) * (y.val + z.val) = 2016)
  (eq2 : (x.val + y.val) * (z.val + x.val) = 1080) :
  (∀ a b c : ℕ+, 
    (a.val + b.val) * (b.val + c.val) = 2016 → 
    (a.val + b.val) * (c.val + a.val) = 1080 → 
    x.val + y.val + z.val ≤ a.val + b.val + c.val) ∧
  x.val + y.val + z.val = 61 :=
sorry

end smallest_sum_xyz_l3727_372725


namespace triangle_side_product_greater_than_circle_diameters_l3727_372794

theorem triangle_side_product_greater_than_circle_diameters 
  (a b c r R : ℝ) (h_positive : a > 0 ∧ b > 0 ∧ c > 0 ∧ r > 0 ∧ R > 0) 
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b) 
  (h_inradius : r = (a + b - c) * (b + c - a) * (c + a - b) / (4 * (a + b + c)))
  (h_circumradius : R = a * b * c / (4 * (a + b - c) * (b + c - a) * (c + a - b))) :
  a * b > 4 * r * R :=
by sorry

end triangle_side_product_greater_than_circle_diameters_l3727_372794


namespace highest_affordable_price_is_8_l3727_372701

/-- The highest whole-dollar price per shirt Alec can afford -/
def highest_affordable_price (total_budget : ℕ) (num_shirts : ℕ) (entrance_fee : ℕ) (tax_rate : ℚ) : ℕ :=
  sorry

/-- The proposition to be proved -/
theorem highest_affordable_price_is_8 :
  highest_affordable_price 180 20 5 (8/100) = 8 := by
  sorry

end highest_affordable_price_is_8_l3727_372701


namespace recurring_decimal_sum_l3727_372788

theorem recurring_decimal_sum : 
  (∃ (x y : ℚ), x = 123 / 999 ∧ y = 123 / 999999 ∧ x + y = 154 / 1001) :=
by sorry

end recurring_decimal_sum_l3727_372788


namespace division_expression_equality_l3727_372723

theorem division_expression_equality : 180 / (8 + 9 * 3 - 4) = 180 / 31 := by
  sorry

end division_expression_equality_l3727_372723


namespace cubic_polynomial_three_distinct_roots_l3727_372752

/-- A cubic polynomial with specific properties has three distinct real roots -/
theorem cubic_polynomial_three_distinct_roots 
  (f : ℝ → ℝ) (a b c : ℝ) 
  (h_cubic : ∀ x, f x = x^3 + a*x^2 + b*x + c) 
  (h_b_neg : b < 0) 
  (h_ab_9c : a * b = 9 * c) : 
  ∃ (x y z : ℝ), x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ f x = 0 ∧ f y = 0 ∧ f z = 0 :=
sorry

end cubic_polynomial_three_distinct_roots_l3727_372752


namespace friends_receiving_pens_correct_l3727_372728

/-- Calculate the number of friends who will receive pens --/
def friends_receiving_pens (kendra_packs tony_packs maria_packs : ℕ)
                           (kendra_pens_per_pack tony_pens_per_pack maria_pens_per_pack : ℕ)
                           (pens_kept_per_person : ℕ) : ℕ :=
  let kendra_total := kendra_packs * kendra_pens_per_pack
  let tony_total := tony_packs * tony_pens_per_pack
  let maria_total := maria_packs * maria_pens_per_pack
  let total_pens := kendra_total + tony_total + maria_total
  let total_kept := 3 * pens_kept_per_person
  total_pens - total_kept

theorem friends_receiving_pens_correct :
  friends_receiving_pens 7 5 9 4 6 5 3 = 94 := by
  sorry

end friends_receiving_pens_correct_l3727_372728


namespace chicken_flash_sale_theorem_l3727_372745

/-- Represents the original selling price of a free-range ecological chicken -/
def original_price : ℝ := sorry

/-- Represents the flash sale price of a free-range ecological chicken -/
def flash_sale_price : ℝ := original_price - 15

/-- Represents the percentage increase in buyers every 30 minutes -/
def m : ℝ := sorry

theorem chicken_flash_sale_theorem :
  (120 / flash_sale_price = 2 * (90 / original_price)) ∧
  (50 + 50 * (1 + m / 100) + 50 * (1 + m / 100)^2 = 5460 / flash_sale_price) →
  original_price = 45 ∧ m = 20 := by sorry

end chicken_flash_sale_theorem_l3727_372745


namespace trig_identity_l3727_372720

theorem trig_identity (α β : Real) 
  (h : (Real.cos α)^6 / (Real.cos β)^3 + (Real.sin α)^6 / (Real.sin β)^3 = 1) :
  (Real.sin β)^6 / (Real.sin α)^3 + (Real.cos β)^6 / (Real.cos α)^3 = 1 := by
  sorry

end trig_identity_l3727_372720


namespace triangle_angle_difference_l3727_372735

theorem triangle_angle_difference (a b c : ℝ) : 
  a = 64 ∧ b = 64 ∧ c < a ∧ a + b + c = 180 → a - c = 12 := by
  sorry

end triangle_angle_difference_l3727_372735


namespace rectangle_area_l3727_372705

/-- Given a rectangle with width 10 meters, if its length is increased such that the new area is 4/3 times the original area and the new perimeter is 60 meters, then the original area of the rectangle is 150 square meters. -/
theorem rectangle_area (original_length : ℝ) : 
  let original_width : ℝ := 10
  let new_length : ℝ := (60 - 2 * original_width) / 2
  let new_area : ℝ := new_length * original_width
  let original_area : ℝ := original_length * original_width
  new_area = (4/3) * original_area → original_area = 150 := by
sorry


end rectangle_area_l3727_372705


namespace vector_equation_solution_l3727_372767

/-- Given vector a and an equation involving a and b, prove that b equals (1, -2) -/
theorem vector_equation_solution (a b : ℝ × ℝ) : 
  a = (1, 2) → 
  (2 • a) + b = (3, 2) → 
  b = (1, -2) := by
sorry

end vector_equation_solution_l3727_372767


namespace base_k_representation_of_5_29_l3727_372706

theorem base_k_representation_of_5_29 (k : ℕ) : k > 0 → (
  (5 : ℚ) / 29 = (k + 3 : ℚ) / (k^2 - 1) ↔ k = 8
) := by sorry

end base_k_representation_of_5_29_l3727_372706
