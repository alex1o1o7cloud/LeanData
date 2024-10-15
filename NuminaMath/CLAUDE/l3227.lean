import Mathlib

namespace NUMINAMATH_CALUDE_ball_final_position_l3227_322752

/-- Represents the possible final positions of the ball -/
inductive FinalPosition
  | B
  | A
  | C

/-- Determines the final position of the ball based on the parity of m and n -/
def finalBallPosition (m n : ℕ) : FinalPosition :=
  if m % 2 = 1 ∧ n % 2 = 1 then FinalPosition.B
  else if m % 2 = 0 ∧ n % 2 = 1 then FinalPosition.A
  else FinalPosition.C

/-- Theorem stating the final position of the ball -/
theorem ball_final_position (m n : ℕ) :
  (m > 0 ∧ n > 0) →
  (finalBallPosition m n = FinalPosition.B ↔ m % 2 = 1 ∧ n % 2 = 1) ∧
  (finalBallPosition m n = FinalPosition.A ↔ m % 2 = 0 ∧ n % 2 = 1) ∧
  (finalBallPosition m n = FinalPosition.C ↔ m % 2 = 1 ∧ n % 2 = 0) :=
by sorry

end NUMINAMATH_CALUDE_ball_final_position_l3227_322752


namespace NUMINAMATH_CALUDE_pages_already_read_l3227_322772

/-- Theorem: Number of pages Rich has already read
Given a book with 372 pages, where Rich skipped 16 pages of maps and has 231 pages left to read,
prove that Rich has already read 125 pages. -/
theorem pages_already_read
  (total_pages : ℕ)
  (skipped_pages : ℕ)
  (pages_left : ℕ)
  (h1 : total_pages = 372)
  (h2 : skipped_pages = 16)
  (h3 : pages_left = 231) :
  total_pages - skipped_pages - pages_left = 125 := by
  sorry

end NUMINAMATH_CALUDE_pages_already_read_l3227_322772


namespace NUMINAMATH_CALUDE_parabola_kite_sum_l3227_322703

/-- Two parabolas intersecting coordinate axes to form a kite -/
def parabola_kite (a b : ℝ) : Prop :=
  ∃ (x₁ x₂ y₁ y₂ : ℝ),
    -- Parabola equations
    (∀ x, a * x^2 - 4 = 6 - b * x^2 → x = x₁ ∨ x = x₂) ∧
    (∀ y, y = a * 0^2 - 4 → y = y₁) ∧
    (∀ y, y = 6 - b * 0^2 → y = y₂) ∧
    -- Four distinct intersection points
    x₁ ≠ x₂ ∧ y₁ ≠ y₂ ∧
    -- Kite area
    (1/2) * (x₂ - x₁) * (y₂ - y₁) = 18

/-- Theorem: If two parabolas form a kite with area 18, then a + b = 125/36 -/
theorem parabola_kite_sum (a b : ℝ) :
  parabola_kite a b → a + b = 125/36 := by
  sorry

end NUMINAMATH_CALUDE_parabola_kite_sum_l3227_322703


namespace NUMINAMATH_CALUDE_squared_sum_minus_sum_of_squares_l3227_322721

theorem squared_sum_minus_sum_of_squares : (37 + 12)^2 - (37^2 + 12^2) = 888 := by
  sorry

end NUMINAMATH_CALUDE_squared_sum_minus_sum_of_squares_l3227_322721


namespace NUMINAMATH_CALUDE_four_numbers_perfect_square_product_l3227_322779

/-- A set of positive integers where all prime divisors are smaller than 30 -/
def SmallPrimeDivisorSet : Type := {s : Finset ℕ+ // ∀ n ∈ s, ∀ p : ℕ, Prime p → p ∣ n → p < 30}

theorem four_numbers_perfect_square_product (A : SmallPrimeDivisorSet) (h : A.val.card = 2016) :
  ∃ a b c d : ℕ+, a ∈ A.val ∧ b ∈ A.val ∧ c ∈ A.val ∧ d ∈ A.val ∧
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
  ∃ k : ℕ+, (a * b * c * d : ℕ) = k ^ 2 :=
sorry

end NUMINAMATH_CALUDE_four_numbers_perfect_square_product_l3227_322779


namespace NUMINAMATH_CALUDE_pizza_recipe_water_amount_l3227_322707

theorem pizza_recipe_water_amount :
  ∀ (water flour salt : ℚ),
    flour = 16 →
    salt = (1/2) * flour →
    water + flour + salt = 34 →
    water = 10 :=
by sorry

end NUMINAMATH_CALUDE_pizza_recipe_water_amount_l3227_322707


namespace NUMINAMATH_CALUDE_math_team_combinations_l3227_322749

def number_of_teams (n_girls m_boys k_girls l_boys : ℕ) : ℕ :=
  Nat.choose n_girls k_girls * Nat.choose m_boys l_boys

theorem math_team_combinations :
  number_of_teams 5 7 3 2 = 210 := by
  sorry

end NUMINAMATH_CALUDE_math_team_combinations_l3227_322749


namespace NUMINAMATH_CALUDE_only_setD_forms_triangle_l3227_322708

-- Define a structure for a set of three line segments
structure SegmentSet where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define the triangle inequality condition
def satisfiesTriangleInequality (s : SegmentSet) : Prop :=
  s.a + s.b > s.c ∧ s.b + s.c > s.a ∧ s.c + s.a > s.b

-- Define the given sets of line segments
def setA : SegmentSet := ⟨1, 2, 3.5⟩
def setB : SegmentSet := ⟨4, 5, 9⟩
def setC : SegmentSet := ⟨5, 8, 15⟩
def setD : SegmentSet := ⟨6, 8, 9⟩

-- Theorem stating that only setD satisfies the triangle inequality
theorem only_setD_forms_triangle :
  ¬(satisfiesTriangleInequality setA) ∧
  ¬(satisfiesTriangleInequality setB) ∧
  ¬(satisfiesTriangleInequality setC) ∧
  satisfiesTriangleInequality setD :=
sorry

end NUMINAMATH_CALUDE_only_setD_forms_triangle_l3227_322708


namespace NUMINAMATH_CALUDE_min_both_beethoven_chopin_survey_result_l3227_322729

theorem min_both_beethoven_chopin 
  (total : ℕ) 
  (likes_beethoven : ℕ) 
  (likes_chopin : ℕ) 
  (h1 : total = 200)
  (h2 : likes_beethoven = 160)
  (h3 : likes_chopin = 150)
  : ℕ := by
  
  -- Define the minimum number who like both
  let min_both := likes_beethoven + likes_chopin - total
  
  -- Prove that min_both is the minimum number who like both Beethoven and Chopin
  sorry

-- State the theorem
theorem survey_result : min_both_beethoven_chopin 200 160 150 rfl rfl rfl = 110 := by
  sorry

end NUMINAMATH_CALUDE_min_both_beethoven_chopin_survey_result_l3227_322729


namespace NUMINAMATH_CALUDE_bales_in_barn_after_addition_l3227_322777

/-- The number of bales in the barn after addition -/
def bales_after_addition (initial_bales : ℕ) (added_bales : ℕ) : ℕ :=
  initial_bales + added_bales

/-- Theorem stating that the number of bales after Benny's addition is 82 -/
theorem bales_in_barn_after_addition :
  bales_after_addition 47 35 = 82 := by
  sorry

end NUMINAMATH_CALUDE_bales_in_barn_after_addition_l3227_322777


namespace NUMINAMATH_CALUDE_brothers_difference_l3227_322789

theorem brothers_difference (aaron_brothers : ℕ) (bennett_brothers : ℕ) : 
  aaron_brothers = 4 → bennett_brothers = 6 → 2 * aaron_brothers - bennett_brothers = 2 := by
  sorry

end NUMINAMATH_CALUDE_brothers_difference_l3227_322789


namespace NUMINAMATH_CALUDE_work_completion_time_l3227_322775

-- Define the rates of work for A and B
def rate_A : ℚ := 1 / 16
def rate_B : ℚ := rate_A / 3

-- Define the total rate when A and B work together
def total_rate : ℚ := rate_A + rate_B

-- Theorem statement
theorem work_completion_time :
  (1 : ℚ) / total_rate = 12 := by sorry

end NUMINAMATH_CALUDE_work_completion_time_l3227_322775


namespace NUMINAMATH_CALUDE_triangle_properties_l3227_322716

noncomputable def angle_A (A B C : ℝ) (a b c : ℝ) : Prop :=
  a * Real.cos C + (1/2) * c = b ∧ a = 1 → A = Real.pi / 3

def perimeter_range (A B C : ℝ) (a b c : ℝ) : Prop :=
  a * Real.cos C + (1/2) * c = b ∧ a = 1 →
  let l := a + b + c
  2 < l ∧ l ≤ 3

theorem triangle_properties (A B C : ℝ) (a b c : ℝ) :
  angle_A A B C a b c ∧ perimeter_range A B C a b c :=
sorry

end NUMINAMATH_CALUDE_triangle_properties_l3227_322716


namespace NUMINAMATH_CALUDE_butterfat_mixture_l3227_322791

/-- Proves that mixing 8 gallons of 50% butterfat milk with 24 gallons of 10% butterfat milk results in a mixture that is 20% butterfat. -/
theorem butterfat_mixture : 
  let milk_50_percent : ℝ := 8
  let milk_10_percent : ℝ := 24
  let butterfat_50_percent : ℝ := 0.5
  let butterfat_10_percent : ℝ := 0.1
  let total_volume : ℝ := milk_50_percent + milk_10_percent
  let total_butterfat : ℝ := milk_50_percent * butterfat_50_percent + milk_10_percent * butterfat_10_percent
  total_butterfat / total_volume = 0.2 := by
sorry

end NUMINAMATH_CALUDE_butterfat_mixture_l3227_322791


namespace NUMINAMATH_CALUDE_unique_natural_number_with_specific_divisor_differences_l3227_322739

theorem unique_natural_number_with_specific_divisor_differences :
  ∃! n : ℕ,
    (∃ d₁ d₂ : ℕ, d₁ ∣ n ∧ d₂ ∣ n ∧ d₁ < d₂ ∧ ∀ d : ℕ, d ∣ n → d = d₁ ∨ d ≥ d₂) ∧
    (d₂ - d₁ = 4) ∧
    (∃ d₃ d₄ : ℕ, d₃ ∣ n ∧ d₄ ∣ n ∧ d₃ < d₄ ∧ ∀ d : ℕ, d ∣ n → d ≤ d₃ ∨ d = d₄) ∧
    (d₄ - d₃ = 308) ∧
    n = 385 :=
by
  sorry

end NUMINAMATH_CALUDE_unique_natural_number_with_specific_divisor_differences_l3227_322739


namespace NUMINAMATH_CALUDE_projection_a_onto_b_l3227_322750

def a : ℝ × ℝ := (3, 4)
def b : ℝ × ℝ := (-2, 4)

theorem projection_a_onto_b :
  let proj := (a.1 * b.1 + a.2 * b.2) / Real.sqrt ((b.1 ^ 2 + b.2 ^ 2))
  proj = Real.sqrt 5 := by sorry

end NUMINAMATH_CALUDE_projection_a_onto_b_l3227_322750


namespace NUMINAMATH_CALUDE_vector_complex_correspondence_l3227_322738

theorem vector_complex_correspondence (z : ℂ) :
  z = -3 + 2*I → (-z) = 3 - 2*I := by sorry

end NUMINAMATH_CALUDE_vector_complex_correspondence_l3227_322738


namespace NUMINAMATH_CALUDE_strawberry_distribution_l3227_322797

theorem strawberry_distribution (initial : ℕ) (additional : ℕ) (boxes : ℕ) 
  (h1 : initial = 42)
  (h2 : additional = 78)
  (h3 : boxes = 6)
  (h4 : boxes ≠ 0) :
  (initial + additional) / boxes = 20 := by
  sorry

end NUMINAMATH_CALUDE_strawberry_distribution_l3227_322797


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l3227_322711

theorem quadratic_equation_solution (x : ℝ) : -x^2 - (-16 + 10)*x - 8 = -(x - 2)*(x - 4) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l3227_322711


namespace NUMINAMATH_CALUDE_intersection_AB_XOZ_plane_l3227_322736

/-- Given two points A and B in 3D space, this function returns the coordinates of the 
    intersection point of the line passing through A and B with the XOZ plane. -/
def intersectionWithXOZPlane (A B : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  sorry

/-- Theorem stating that the intersection of the line passing through A(1,-2,-3) and B(2,-1,-1) 
    with the XOZ plane is the point (3,0,1). -/
theorem intersection_AB_XOZ_plane :
  let A : ℝ × ℝ × ℝ := (1, -2, -3)
  let B : ℝ × ℝ × ℝ := (2, -1, -1)
  intersectionWithXOZPlane A B = (3, 0, 1) := by
  sorry

end NUMINAMATH_CALUDE_intersection_AB_XOZ_plane_l3227_322736


namespace NUMINAMATH_CALUDE_angle_C_value_l3227_322778

-- Define the triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the conditions
def satisfiesConditions (t : Triangle) : Prop :=
  t.a = 5 ∧ 
  t.b + t.c = 2 * t.a ∧ 
  3 * Real.sin t.A = 5 * Real.sin t.B

-- Theorem statement
theorem angle_C_value (t : Triangle) (h : satisfiesConditions t) : t.C = 2 * Real.pi / 3 := by
  sorry

end NUMINAMATH_CALUDE_angle_C_value_l3227_322778


namespace NUMINAMATH_CALUDE_quadratic_inequality_properties_l3227_322743

/-- Given that the solution set of ax² - bx + c > 0 is (-1, 2), prove the following properties -/
theorem quadratic_inequality_properties 
  (a b c : ℝ) 
  (h : ∀ x : ℝ, ax^2 - b*x + c > 0 ↔ -1 < x ∧ x < 2) : 
  (a + b + c = 0) ∧ (a < 0) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_properties_l3227_322743


namespace NUMINAMATH_CALUDE_square_39_equals_square_40_minus_79_l3227_322782

theorem square_39_equals_square_40_minus_79 : (39 : ℤ)^2 = (40 : ℤ)^2 - 79 := by
  sorry

end NUMINAMATH_CALUDE_square_39_equals_square_40_minus_79_l3227_322782


namespace NUMINAMATH_CALUDE_circles_tangent_m_value_l3227_322737

-- Define the circles
def circle_C1 (x y : ℝ) : Prop := x^2 + y^2 = 1
def circle_C2 (x y m : ℝ) : Prop := x^2 + y^2 - 6*x - 8*y + m = 0

-- Define the tangency condition
def are_tangent (C1 C2 : ℝ → ℝ → Prop) : Prop :=
  ∃ (x y : ℝ), C1 x y ∧ C2 x y ∧
  ∀ (x' y' : ℝ), (C1 x' y' ∧ C2 x' y') → (x' = x ∧ y' = y)

-- Theorem statement
theorem circles_tangent_m_value :
  are_tangent circle_C1 (circle_C2 · · 9) :=
sorry

end NUMINAMATH_CALUDE_circles_tangent_m_value_l3227_322737


namespace NUMINAMATH_CALUDE_work_efficiency_ratio_l3227_322742

/-- Given a road that can be repaired by A in 4 days or by B in 5 days,
    the ratio of A's work efficiency to B's work efficiency is 5/4. -/
theorem work_efficiency_ratio (road : ℝ) (days_A days_B : ℕ) 
  (h_A : road / days_A = road / 4)
  (h_B : road / days_B = road / 5) :
  (road / days_A) / (road / days_B) = 5 / 4 := by
  sorry

end NUMINAMATH_CALUDE_work_efficiency_ratio_l3227_322742


namespace NUMINAMATH_CALUDE_total_children_l3227_322732

/-- The number of children who like cabbage -/
def cabbage_lovers : ℕ := 7

/-- The number of children who like carrots -/
def carrot_lovers : ℕ := 6

/-- The number of children who like peas -/
def pea_lovers : ℕ := 5

/-- The number of children who like both cabbage and carrots -/
def cabbage_carrot_lovers : ℕ := 4

/-- The number of children who like both cabbage and peas -/
def cabbage_pea_lovers : ℕ := 3

/-- The number of children who like both carrots and peas -/
def carrot_pea_lovers : ℕ := 2

/-- The number of children who like all three vegetables -/
def all_veg_lovers : ℕ := 1

/-- The theorem stating the total number of children in the family -/
theorem total_children : 
  cabbage_lovers + carrot_lovers + pea_lovers - 
  cabbage_carrot_lovers - cabbage_pea_lovers - carrot_pea_lovers + 
  all_veg_lovers = 10 := by
  sorry

end NUMINAMATH_CALUDE_total_children_l3227_322732


namespace NUMINAMATH_CALUDE_min_b_minus_a_l3227_322722

open Real

noncomputable def f (x : ℝ) : ℝ := log x - 1 / x

noncomputable def g (a b x : ℝ) : ℝ := -a * x + b

def is_tangent_line (f g : ℝ → ℝ) : Prop :=
  ∃ x₀, (∀ x, g x = f x₀ + (deriv f x₀) * (x - x₀))

theorem min_b_minus_a (a b : ℝ) :
  (∀ x, x > 0 → f x = f x) →
  is_tangent_line f (g a b) →
  b - a ≥ -1 ∧ ∃ a₀ b₀, b₀ - a₀ = -1 :=
sorry

end NUMINAMATH_CALUDE_min_b_minus_a_l3227_322722


namespace NUMINAMATH_CALUDE_geric_bills_count_geric_bills_proof_l3227_322730

theorem geric_bills_count : ℕ → ℕ → ℕ → Prop :=
  fun geric_bills kyla_bills jessa_bills =>
    (geric_bills = 2 * kyla_bills) ∧
    (kyla_bills = jessa_bills - 2) ∧
    (jessa_bills - 3 = 7) →
    geric_bills = 16

-- The proof goes here
theorem geric_bills_proof : ∃ g k j, geric_bills_count g k j :=
  sorry

end NUMINAMATH_CALUDE_geric_bills_count_geric_bills_proof_l3227_322730


namespace NUMINAMATH_CALUDE_whiteboard_washing_l3227_322717

theorem whiteboard_washing (kids : ℕ) (whiteboards : ℕ) (time : ℕ) :
  kids = 4 →
  whiteboards = 3 →
  time = 20 →
  (1 : ℝ) * 160 * whiteboards = kids * time * 6 :=
by sorry

end NUMINAMATH_CALUDE_whiteboard_washing_l3227_322717


namespace NUMINAMATH_CALUDE_phone_reselling_profit_l3227_322723

theorem phone_reselling_profit (initial_investment : ℝ) (profit_ratio : ℝ) (selling_price : ℝ) :
  initial_investment = 3000 →
  profit_ratio = 1 / 3 →
  selling_price = 20 →
  (initial_investment * (1 + profit_ratio)) / selling_price = 200 := by
  sorry

end NUMINAMATH_CALUDE_phone_reselling_profit_l3227_322723


namespace NUMINAMATH_CALUDE_quadratic_equation_general_form_l3227_322705

theorem quadratic_equation_general_form :
  ∀ x : ℝ, (x - 2) * (x + 3) = 1 ↔ x^2 + x - 7 = 0 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_general_form_l3227_322705


namespace NUMINAMATH_CALUDE_jonathan_book_purchase_l3227_322704

-- Define the costs of the books and Jonathan's savings
def dictionary_cost : ℕ := 11
def dinosaur_book_cost : ℕ := 19
def cookbook_cost : ℕ := 7
def savings : ℕ := 8

-- Define the total cost of the books
def total_cost : ℕ := dictionary_cost + dinosaur_book_cost + cookbook_cost

-- Define the amount Jonathan needs
def amount_needed : ℕ := total_cost - savings

-- Theorem statement
theorem jonathan_book_purchase :
  amount_needed = 29 :=
by sorry

end NUMINAMATH_CALUDE_jonathan_book_purchase_l3227_322704


namespace NUMINAMATH_CALUDE_diamond_calculation_l3227_322757

-- Define the diamond operation
def diamond (a b : ℚ) : ℚ := a - 1 / b

-- Theorem statement
theorem diamond_calculation :
  (diamond (diamond 2 3) 4) - (diamond 2 (diamond 3 4)) = -29/132 :=
by sorry

end NUMINAMATH_CALUDE_diamond_calculation_l3227_322757


namespace NUMINAMATH_CALUDE_sum_of_cubes_of_roots_l3227_322784

theorem sum_of_cubes_of_roots (a b c : ℝ) (h : b^2 - 4*a*c > 0) :
  let x₁ := (-b + Real.sqrt (b^2 - 4*a*c)) / (2*a)
  let x₂ := (-b - Real.sqrt (b^2 - 4*a*c)) / (2*a)
  x₁^3 + x₂^3 = 95/8 :=
by
  sorry

end NUMINAMATH_CALUDE_sum_of_cubes_of_roots_l3227_322784


namespace NUMINAMATH_CALUDE_apple_pyramid_theorem_l3227_322799

/-- Calculates the number of apples in a layer of the pyramid -/
def apples_in_layer (base_width : ℕ) (base_length : ℕ) (layer : ℕ) : ℕ :=
  (base_width - layer + 1) * (base_length - layer + 1)

/-- Calculates the total number of apples in the pyramid stack -/
def total_apples (base_width : ℕ) (base_length : ℕ) : ℕ :=
  let num_layers := min base_width base_length
  (List.range num_layers).foldl (fun acc i => acc + apples_in_layer base_width base_length i) 0 + 1

theorem apple_pyramid_theorem :
  total_apples 6 9 = 155 := by
  sorry

#eval total_apples 6 9

end NUMINAMATH_CALUDE_apple_pyramid_theorem_l3227_322799


namespace NUMINAMATH_CALUDE_infinite_nonzero_digit_sum_equality_l3227_322795

/-- Sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- Check if a natural number contains zero in its digits -/
def contains_zero (n : ℕ) : Prop := sorry

theorem infinite_nonzero_digit_sum_equality :
  ∀ k : ℕ, ∃ f : ℕ → ℕ,
    (∀ n : ℕ, ¬contains_zero (f n)) ∧
    (∀ n : ℕ, sum_of_digits (f n) = sum_of_digits (k * f n)) ∧
    (∀ n : ℕ, f n < f (n + 1)) :=
by sorry

end NUMINAMATH_CALUDE_infinite_nonzero_digit_sum_equality_l3227_322795


namespace NUMINAMATH_CALUDE_smallest_n_terminating_with_3_l3227_322765

def is_terminating_decimal (n : ℕ) : Prop :=
  ∃ (a b : ℕ), n = 2^a * 5^b

def contains_digit_3 (n : ℕ) : Prop :=
  ∃ (d : ℕ), d < 10 ∧ (n / 10^d) % 10 = 3

theorem smallest_n_terminating_with_3 :
  ∀ n : ℕ, n > 0 →
    (is_terminating_decimal n ∧ contains_digit_3 n) →
    n ≥ 32 :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_terminating_with_3_l3227_322765


namespace NUMINAMATH_CALUDE_coefficient_of_monomial_degree_of_monomial_l3227_322768

-- Define the monomial
def monomial : ℚ × (ℕ × ℕ) := (-4/3, (2, 1))

-- Theorem for the coefficient
theorem coefficient_of_monomial :
  (monomial.fst : ℚ) = -4/3 := by sorry

-- Theorem for the degree
theorem degree_of_monomial :
  (monomial.snd.fst + monomial.snd.snd : ℕ) = 3 := by sorry

end NUMINAMATH_CALUDE_coefficient_of_monomial_degree_of_monomial_l3227_322768


namespace NUMINAMATH_CALUDE_circles_in_rectangle_l3227_322781

theorem circles_in_rectangle (targetSum : ℝ) (h : targetSum = 1962) :
  ∃ (α : ℝ), 0 < α ∧ α < 1 / 3925 ∧
  ∀ (rectangle : Set (ℝ × ℝ)),
    (∃ a b, rectangle = {(x, y) | 0 ≤ x ∧ x ≤ a ∧ 0 ≤ y ∧ y ≤ b} ∧ a * b = 1) →
    ∃ (n m : ℕ),
      (n : ℝ) * (m : ℝ) * (α / 2) > targetSum :=
by sorry

end NUMINAMATH_CALUDE_circles_in_rectangle_l3227_322781


namespace NUMINAMATH_CALUDE_unique_sequence_l3227_322770

/-- A strictly increasing sequence of natural numbers -/
def StrictlyIncreasingSeq (a : ℕ → ℕ) : Prop :=
  ∀ n m : ℕ, n < m → a n < a m

/-- The property that a₂ = 2 -/
def SecondTermIsTwo (a : ℕ → ℕ) : Prop :=
  a 2 = 2

/-- The property that aₙₘ = aₙ * aₘ for any natural numbers n and m -/
def MultiplicativeProperty (a : ℕ → ℕ) : Prop :=
  ∀ n m : ℕ, a (n * m) = a n * a m

/-- The theorem stating that the only sequence satisfying all conditions is aₙ = n -/
theorem unique_sequence :
  ∀ a : ℕ → ℕ,
    StrictlyIncreasingSeq a →
    SecondTermIsTwo a →
    MultiplicativeProperty a →
    ∀ n : ℕ, a n = n :=
by sorry

end NUMINAMATH_CALUDE_unique_sequence_l3227_322770


namespace NUMINAMATH_CALUDE_function_range_l3227_322745

theorem function_range (a : ℝ) : 
  (a > 0) →
  (∀ x₁ : ℝ, ∃ x₂ : ℝ, x₂ ≥ -2 ∧ (x₁^2 - 2*x₁) > (a*x₂ + 2)) →
  a > 3/2 := by
sorry

end NUMINAMATH_CALUDE_function_range_l3227_322745


namespace NUMINAMATH_CALUDE_yogurt_satisfaction_probability_l3227_322776

theorem yogurt_satisfaction_probability 
  (total_sample : ℕ) 
  (satisfied_with_yogurt : ℕ) 
  (h1 : total_sample = 500) 
  (h2 : satisfied_with_yogurt = 370) : 
  (satisfied_with_yogurt : ℚ) / total_sample = 37 / 50 := by
  sorry

end NUMINAMATH_CALUDE_yogurt_satisfaction_probability_l3227_322776


namespace NUMINAMATH_CALUDE_min_perimeter_rectangle_is_square_l3227_322753

/-- For a rectangle with area S and sides a and b, the perimeter is minimized when it's a square -/
theorem min_perimeter_rectangle_is_square (S : ℝ) (h : S > 0) :
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ a * b = S ∧
  ∀ (x y : ℝ), x > 0 → y > 0 → x * y = S →
  2 * (a + b) ≤ 2 * (x + y) ∧
  (2 * (a + b) = 2 * (x + y) → a = b) :=
by sorry


end NUMINAMATH_CALUDE_min_perimeter_rectangle_is_square_l3227_322753


namespace NUMINAMATH_CALUDE_sum_of_sqrt_geq_sum_of_products_l3227_322701

theorem sum_of_sqrt_geq_sum_of_products (a b c : ℝ) 
  (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) 
  (h_sum : a + b + c = 3) : 
  Real.sqrt a + Real.sqrt b + Real.sqrt c ≥ a * b + b * c + a * c := by
  sorry

end NUMINAMATH_CALUDE_sum_of_sqrt_geq_sum_of_products_l3227_322701


namespace NUMINAMATH_CALUDE_range_of_f_l3227_322746

def f (x : ℝ) : ℝ := x^2 - 2*x

def domain : Set ℝ := {0, 1, 2, 3}

theorem range_of_f : 
  {y | ∃ x ∈ domain, f x = y} = {-1, 0, 3} := by sorry

end NUMINAMATH_CALUDE_range_of_f_l3227_322746


namespace NUMINAMATH_CALUDE_factorization_equality_l3227_322712

theorem factorization_equality (m a : ℝ) : 3 * m * a^2 - 6 * m * a + 3 * m = 3 * m * (a - 1)^2 := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l3227_322712


namespace NUMINAMATH_CALUDE_johns_snack_spending_l3227_322748

theorem johns_snack_spending (initial_amount : ℝ) (remaining_amount : ℝ) 
  (snack_fraction : ℝ) (necessity_fraction : ℝ) :
  initial_amount = 20 →
  remaining_amount = 4 →
  necessity_fraction = 3/4 →
  remaining_amount = initial_amount * (1 - snack_fraction) * (1 - necessity_fraction) →
  snack_fraction = 1/5 := by
  sorry

end NUMINAMATH_CALUDE_johns_snack_spending_l3227_322748


namespace NUMINAMATH_CALUDE_norway_visitors_l3227_322728

/-- Given a group of people with information about their visits to Iceland and Norway,
    calculate the number of people who visited Norway. -/
theorem norway_visitors
  (total : ℕ)
  (iceland : ℕ)
  (both : ℕ)
  (neither : ℕ)
  (h1 : total = 50)
  (h2 : iceland = 25)
  (h3 : both = 21)
  (h4 : neither = 23) :
  total = iceland + (norway : ℕ) - both + neither ∧ norway = 23 :=
by sorry

end NUMINAMATH_CALUDE_norway_visitors_l3227_322728


namespace NUMINAMATH_CALUDE_chord_line_equation_l3227_322713

/-- Given an ellipse and a chord midpoint, prove the equation of the line containing the chord -/
theorem chord_line_equation (x y : ℝ) :
  (x^2 / 4 + y^2 / 3 = 1) →  -- Ellipse equation
  (∃ x1 y1 x2 y2 : ℝ,        -- Endpoints of the chord
    x1^2 / 4 + y1^2 / 3 = 1 ∧
    x2^2 / 4 + y2^2 / 3 = 1 ∧
    (x1 + x2) / 2 = -1 ∧     -- Midpoint x-coordinate
    (y1 + y2) / 2 = 1) →     -- Midpoint y-coordinate
  (∃ a b c : ℝ,              -- Line equation coefficients
    a * x + b * y + c = 0 ∧  -- General form of line equation
    a = 3 ∧ b = -4 ∧ c = 7)  -- Specific coefficients for the answer
  := by sorry

end NUMINAMATH_CALUDE_chord_line_equation_l3227_322713


namespace NUMINAMATH_CALUDE_parabola_vertex_l3227_322798

/-- The parabola equation -/
def parabola_equation (x y : ℝ) : Prop :=
  y^2 - 4*y + 3*x + 7 = 0

/-- The vertex of the parabola -/
def vertex : ℝ × ℝ := (-1, 2)

/-- Theorem stating that the vertex of the parabola is (-1, 2) -/
theorem parabola_vertex :
  ∀ (x y : ℝ), parabola_equation x y → 
  ∃! (vx vy : ℝ), vx = vertex.1 ∧ vy = vertex.2 ∧
  (∀ (x' y' : ℝ), parabola_equation x' y' → (x' - vx)^2 + (y' - vy)^2 ≤ (x - vx)^2 + (y - vy)^2) :=
by sorry

end NUMINAMATH_CALUDE_parabola_vertex_l3227_322798


namespace NUMINAMATH_CALUDE_inequality_proof_l3227_322787

theorem inequality_proof (x y : ℝ) (hx : x > 1) (hy : y > 1) :
  (x^2 / (y - 1)) + (y^2 / (x - 1)) ≥ 8 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3227_322787


namespace NUMINAMATH_CALUDE_cube_sum_from_sum_and_square_sum_l3227_322710

theorem cube_sum_from_sum_and_square_sum (x y : ℝ) 
  (h1 : x + y = 5) 
  (h2 : x^2 + y^2 = 17) : 
  x^3 + y^3 = 65 := by
sorry

end NUMINAMATH_CALUDE_cube_sum_from_sum_and_square_sum_l3227_322710


namespace NUMINAMATH_CALUDE_geometric_sequence_first_term_range_l3227_322726

/-- Given an infinite geometric sequence {a_n} with common ratio q,
    if the sum of all terms is equal to q, then the range of the first term a_1 is:
    -2 < a_1 ≤ 1/4 and a_1 ≠ 0 -/
theorem geometric_sequence_first_term_range (a : ℕ → ℝ) (q : ℝ) :
  (∀ n : ℕ, a (n + 1) = q * a n) →  -- Common ratio is q
  (∃ S : ℝ, S = q ∧ S = ∑' n, a n) →  -- Sum of all terms is q
  (-2 < a 0 ∧ a 0 ≤ 1/4 ∧ a 0 ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_first_term_range_l3227_322726


namespace NUMINAMATH_CALUDE_five_digit_base10_to_base2_sum_l3227_322724

theorem five_digit_base10_to_base2_sum : ∃ (min max : ℕ),
  (∀ n : ℕ, 10000 ≤ n ∧ n ≤ 99999 →
    min ≤ (Nat.log 2 n + 1) ∧ (Nat.log 2 n + 1) ≤ max) ∧
  (max - min + 1) * (min + max) / 2 = 62 := by
  sorry

end NUMINAMATH_CALUDE_five_digit_base10_to_base2_sum_l3227_322724


namespace NUMINAMATH_CALUDE_prob_different_cities_l3227_322783

/-- The probability that student A attends university in city A -/
def prob_A_cityA : ℝ := 0.6

/-- The probability that student B attends university in city A -/
def prob_B_cityA : ℝ := 0.3

/-- The theorem stating that the probability of A and B not attending university 
    in the same city is 0.54, given the probabilities of each student 
    attending city A -/
theorem prob_different_cities (h1 : 0 ≤ prob_A_cityA ∧ prob_A_cityA ≤ 1) 
                               (h2 : 0 ≤ prob_B_cityA ∧ prob_B_cityA ≤ 1) : 
  prob_A_cityA * (1 - prob_B_cityA) + (1 - prob_A_cityA) * prob_B_cityA = 0.54 := by
  sorry

end NUMINAMATH_CALUDE_prob_different_cities_l3227_322783


namespace NUMINAMATH_CALUDE_complex_number_problem_l3227_322741

theorem complex_number_problem (z : ℂ) :
  Complex.abs z = 5 ∧ (Complex.I * Complex.im ((3 + 4 * Complex.I) * z) = (3 + 4 * Complex.I) * z) →
  z = 4 + 3 * Complex.I ∨ z = -4 - 3 * Complex.I :=
by sorry

end NUMINAMATH_CALUDE_complex_number_problem_l3227_322741


namespace NUMINAMATH_CALUDE_min_value_problem_l3227_322725

theorem min_value_problem (a b c d e f g h : ℝ) 
  (h1 : a * b * c * d = 16) 
  (h2 : e * f * g * h = 36) : 
  (a*e)^2 + (b*f)^2 + (c*g)^2 + (d*h)^2 ≥ 576 := by
  sorry

end NUMINAMATH_CALUDE_min_value_problem_l3227_322725


namespace NUMINAMATH_CALUDE_brothers_age_difference_l3227_322780

/-- Bush and Matt are brothers with an age difference --/
def age_difference (bush_age : ℕ) (matt_future_age : ℕ) (years_to_future : ℕ) : ℕ :=
  (matt_future_age - years_to_future) - bush_age

/-- Theorem stating the age difference between Matt and Bush --/
theorem brothers_age_difference :
  age_difference 12 25 10 = 3 := by
  sorry

end NUMINAMATH_CALUDE_brothers_age_difference_l3227_322780


namespace NUMINAMATH_CALUDE_megan_popsicle_consumption_l3227_322709

/-- The number of Popsicles Megan can finish in 5 hours -/
def popsicles_in_5_hours : ℕ := 15

/-- The time in minutes it takes Megan to eat one Popsicle -/
def minutes_per_popsicle : ℕ := 20

/-- The number of hours given in the problem -/
def hours : ℕ := 5

/-- Theorem stating that Megan can finish 15 Popsicles in 5 hours -/
theorem megan_popsicle_consumption :
  popsicles_in_5_hours = (hours * 60) / minutes_per_popsicle :=
by sorry

end NUMINAMATH_CALUDE_megan_popsicle_consumption_l3227_322709


namespace NUMINAMATH_CALUDE_complex_equation_solution_l3227_322754

theorem complex_equation_solution (z : ℂ) : 
  Complex.I * z = 1 - 2 * Complex.I → z = 2 - Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l3227_322754


namespace NUMINAMATH_CALUDE_third_player_games_l3227_322773

/-- Represents a chess tournament with three players. -/
structure ChessTournament where
  total_games : ℕ
  player1_games : ℕ
  player2_games : ℕ
  player3_games : ℕ

/-- The theorem stating the number of games played by the third player. -/
theorem third_player_games (t : ChessTournament) 
  (h1 : t.total_games = 27)
  (h2 : t.player1_games = 13)
  (h3 : t.player2_games = 27)
  (h4 : t.player1_games + t.player2_games + t.player3_games = 2 * t.total_games) :
  t.player3_games = 14 := by
  sorry


end NUMINAMATH_CALUDE_third_player_games_l3227_322773


namespace NUMINAMATH_CALUDE_prob_ride_all_cars_l3227_322747

/-- The number of cars in the roller coaster -/
def num_cars : ℕ := 4

/-- The number of times the passenger rides the roller coaster -/
def num_rides : ℕ := 4

/-- The probability of choosing any specific car for a single ride -/
def prob_single_car : ℚ := 1 / num_cars

/-- The probability of riding in each of the 4 cars exactly once in 4 rides -/
def prob_all_cars : ℚ := 3 / 32

/-- Theorem stating that the probability of riding in each car exactly once is 3/32 -/
theorem prob_ride_all_cars : 
  prob_all_cars = (num_cars.factorial : ℚ) / num_cars ^ num_rides :=
sorry

end NUMINAMATH_CALUDE_prob_ride_all_cars_l3227_322747


namespace NUMINAMATH_CALUDE_compare_fractions_l3227_322764

theorem compare_fractions : -4/3 < -5/4 := by
  sorry

end NUMINAMATH_CALUDE_compare_fractions_l3227_322764


namespace NUMINAMATH_CALUDE_perpendicular_condition_l3227_322788

-- Define the lines l₁ and l₂
def l₁ (x y a : ℝ) : Prop := x + a * y - 2 = 0
def l₂ (x y a : ℝ) : Prop := x - a * y - 1 = 0

-- Define perpendicularity of lines
def perpendicular (a : ℝ) : Prop := 1 + a * (-a) = 0

-- Define sufficient condition
def sufficient (P Q : Prop) : Prop := P → Q

-- Define necessary condition
def necessary (P Q : Prop) : Prop := Q → P

theorem perpendicular_condition (a : ℝ) :
  sufficient (a = -1) (perpendicular a) ∧
  ¬ necessary (a = -1) (perpendicular a) :=
by sorry

end NUMINAMATH_CALUDE_perpendicular_condition_l3227_322788


namespace NUMINAMATH_CALUDE_hilt_current_rocks_l3227_322792

/-- The number of rocks Mrs. Hilt needs to complete the border -/
def total_rocks_needed : ℕ := 125

/-- The number of additional rocks Mrs. Hilt needs -/
def additional_rocks_needed : ℕ := 61

/-- The number of rocks Mrs. Hilt currently has -/
def current_rocks : ℕ := total_rocks_needed - additional_rocks_needed

theorem hilt_current_rocks :
  current_rocks = 64 :=
sorry

end NUMINAMATH_CALUDE_hilt_current_rocks_l3227_322792


namespace NUMINAMATH_CALUDE_memorial_day_weather_probability_l3227_322774

/-- The probability of exactly k successes in n independent Bernoulli trials --/
def binomial_probability (n k : ℕ) (p : ℝ) : ℝ :=
  (n.choose k : ℝ) * p^k * (1 - p)^(n - k)

/-- The number of days in the Memorial Day weekend --/
def num_days : ℕ := 5

/-- The probability of rain on each day --/
def rain_probability : ℝ := 0.8

/-- The number of desired sunny days --/
def desired_sunny_days : ℕ := 2

theorem memorial_day_weather_probability :
  binomial_probability num_days desired_sunny_days (1 - rain_probability) = 51 / 250 := by
  sorry

end NUMINAMATH_CALUDE_memorial_day_weather_probability_l3227_322774


namespace NUMINAMATH_CALUDE_muffin_banana_price_ratio_l3227_322731

/-- The price ratio of a muffin to a banana is 2 -/
theorem muffin_banana_price_ratio :
  ∀ (m b S : ℝ),
  (3 * m + 5 * b = S) →
  (5 * m + 7 * b = 3 * S) →
  m = 2 * b :=
by sorry

end NUMINAMATH_CALUDE_muffin_banana_price_ratio_l3227_322731


namespace NUMINAMATH_CALUDE_equality_or_sum_zero_l3227_322734

theorem equality_or_sum_zero (a b c d : ℝ) :
  (a + b) / (b + c) = (c + d) / (d + a) →
  (a = c ∨ a + b + c + d = 0) :=
by sorry

end NUMINAMATH_CALUDE_equality_or_sum_zero_l3227_322734


namespace NUMINAMATH_CALUDE_cubic_root_sum_l3227_322767

theorem cubic_root_sum (a b c : ℝ) : 
  a^3 - 6*a^2 + 11*a - 6 = 0 →
  b^3 - 6*b^2 + 11*b - 6 = 0 →
  c^3 - 6*c^2 + 11*c - 6 = 0 →
  a*b/c + b*c/a + c*a/b = 49/6 := by
sorry

end NUMINAMATH_CALUDE_cubic_root_sum_l3227_322767


namespace NUMINAMATH_CALUDE_a_lt_neg_four_sufficient_not_necessary_l3227_322785

/-- The function f(x) = ax + 3 -/
def f (a : ℝ) (x : ℝ) : ℝ := a * x + 3

/-- The condition for f to have a zero point on [-1,1] -/
def has_zero_point (a : ℝ) : Prop := ∃ x : ℝ, x ∈ Set.Icc (-1) 1 ∧ f a x = 0

/-- The statement that a < -4 is sufficient but not necessary for f to have a zero point on [-1,1] -/
theorem a_lt_neg_four_sufficient_not_necessary :
  (∀ a : ℝ, a < -4 → has_zero_point a) ∧
  ¬(∀ a : ℝ, has_zero_point a → a < -4) :=
sorry

end NUMINAMATH_CALUDE_a_lt_neg_four_sufficient_not_necessary_l3227_322785


namespace NUMINAMATH_CALUDE_skidding_distance_speed_relation_l3227_322735

theorem skidding_distance_speed_relation 
  (a b : ℝ) 
  (h1 : b = a * 60^2) 
  (h2 : 3 * b = a * x^2) : 
  x = 60 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_skidding_distance_speed_relation_l3227_322735


namespace NUMINAMATH_CALUDE_transform_point_l3227_322763

/-- Rotate a point 90 degrees clockwise around a center point -/
def rotate90Clockwise (p : ℝ × ℝ) (center : ℝ × ℝ) : ℝ × ℝ :=
  let (x, y) := p
  let (cx, cy) := center
  (cx + (y - cy), cy - (x - cx))

/-- Reflect a point over the x-axis -/
def reflectOverX (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.1, -p.2)

/-- The main theorem -/
theorem transform_point :
  let A : ℝ × ℝ := (-4, 1)
  let center : ℝ × ℝ := (1, 1)
  let rotated := rotate90Clockwise A center
  let final := reflectOverX rotated
  final = (1, -6) := by sorry

end NUMINAMATH_CALUDE_transform_point_l3227_322763


namespace NUMINAMATH_CALUDE_unique_two_digit_integer_l3227_322740

theorem unique_two_digit_integer (t : ℕ) : 
  (t ≥ 10 ∧ t < 100) ∧ (13 * t) % 100 = 45 ↔ t = 65 := by
  sorry

end NUMINAMATH_CALUDE_unique_two_digit_integer_l3227_322740


namespace NUMINAMATH_CALUDE_solve_system_of_equations_l3227_322794

theorem solve_system_of_equations :
  ∃ (x y : ℝ), (x / 6) * 12 = 11 ∧ 4 * (x - y) + 5 = 11 ∧ x = 5.5 ∧ y = 4 := by
  sorry

end NUMINAMATH_CALUDE_solve_system_of_equations_l3227_322794


namespace NUMINAMATH_CALUDE_inequality_proof_l3227_322755

theorem inequality_proof (x y z w : ℝ) 
  (h_positive : x > 0 ∧ y > 0 ∧ z > 0 ∧ w > 0) 
  (h_sum_squares : x^2 + y^2 + z^2 + w^2 = 1) : 
  x^2 * y * z * w + x * y^2 * z * w + x * y * z^2 * w + x * y * z * w^2 ≤ 1/8 := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l3227_322755


namespace NUMINAMATH_CALUDE_animals_equal_humps_l3227_322796

/-- Represents the number of animals of each type in the herd -/
structure Herd where
  horses : ℕ
  oneHumpCamels : ℕ
  twoHumpCamels : ℕ

/-- Calculates the total number of humps in the herd -/
def totalHumps (h : Herd) : ℕ :=
  h.oneHumpCamels + 2 * h.twoHumpCamels

/-- Calculates the total number of animals in the herd -/
def totalAnimals (h : Herd) : ℕ :=
  h.horses + h.oneHumpCamels + h.twoHumpCamels

/-- Theorem stating that under the given conditions, the total number of animals equals the total number of humps -/
theorem animals_equal_humps (h : Herd) 
    (hump_count : totalHumps h = 200) 
    (equal_horses_twohumps : h.horses = h.twoHumpCamels) : 
  totalAnimals h = 200 := by
  sorry


end NUMINAMATH_CALUDE_animals_equal_humps_l3227_322796


namespace NUMINAMATH_CALUDE_choir_robe_expenditure_is_36_l3227_322700

/-- Calculates the total expenditure for additional choir robes. -/
def choir_robe_expenditure (total_singers : ℕ) (existing_robes : ℕ) (cost_per_robe : ℕ) : ℕ :=
  (total_singers - existing_robes) * cost_per_robe

/-- Proves that the expenditure for additional choir robes is $36 given the specified conditions. -/
theorem choir_robe_expenditure_is_36 :
  choir_robe_expenditure 30 12 2 = 36 := by
  sorry

end NUMINAMATH_CALUDE_choir_robe_expenditure_is_36_l3227_322700


namespace NUMINAMATH_CALUDE_square_not_end_two_odd_digits_l3227_322762

theorem square_not_end_two_odd_digits (n : ℕ) : 
  ∃ (d₁ d₂ : ℕ), d₁ < 10 ∧ d₂ < 10 ∧ n^2 % 100 = 10 * d₁ + d₂ → (d₁ % 2 = 0 ∨ d₂ % 2 = 0) :=
sorry

end NUMINAMATH_CALUDE_square_not_end_two_odd_digits_l3227_322762


namespace NUMINAMATH_CALUDE_odd_function_property_l3227_322727

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

theorem odd_function_property (f : ℝ → ℝ) (h1 : is_odd_function f) (h2 : f 3 - f 2 = 1) :
  f (-2) - f (-3) = 1 := by sorry

end NUMINAMATH_CALUDE_odd_function_property_l3227_322727


namespace NUMINAMATH_CALUDE_smallest_odd_with_three_prime_factors_l3227_322771

-- Define a function to check if a number has exactly three distinct prime factors
def has_three_distinct_prime_factors (n : ℕ) : Prop :=
  ∃ p q r : ℕ, Nat.Prime p ∧ Nat.Prime q ∧ Nat.Prime r ∧
  p ≠ q ∧ p ≠ r ∧ q ≠ r ∧
  n = p * q * r

-- State the theorem
theorem smallest_odd_with_three_prime_factors :
  (∀ m : ℕ, m < 105 → m % 2 = 1 → ¬(has_three_distinct_prime_factors m)) ∧
  (105 % 2 = 1) ∧
  (has_three_distinct_prime_factors 105) :=
sorry

end NUMINAMATH_CALUDE_smallest_odd_with_three_prime_factors_l3227_322771


namespace NUMINAMATH_CALUDE_anna_apple_count_l3227_322769

/-- The number of apples Anna ate on Tuesday -/
def tuesday_apples : ℕ := 4

/-- The number of apples Anna ate on Wednesday -/
def wednesday_apples : ℕ := 2 * tuesday_apples

/-- The number of apples Anna ate on Thursday -/
def thursday_apples : ℕ := tuesday_apples / 2

/-- The total number of apples Anna ate over the three days -/
def total_apples : ℕ := tuesday_apples + wednesday_apples + thursday_apples

theorem anna_apple_count : total_apples = 14 := by
  sorry

end NUMINAMATH_CALUDE_anna_apple_count_l3227_322769


namespace NUMINAMATH_CALUDE_rectangle_area_increase_l3227_322719

theorem rectangle_area_increase : 
  let original_length : ℕ := 13
  let original_width : ℕ := 10
  let increase : ℕ := 2
  let original_area := original_length * original_width
  let new_length := original_length + increase
  let new_width := original_width + increase
  let new_area := new_length * new_width
  new_area - original_area = 50 := by sorry

end NUMINAMATH_CALUDE_rectangle_area_increase_l3227_322719


namespace NUMINAMATH_CALUDE_unique_n_for_divisibility_by_15_l3227_322761

def is_divisible_by (a b : ℕ) : Prop := ∃ k, a = b * k

theorem unique_n_for_divisibility_by_15 : 
  ∃! n : ℕ, n < 10 ∧ is_divisible_by (80000 + 10000 * n + 945) 15 :=
sorry

end NUMINAMATH_CALUDE_unique_n_for_divisibility_by_15_l3227_322761


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l3227_322733

def set_A : Set ℝ := {x | x^2 - 4 > 0}
def set_B : Set ℝ := {x | x + 2 < 0}

theorem intersection_of_A_and_B :
  set_A ∩ set_B = {x : ℝ | x < -2} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l3227_322733


namespace NUMINAMATH_CALUDE_point_in_fourth_quadrant_implies_m_greater_than_two_l3227_322702

/-- A point in the 2D plane -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Determines if a point is in the fourth quadrant -/
def isInFourthQuadrant (p : Point2D) : Prop :=
  p.x > 0 ∧ p.y < 0

/-- The point P as a function of m -/
def P (m : ℝ) : Point2D :=
  { x := m - 1, y := 2 - m }

theorem point_in_fourth_quadrant_implies_m_greater_than_two :
  ∀ m : ℝ, isInFourthQuadrant (P m) → m > 2 := by
  sorry

end NUMINAMATH_CALUDE_point_in_fourth_quadrant_implies_m_greater_than_two_l3227_322702


namespace NUMINAMATH_CALUDE_bridge_length_l3227_322786

/-- The length of a bridge given train specifications and crossing time -/
theorem bridge_length (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time : ℝ) :
  train_length = 160 ∧ 
  train_speed_kmh = 45 ∧ 
  crossing_time = 30 →
  ∃ (bridge_length : ℝ), bridge_length = 215 := by
  sorry

end NUMINAMATH_CALUDE_bridge_length_l3227_322786


namespace NUMINAMATH_CALUDE_smallest_number_with_remainder_l3227_322756

theorem smallest_number_with_remainder (n : ℤ) : 
  (n % 5 = 2) ∧ 
  ((n + 1) % 5 = 2) ∧ 
  ((n + 2) % 5 = 2) ∧ 
  (n + (n + 1) + (n + 2) = 336) →
  n = 107 := by
sorry

end NUMINAMATH_CALUDE_smallest_number_with_remainder_l3227_322756


namespace NUMINAMATH_CALUDE_integral_sum_equals_pi_over_four_plus_ln_two_l3227_322766

theorem integral_sum_equals_pi_over_four_plus_ln_two :
  (∫ (x : ℝ) in (0)..(1), Real.sqrt (1 - x^2)) + (∫ (x : ℝ) in (1)..(2), 1/x) = π/4 + Real.log 2 := by
  sorry

end NUMINAMATH_CALUDE_integral_sum_equals_pi_over_four_plus_ln_two_l3227_322766


namespace NUMINAMATH_CALUDE_mistaken_multiplication_l3227_322790

/-- Given a polynomial P(x) such that (-3x) * P(x) = 3x³ - 3x² + 3x,
    prove that P(x) - 3x = -x² - 2x - 1 -/
theorem mistaken_multiplication (P : ℝ → ℝ) :
  (∀ x, (-3 * x) * P x = 3 * x^3 - 3 * x^2 + 3 * x) →
  (∀ x, P x - 3 * x = -x^2 - 2 * x - 1) :=
by sorry

end NUMINAMATH_CALUDE_mistaken_multiplication_l3227_322790


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l3227_322718

theorem sufficient_not_necessary_condition (a : ℝ) : 
  (∀ x, x > 1 → x > a) ∧ (∃ x, x > a ∧ x ≤ 1) → a < 1 := by
  sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l3227_322718


namespace NUMINAMATH_CALUDE_student_average_score_l3227_322759

theorem student_average_score (M P C : ℕ) : 
  M + P = 50 → C = P + 20 → (M + C) / 2 = 35 := by
  sorry

end NUMINAMATH_CALUDE_student_average_score_l3227_322759


namespace NUMINAMATH_CALUDE_solutions_exist_and_finite_l3227_322793

theorem solutions_exist_and_finite :
  ∃ (n : ℕ) (S : Finset ℝ),
    (∀ θ ∈ S, 0 < θ ∧ θ < 2 * Real.pi) ∧
    (∀ θ ∈ S, Real.sin (7 * Real.pi * Real.cos θ) = Real.cos (7 * Real.pi * Real.sin θ)) ∧
    S.card = n :=
by sorry

end NUMINAMATH_CALUDE_solutions_exist_and_finite_l3227_322793


namespace NUMINAMATH_CALUDE_average_games_per_month_l3227_322751

def total_games : ℕ := 323
def season_months : ℕ := 19

theorem average_games_per_month :
  (total_games : ℚ) / season_months = 17 := by sorry

end NUMINAMATH_CALUDE_average_games_per_month_l3227_322751


namespace NUMINAMATH_CALUDE_least_prime_factor_of_5_4_minus_5_3_l3227_322744

theorem least_prime_factor_of_5_4_minus_5_3 :
  Nat.minFac (5^4 - 5^3) = 2 := by
  sorry

end NUMINAMATH_CALUDE_least_prime_factor_of_5_4_minus_5_3_l3227_322744


namespace NUMINAMATH_CALUDE_badminton_players_count_l3227_322720

/-- Represents a sports club with members playing badminton and tennis -/
structure SportsClub where
  total : ℕ
  tennis : ℕ
  neither : ℕ
  both : ℕ

/-- Calculates the number of members playing badminton in a sports club -/
def badminton_players (club : SportsClub) : ℕ :=
  club.total - club.neither - (club.tennis - club.both)

/-- Theorem stating the number of badminton players in the given sports club -/
theorem badminton_players_count (club : SportsClub) 
  (h_total : club.total = 30)
  (h_tennis : club.tennis = 19)
  (h_neither : club.neither = 3)
  (h_both : club.both = 9) :
  badminton_players club = 17 := by
  sorry

#eval badminton_players { total := 30, tennis := 19, neither := 3, both := 9 }

end NUMINAMATH_CALUDE_badminton_players_count_l3227_322720


namespace NUMINAMATH_CALUDE_exists_cubic_positive_l3227_322706

theorem exists_cubic_positive : ∃ x : ℝ, x^3 - x^2 + 1 > 0 := by sorry

end NUMINAMATH_CALUDE_exists_cubic_positive_l3227_322706


namespace NUMINAMATH_CALUDE_trapezoid_bases_solutions_l3227_322760

theorem trapezoid_bases_solutions :
  let area : ℕ := 1800
  let altitude : ℕ := 60
  let base_sum : ℕ := 2 * area / altitude
  let valid_base_pair := λ b₁ b₂ : ℕ =>
    b₁ % 10 = 0 ∧ b₂ % 10 = 0 ∧ b₁ + b₂ = base_sum
  (∃! (solutions : Finset (ℕ × ℕ)), solutions.card = 4 ∧
    ∀ pair : ℕ × ℕ, pair ∈ solutions ↔ valid_base_pair pair.1 pair.2) :=
by sorry

end NUMINAMATH_CALUDE_trapezoid_bases_solutions_l3227_322760


namespace NUMINAMATH_CALUDE_F_10_squares_l3227_322715

/-- Represents the number of squares in figure F_n -/
def num_squares (n : ℕ) : ℕ :=
  1 + 3 * (n - 1) * n

/-- The theorem stating that F_10 contains 271 squares -/
theorem F_10_squares : num_squares 10 = 271 := by
  sorry

end NUMINAMATH_CALUDE_F_10_squares_l3227_322715


namespace NUMINAMATH_CALUDE_product_equals_3408_decimal_product_l3227_322714

theorem product_equals_3408 : 213 * 16 = 3408 := by
  sorry

-- Additional fact (not used in the proof)
theorem decimal_product : 0.16 * 2.13 = 0.3408 := by
  sorry

end NUMINAMATH_CALUDE_product_equals_3408_decimal_product_l3227_322714


namespace NUMINAMATH_CALUDE_range_of_m_l3227_322758

theorem range_of_m (m : ℝ) : 
  (∃ x : ℝ, x ∈ Set.Icc 2 4 ∧ x^2 - 2*x + 5 - m < 0) → m > 5 := by
  sorry

end NUMINAMATH_CALUDE_range_of_m_l3227_322758
