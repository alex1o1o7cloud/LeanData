import Mathlib

namespace correct_d_value_l34_3466

/-- The exchange rate from U.S. dollars to Mexican pesos -/
def exchange_rate : ℚ := 13 / 9

/-- The amount of pesos spent -/
def pesos_spent : ℕ := 117

/-- The function that calculates the remaining pesos after exchange and spending -/
def remaining_pesos (d : ℕ) : ℚ := exchange_rate * d - pesos_spent

/-- The theorem stating that 264 is the correct value for d -/
theorem correct_d_value : ∃ (d : ℕ), d = 264 ∧ remaining_pesos d = d := by sorry

end correct_d_value_l34_3466


namespace one_fourth_of_8_4_l34_3465

theorem one_fourth_of_8_4 : (8.4 : ℚ) / 4 = 21 / 10 := by
  sorry

end one_fourth_of_8_4_l34_3465


namespace num_regions_convex_ngon_l34_3432

/-- A convex n-gon is a polygon with n sides where all interior angles are less than 180 degrees. -/
structure ConvexNGon (n : ℕ) where
  -- Add necessary fields here
  n_ge_3 : n ≥ 3

/-- The number of regions formed by the diagonals of a convex n-gon where no three diagonals intersect at a single interior point. -/
def num_regions (n : ℕ) : ℕ := Nat.choose n 4 + Nat.choose (n - 1) 2

/-- Theorem stating that the number of regions formed by the diagonals of a convex n-gon
    where no three diagonals intersect at a single interior point is C_n^4 + C_{n-1}^2. -/
theorem num_regions_convex_ngon (n : ℕ) (polygon : ConvexNGon n) :
  num_regions n = Nat.choose n 4 + Nat.choose (n - 1) 2 := by
  sorry

#check num_regions_convex_ngon

end num_regions_convex_ngon_l34_3432


namespace sports_and_literature_enthusiasts_l34_3439

theorem sports_and_literature_enthusiasts
  (total_students : ℕ)
  (sports_enthusiasts : ℕ)
  (literature_enthusiasts : ℕ)
  (h_total : total_students = 100)
  (h_sports : sports_enthusiasts = 60)
  (h_literature : literature_enthusiasts = 65) :
  ∃ (m n : ℕ),
    m = max sports_enthusiasts literature_enthusiasts ∧
    n = max 0 (sports_enthusiasts + literature_enthusiasts - total_students) ∧
    m + n = 85 :=
by sorry

end sports_and_literature_enthusiasts_l34_3439


namespace line_tangent_to_circle_l34_3400

/-- A line y = 2x + a is tangent to the circle x^2 + y^2 = 9 if and only if a = ±3√5 -/
theorem line_tangent_to_circle (a : ℝ) : 
  (∀ x y : ℝ, y = 2*x + a ∧ x^2 + y^2 = 9 → (∀ ε > 0, ∃ δ > 0, ∀ x' y', 
    x'^2 + y'^2 = 9 → (x' - x)^2 + (y' - y)^2 < δ^2 → y' ≠ 2*x' + a)) ↔ 
  a = 3 * Real.sqrt 5 ∨ a = -3 * Real.sqrt 5 :=
sorry

end line_tangent_to_circle_l34_3400


namespace novosibirsk_divisible_by_three_l34_3446

/-- Represents a mapping from letters to digits -/
def LetterToDigitMap := Char → Nat

/-- Checks if a mapping is valid for the word "NOVOSIBIRSK" -/
def isValidMapping (m : LetterToDigitMap) : Prop :=
  m 'N' ≠ m 'O' ∧ m 'N' ≠ m 'V' ∧ m 'N' ≠ m 'S' ∧ m 'N' ≠ m 'I' ∧ m 'N' ≠ m 'B' ∧ m 'N' ≠ m 'R' ∧ m 'N' ≠ m 'K' ∧
  m 'O' ≠ m 'V' ∧ m 'O' ≠ m 'S' ∧ m 'O' ≠ m 'I' ∧ m 'O' ≠ m 'B' ∧ m 'O' ≠ m 'R' ∧ m 'O' ≠ m 'K' ∧
  m 'V' ≠ m 'S' ∧ m 'V' ≠ m 'I' ∧ m 'V' ≠ m 'B' ∧ m 'V' ≠ m 'R' ∧ m 'V' ≠ m 'K' ∧
  m 'S' ≠ m 'I' ∧ m 'S' ≠ m 'B' ∧ m 'S' ≠ m 'R' ∧ m 'S' ≠ m 'K' ∧
  m 'I' ≠ m 'B' ∧ m 'I' ≠ m 'R' ∧ m 'I' ≠ m 'K' ∧
  m 'B' ≠ m 'R' ∧ m 'B' ≠ m 'K' ∧
  m 'R' ≠ m 'K'

/-- Calculates the sum of digits for "NOVOSIBIRSK" using the given mapping -/
def sumOfDigits (m : LetterToDigitMap) : Nat :=
  m 'N' + m 'O' + m 'V' + m 'O' + m 'S' + m 'I' + m 'B' + m 'I' + m 'R' + m 'S' + m 'K'

/-- Theorem: There exists a valid mapping for "NOVOSIBIRSK" that results in a number divisible by 3 -/
theorem novosibirsk_divisible_by_three : ∃ (m : LetterToDigitMap), isValidMapping m ∧ sumOfDigits m % 3 = 0 := by
  sorry

end novosibirsk_divisible_by_three_l34_3446


namespace jason_shopping_total_l34_3463

theorem jason_shopping_total (jacket_cost shorts_cost : ℚ) 
  (h1 : jacket_cost = 4.74)
  (h2 : shorts_cost = 9.54) :
  jacket_cost + shorts_cost = 14.28 := by
  sorry

end jason_shopping_total_l34_3463


namespace parabola_vertex_l34_3497

/-- The parabola equation -/
def parabola_equation (x y : ℝ) : Prop := y = (x - 6)^2 + 3

/-- The vertex of the parabola -/
def vertex : ℝ × ℝ := (6, 3)

/-- Theorem: The vertex of the parabola y = (x - 6)^2 + 3 is at (6, 3) -/
theorem parabola_vertex :
  ∀ x y : ℝ, parabola_equation x y → (x, y) = vertex :=
sorry

end parabola_vertex_l34_3497


namespace min_distance_to_line_l34_3495

-- Define the vectors
def a : ℝ × ℝ := (1, 0)
def b : ℝ × ℝ := (0, 1)

-- Define the theorem
theorem min_distance_to_line 
  (m n : ℝ) 
  (h : (a.1 - m) * (-m) + (a.2 - n) * (b.2 - n) = 0) : 
  ∃ (d : ℝ), d = Real.sqrt 2 / 2 ∧ 
  ∀ (x y : ℝ), x + y + 1 = 0 → 
  Real.sqrt ((x - m)^2 + (y - n)^2) ≥ d :=
sorry

end min_distance_to_line_l34_3495


namespace faster_train_speed_l34_3481

/-- Proves that the speed of the faster train is 50 km/hr given the conditions of the problem -/
theorem faster_train_speed
  (train_length : ℝ)
  (slower_speed : ℝ)
  (passing_time : ℝ)
  (h1 : train_length = 70)
  (h2 : slower_speed = 36)
  (h3 : passing_time = 36)
  : ∃ (faster_speed : ℝ), faster_speed = 50 ∧ 
    (faster_speed - slower_speed) * (1000 / 3600) * passing_time = 2 * train_length :=
by sorry

end faster_train_speed_l34_3481


namespace ali_remaining_money_l34_3451

def calculate_remaining_money (initial_amount : ℚ) : ℚ :=
  let after_food := initial_amount * (1 - 3/8)
  let after_glasses := after_food * (1 - 2/5)
  let after_gift := after_glasses * (1 - 1/4)
  after_gift

theorem ali_remaining_money :
  calculate_remaining_money 480 = 135 := by
  sorry

end ali_remaining_money_l34_3451


namespace wendys_score_l34_3424

/-- The score for each treasure found in the game. -/
def points_per_treasure : ℕ := 5

/-- The number of treasures Wendy found on the first level. -/
def treasures_level1 : ℕ := 4

/-- The number of treasures Wendy found on the second level. -/
def treasures_level2 : ℕ := 3

/-- Wendy's total score in the game. -/
def total_score : ℕ := points_per_treasure * (treasures_level1 + treasures_level2)

/-- Theorem stating that Wendy's total score is 35 points. -/
theorem wendys_score : total_score = 35 := by
  sorry

end wendys_score_l34_3424


namespace high_school_students_l34_3476

/-- The number of students in a high school, given information about music and art classes -/
theorem high_school_students (music_students : ℕ) (art_students : ℕ) (both_students : ℕ) (neither_students : ℕ)
  (h1 : music_students = 20)
  (h2 : art_students = 20)
  (h3 : both_students = 10)
  (h4 : neither_students = 470) :
  music_students + art_students - both_students + neither_students = 500 :=
by sorry

end high_school_students_l34_3476


namespace best_sampling_methods_l34_3488

/-- Represents different income levels of families -/
inductive IncomeLevel
  | High
  | Middle
  | Low

/-- Represents different sampling methods -/
inductive SamplingMethod
  | SimpleRandom
  | Stratified
  | Systematic

/-- Structure representing a community -/
structure Community where
  total_families : ℕ
  high_income : ℕ
  middle_income : ℕ
  low_income : ℕ
  sample_size : ℕ

/-- Structure representing a group of student-athletes -/
structure StudentAthleteGroup where
  total_athletes : ℕ
  sample_size : ℕ

/-- Function to determine the best sampling method for a community survey -/
def best_community_sampling_method (c : Community) : SamplingMethod := sorry

/-- Function to determine the best sampling method for a student-athlete survey -/
def best_student_athlete_sampling_method (g : StudentAthleteGroup) : SamplingMethod := sorry

/-- Theorem stating the best sampling methods for the given scenarios -/
theorem best_sampling_methods 
  (community : Community) 
  (student_athletes : StudentAthleteGroup) : 
  community.total_families = 500 ∧ 
  community.high_income = 125 ∧ 
  community.middle_income = 280 ∧ 
  community.low_income = 95 ∧ 
  community.sample_size = 100 ∧
  student_athletes.total_athletes = 12 ∧ 
  student_athletes.sample_size = 3 →
  best_community_sampling_method community = SamplingMethod.Stratified ∧
  best_student_athlete_sampling_method student_athletes = SamplingMethod.SimpleRandom := by
  sorry

end best_sampling_methods_l34_3488


namespace existence_of_six_snakes_l34_3445

/-- A point in the plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- A snake is a polyline with 5 segments connecting 6 points -/
structure Snake where
  points : Fin 6 → Point
  is_valid : Bool

/-- Check if two snakes are different -/
def are_different_snakes (s1 s2 : Snake) : Bool :=
  sorry

/-- Check if a snake satisfies the angle condition -/
def satisfies_angle_condition (s : Snake) : Bool :=
  sorry

/-- Check if a snake satisfies the half-plane condition -/
def satisfies_half_plane_condition (s : Snake) : Bool :=
  sorry

/-- The main theorem stating that a configuration of 6 points exists
    that can form 6 different valid snakes -/
theorem existence_of_six_snakes :
  ∃ (points : Fin 6 → Point),
    ∃ (snakes : Fin 6 → Snake),
      (∀ i : Fin 6, (snakes i).points = points) ∧
      (∀ i : Fin 6, (snakes i).is_valid) ∧
      (∀ i j : Fin 6, i ≠ j → are_different_snakes (snakes i) (snakes j)) ∧
      (∀ i : Fin 6, satisfies_angle_condition (snakes i)) ∧
      (∀ i : Fin 6, satisfies_half_plane_condition (snakes i)) :=
  sorry

end existence_of_six_snakes_l34_3445


namespace max_profit_selling_price_l34_3493

-- Define the profit function
def profit (x : ℝ) : ℝ := 10 * (-x^2 + 140*x - 4000)

-- Define the theorem
theorem max_profit_selling_price :
  -- Given conditions
  let cost_price : ℝ := 40
  let initial_price : ℝ := 50
  let initial_sales : ℝ := 500
  let price_sensitivity : ℝ := 10

  -- Theorem statement
  ∃ (max_price max_profit : ℝ),
    -- The maximum price is greater than the cost price
    max_price > cost_price ∧
    -- The maximum profit occurs at the maximum price
    profit max_price = max_profit ∧
    -- The maximum profit is indeed the maximum
    ∀ x > cost_price, profit x ≤ max_profit ∧
    -- The specific values for maximum price and profit
    max_price = 70 ∧ max_profit = 9000 := by
  sorry

end max_profit_selling_price_l34_3493


namespace equation_holds_for_all_y_l34_3413

theorem equation_holds_for_all_y (x : ℝ) : 
  (∀ y : ℝ, 10 * x * y - 15 * y + 5 * x - 7 = 0) ↔ x = 3/2 := by
sorry

end equation_holds_for_all_y_l34_3413


namespace pizza_price_l34_3415

theorem pizza_price (num_pizzas : ℕ) (tip : ℝ) (bill : ℝ) (change : ℝ) :
  num_pizzas = 4 ∧ tip = 5 ∧ bill = 50 ∧ change = 5 →
  ∃ (price : ℝ), price = 10 ∧ num_pizzas * price + tip = bill - change :=
by sorry

end pizza_price_l34_3415


namespace ndfl_calculation_l34_3467

/-- Calculates the personal income tax (NDFL) for a Russian resident --/
def calculate_ndfl (monthly_income : ℚ) (bonus : ℚ) (car_sale : ℚ) (land_purchase : ℚ) : ℚ :=
  let annual_income := monthly_income * 12 + bonus + car_sale
  let total_deductions := car_sale + land_purchase
  let taxable_income := max (annual_income - total_deductions) 0
  let tax_rate := 13 / 100
  taxable_income * tax_rate

/-- Theorem stating that the NDFL for the given conditions is 10400 rubles --/
theorem ndfl_calculation :
  calculate_ndfl 30000 20000 250000 300000 = 10400 := by
  sorry

end ndfl_calculation_l34_3467


namespace imaginary_part_of_complex_fraction_l34_3427

theorem imaginary_part_of_complex_fraction (i : ℂ) (h : i^2 = -1) :
  let z := (3 + i) / (1 - i)
  Complex.im z = 2 := by sorry

end imaginary_part_of_complex_fraction_l34_3427


namespace imaginary_part_of_symmetrical_complex_ratio_l34_3492

theorem imaginary_part_of_symmetrical_complex_ratio :
  ∀ (z₁ z₂ : ℂ),
  z₁ = 1 - 2*I →
  (z₂.re = -z₁.re ∧ z₂.im = z₁.im) →
  Complex.im (z₂ / z₁) = -4/5 := by
sorry

end imaginary_part_of_symmetrical_complex_ratio_l34_3492


namespace area_of_intersection_l34_3449

/-- Given two overlapping rectangles ABNF and CMKD, prove the area of their intersection MNFK --/
theorem area_of_intersection (BN KD : ℝ) (area_ABMK area_CDFN : ℝ) :
  BN = 8 →
  KD = 9 →
  area_ABMK = 25 →
  area_CDFN = 32 →
  ∃ (AB CD : ℝ),
    AB * BN - area_ABMK = CD * KD - area_CDFN ∧
    AB * BN - area_ABMK = 31 :=
by sorry

end area_of_intersection_l34_3449


namespace river_depth_l34_3453

/-- Proves that given a river with specified width, flow rate, and volume of water flowing into the sea per minute, the depth of the river is 5 meters. -/
theorem river_depth 
  (width : ℝ) 
  (flow_rate_kmph : ℝ) 
  (volume_per_minute : ℝ) 
  (h1 : width = 35) 
  (h2 : flow_rate_kmph = 2) 
  (h3 : volume_per_minute = 5833.333333333333) : 
  (volume_per_minute / (flow_rate_kmph * 1000 / 60 * width)) = 5 := by
  sorry

end river_depth_l34_3453


namespace pizza_slice_angle_l34_3438

theorem pizza_slice_angle (p : ℝ) (h1 : p > 0) (h2 : p < 1) (h3 : p = 1/8) :
  let angle := p * 360
  angle = 45 := by sorry

end pizza_slice_angle_l34_3438


namespace only_C_is_perfect_square_l34_3416

-- Define the expressions
def expr_A : ℕ := 3^3 * 4^5 * 7^7
def expr_B : ℕ := 3^4 * 4^4 * 7^5
def expr_C : ℕ := 3^6 * 4^3 * 7^6
def expr_D : ℕ := 3^5 * 4^6 * 7^4
def expr_E : ℕ := 3^4 * 4^6 * 7^6

-- Define a perfect square
def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m^2

-- Theorem statement
theorem only_C_is_perfect_square :
  is_perfect_square expr_C ∧
  ¬is_perfect_square expr_A ∧
  ¬is_perfect_square expr_B ∧
  ¬is_perfect_square expr_D ∧
  ¬is_perfect_square expr_E :=
sorry

end only_C_is_perfect_square_l34_3416


namespace tangent_segment_difference_l34_3410

/-- A quadrilateral inscribed in a circle with an inscribed circle --/
structure CyclicTangentialQuadrilateral where
  -- Sides of the quadrilateral
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ
  -- Condition: quadrilateral is inscribed in a circle
  is_cyclic : True
  -- Condition: quadrilateral has an inscribed circle
  has_incircle : True

/-- Theorem about the difference of segments on a side --/
theorem tangent_segment_difference
  (q : CyclicTangentialQuadrilateral)
  (h1 : q.a = 80)
  (h2 : q.b = 100)
  (h3 : q.c = 120)
  (h4 : q.d = 140)
  (x y : ℝ)
  (h5 : x + y = q.c)
  : |x - y| = 80 := by
  sorry

end tangent_segment_difference_l34_3410


namespace combined_tax_rate_l34_3474

/-- Combined tax rate calculation -/
theorem combined_tax_rate 
  (john_tax_rate : ℝ) 
  (ingrid_tax_rate : ℝ) 
  (john_income : ℝ) 
  (ingrid_income : ℝ) 
  (h1 : john_tax_rate = 0.3)
  (h2 : ingrid_tax_rate = 0.4)
  (h3 : john_income = 58000)
  (h4 : ingrid_income = 72000) :
  (john_tax_rate * john_income + ingrid_tax_rate * ingrid_income) / (john_income + ingrid_income) = 
  (0.3 * 58000 + 0.4 * 72000) / (58000 + 72000) := by
  sorry

end combined_tax_rate_l34_3474


namespace circle_fit_theorem_l34_3494

/-- Represents a square with unit side length -/
structure UnitSquare where
  x : ℝ
  y : ℝ

/-- Represents a rectangle -/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- The main theorem statement -/
theorem circle_fit_theorem (rect : Rectangle) (squares : Finset UnitSquare) :
  rect.width = 20 ∧ rect.height = 25 ∧ squares.card = 120 →
  ∃ (cx cy : ℝ), cx ∈ Set.Icc 0.5 19.5 ∧ cy ∈ Set.Icc 0.5 24.5 ∧
    ∀ (s : UnitSquare), s ∈ squares →
      (cx - s.x) ^ 2 + (cy - s.y) ^ 2 > 0.25 := by
  sorry

end circle_fit_theorem_l34_3494


namespace bucket_weight_l34_3471

/-- Given a bucket with weight c when 1/4 full and weight d when 3/4 full,
    prove that its weight when full is (3d - 3c)/2 -/
theorem bucket_weight (c d : ℝ) 
  (h1 : ∃ x y : ℝ, x + (1/4) * y = c ∧ x + (3/4) * y = d) : 
  ∃ w : ℝ, w = (3*d - 3*c)/2 ∧ 
  (∃ x y : ℝ, x + y = w ∧ x + (1/4) * y = c ∧ x + (3/4) * y = d) :=
by sorry

end bucket_weight_l34_3471


namespace additional_hamburgers_l34_3436

theorem additional_hamburgers (initial : ℕ) (total : ℕ) (h1 : initial = 9) (h2 : total = 12) :
  total - initial = 3 := by
  sorry

end additional_hamburgers_l34_3436


namespace product_scaling_l34_3455

theorem product_scaling (a b c : ℝ) (h : (268 : ℝ) * 74 = 19832) :
  2.68 * 0.74 = 1.9832 := by
  sorry

end product_scaling_l34_3455


namespace gum_distribution_l34_3461

theorem gum_distribution (cousins : ℕ) (total_gum : ℕ) (gum_per_cousin : ℕ) : 
  cousins = 4 → total_gum = 20 → gum_per_cousin = total_gum / cousins → gum_per_cousin = 5 := by
  sorry

end gum_distribution_l34_3461


namespace modulus_of_one_plus_i_l34_3490

/-- The modulus of the complex number z = 1 + i is √2 -/
theorem modulus_of_one_plus_i : Complex.abs (1 + Complex.I) = Real.sqrt 2 := by
  sorry

end modulus_of_one_plus_i_l34_3490


namespace total_students_sum_l34_3417

/-- The number of students in Varsity school -/
def varsity : ℕ := 1300

/-- The number of students in Northwest school -/
def northwest : ℕ := 1400

/-- The number of students in Central school -/
def central : ℕ := 1800

/-- The number of students in Greenbriar school -/
def greenbriar : ℕ := 1650

/-- The total number of students across all schools -/
def total_students : ℕ := varsity + northwest + central + greenbriar

theorem total_students_sum :
  total_students = 6150 := by sorry

end total_students_sum_l34_3417


namespace average_balance_is_200_l34_3480

/-- Represents the balance of a savings account for a given month -/
structure MonthlyBalance where
  month : String
  balance : ℕ

/-- Calculates the average monthly balance given a list of monthly balances -/
def averageMonthlyBalance (balances : List MonthlyBalance) : ℚ :=
  (balances.map (·.balance)).sum / balances.length

/-- Theorem stating that the average monthly balance is $200 -/
theorem average_balance_is_200 (balances : List MonthlyBalance) 
  (h1 : balances = [
    { month := "January", balance := 200 },
    { month := "February", balance := 300 },
    { month := "March", balance := 100 },
    { month := "April", balance := 250 },
    { month := "May", balance := 150 }
  ]) : 
  averageMonthlyBalance balances = 200 := by
  sorry


end average_balance_is_200_l34_3480


namespace right_triangle_area_l34_3437

/-- The area of a right triangle with hypotenuse 15 and one angle 45° --/
theorem right_triangle_area (h : ℝ) (α : ℝ) (area : ℝ) 
  (hyp : h = 15)
  (angle : α = 45 * Real.pi / 180)
  (right_angle : α + α + Real.pi / 2 = Real.pi) : 
  area = 112.5 := by
  sorry

end right_triangle_area_l34_3437


namespace wall_length_calculation_l34_3482

/-- Given a square mirror and a rectangular wall, if the mirror's area is exactly half the wall's area,
    prove that the length of the wall is approximately 43 inches. -/
theorem wall_length_calculation (mirror_side : ℝ) (wall_width : ℝ) : 
  mirror_side = 34 →
  wall_width = 54 →
  (mirror_side ^ 2) * 2 = wall_width * (round ((mirror_side ^ 2) * 2 / wall_width)) :=
by sorry

end wall_length_calculation_l34_3482


namespace parallel_lines_point_on_circle_l34_3454

def line1 (a b x y : ℝ) : Prop := (b + 2) * x + a * y + 4 = 0

def line2 (a b x y : ℝ) : Prop := a * x + (2 - b) * y - 3 = 0

def parallel (f g : ℝ → ℝ → Prop) : Prop := 
  ∀ x₁ y₁ x₂ y₂, f x₁ y₁ ∧ f x₂ y₂ → (x₁ ≠ x₂ → (y₁ - y₂) / (x₁ - x₂) = (y₂ - y₁) / (x₂ - x₁))

theorem parallel_lines_point_on_circle (a b : ℝ) :
  parallel (line1 a b) (line2 a b) → a^2 + b^2 = 4 := by
  sorry

end parallel_lines_point_on_circle_l34_3454


namespace apple_count_l34_3458

/-- The number of apples initially in the basket -/
def initial_apples : ℕ := sorry

/-- The number of oranges initially in the basket -/
def initial_oranges : ℕ := 5

/-- The number of oranges added to the basket -/
def added_oranges : ℕ := 5

/-- The total number of fruits in the basket after adding oranges -/
def total_fruits : ℕ := initial_apples + initial_oranges + added_oranges

theorem apple_count : initial_apples = 10 :=
  by
    have h1 : initial_oranges = 5 := rfl
    have h2 : added_oranges = 5 := rfl
    have h3 : 2 * initial_apples = total_fruits := sorry
    sorry

end apple_count_l34_3458


namespace monotone_special_function_characterization_l34_3431

/-- A monotone function on real numbers satisfying f(x) + 2x = f(f(x)) -/
def MonotoneSpecialFunction (f : ℝ → ℝ) : Prop :=
  Monotone f ∧ ∀ x, f x + 2 * x = f (f x)

/-- The theorem stating that a MonotoneSpecialFunction must be either f(x) = -x or f(x) = 2x -/
theorem monotone_special_function_characterization (f : ℝ → ℝ) 
  (hf : MonotoneSpecialFunction f) : 
  (∀ x, f x = -x) ∨ (∀ x, f x = 2 * x) :=
sorry

end monotone_special_function_characterization_l34_3431


namespace parallel_line_l34_3464

/-- A linear function in two variables -/
def LinearFunction (α : Type) [Ring α] := α → α → α

/-- A point in 2D space -/
structure Point (α : Type) [Ring α] where
  x : α
  y : α

/-- Theorem stating that the given equation represents a line parallel to l -/
theorem parallel_line
  {α : Type} [Field α]
  (f : LinearFunction α)
  (M N : Point α)
  (h1 : f M.x M.y = 0)
  (h2 : f N.x N.y ≠ 0) :
  ∃ (k : α), ∀ (P : Point α),
    f P.x P.y - f M.x M.y - f N.x N.y = 0 ↔ f P.x P.y = k :=
by sorry

end parallel_line_l34_3464


namespace tan_360_minus_45_l34_3469

theorem tan_360_minus_45 : Real.tan (360 * π / 180 - 45 * π / 180) = -1 := by sorry

end tan_360_minus_45_l34_3469


namespace jesses_rooms_l34_3456

theorem jesses_rooms (room_length : ℝ) (room_width : ℝ) (total_carpet : ℝ) :
  room_length = 19 →
  room_width = 18 →
  total_carpet = 6840 →
  (total_carpet / (room_length * room_width) : ℝ) = 20 := by
  sorry

end jesses_rooms_l34_3456


namespace f_max_min_range_l34_3425

/-- A cubic function with parameter a -/
def f (a x : ℝ) : ℝ := x^3 + a*x^2 + (a+6)*x + 1

/-- The derivative of f with respect to x -/
def f_derivative (a x : ℝ) : ℝ := 3*x^2 + 2*a*x + (a+6)

/-- Theorem stating the range of a for which f has both maximum and minimum values -/
theorem f_max_min_range (a : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ 
    (∀ z : ℝ, f a z ≤ f a x) ∧ 
    (∀ z : ℝ, f a z ≥ f a y)) ↔ 
  a < -3 ∨ a > 6 :=
sorry

end f_max_min_range_l34_3425


namespace sign_sum_theorem_l34_3411

theorem sign_sum_theorem (a b c d : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hd : d ≠ 0) :
  ∃ (x : ℤ), x ∈ ({5, 3, 2, 0, -3} : Set ℤ) ∧
  (a / |a| + b / |b| + c / |c| + d / |d| + (a * b * c * d) / |a * b * c * d| = x) := by
  sorry

end sign_sum_theorem_l34_3411


namespace least_subtraction_for_divisibility_l34_3486

theorem least_subtraction_for_divisibility : 
  ∃ (n : ℕ), n = 72 ∧ 
  (∀ (m : ℕ), m < n → ¬(127 ∣ (100203 - m))) ∧ 
  (127 ∣ (100203 - n)) := by
sorry

end least_subtraction_for_divisibility_l34_3486


namespace polynomial_value_at_zero_l34_3499

/-- A polynomial P(x) = x^2 + bx + c satisfying specific conditions -/
def P (b c : ℝ) (x : ℝ) : ℝ := x^2 + b*x + c

theorem polynomial_value_at_zero 
  (b c : ℝ) 
  (h1 : P b c (P b c 1) = 0)
  (h2 : P b c (P b c 2) = 0)
  (h3 : P b c 1 ≠ P b c 2) :
  P b c 0 = -3/2 :=
sorry

end polynomial_value_at_zero_l34_3499


namespace translated_line_proof_l34_3405

/-- Given a line y = 2x + 5 translated down by m units (m > 0) -/
def translated_line (x : ℝ) (m : ℝ) : ℝ := 2 * x + 5 - m

theorem translated_line_proof (m : ℝ) (h_m : m > 0) :
  (translated_line (-2) m = -6 → m = 7) ∧
  (∀ x : ℝ, translated_line x 7 < 0 ↔ x < 1) := by
  sorry

end translated_line_proof_l34_3405


namespace base_8_digit_product_l34_3450

def base_10_num : ℕ := 7890

def to_base_8 (n : ℕ) : List ℕ :=
  sorry

def digit_product (digits : List ℕ) : ℕ :=
  sorry

theorem base_8_digit_product :
  digit_product (to_base_8 base_10_num) = 84 :=
sorry

end base_8_digit_product_l34_3450


namespace inserted_numbers_sum_l34_3483

theorem inserted_numbers_sum (a b : ℝ) : 
  a > 0 ∧ b > 0 ∧ 
  (∃ d : ℝ, a = 2 + d ∧ b = 2 + 2*d) ∧ 
  (∃ r : ℝ, b = a * r ∧ 18 = b * r) →
  a + b = 16 := by
sorry

end inserted_numbers_sum_l34_3483


namespace l₂_slope_l34_3407

-- Define the slope and y-intercept of line l₁
def m₁ : ℝ := 2
def b₁ : ℝ := 3

-- Define the equation of line l₁
def l₁ (x y : ℝ) : Prop := y = m₁ * x + b₁

-- Define the equation of the symmetry line
def symmetry_line (x y : ℝ) : Prop := y = -x

-- Define the symmetry relation between two points
def symmetric (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  symmetry_line ((x₁ + x₂) / 2) ((y₁ + y₂) / 2)

-- Define line l₂ as symmetric to l₁ with respect to y = -x
def l₂ (x y : ℝ) : Prop :=
  ∃ (x₁ y₁ : ℝ), l₁ x₁ y₁ ∧ symmetric x₁ y₁ x y

-- State the theorem
theorem l₂_slope :
  ∃ (m₂ : ℝ), m₂ = 1/2 ∧ ∀ (x y : ℝ), l₂ x y → ∃ (b₂ : ℝ), y = m₂ * x + b₂ :=
sorry

end l₂_slope_l34_3407


namespace sqrt_fourth_root_approx_l34_3408

theorem sqrt_fourth_root_approx : 
  ∃ (x : ℝ), x^2 = (0.000625)^(1/4) ∧ |x - 0.4| < 0.05 := by sorry

end sqrt_fourth_root_approx_l34_3408


namespace popsicle_stick_count_l34_3430

/-- Represents the number of popsicle sticks in Gino's problem -/
structure PopsicleSticks where
  initial : ℕ
  given_away : ℕ
  left : ℕ

/-- Theorem stating that the initial number of popsicle sticks 
    is equal to the sum of those given away and those left -/
theorem popsicle_stick_count (p : PopsicleSticks) 
    (h1 : p.given_away = 50)
    (h2 : p.left = 13)
    : p.initial = p.given_away + p.left := by
  sorry

#check popsicle_stick_count

end popsicle_stick_count_l34_3430


namespace percent_lost_is_twenty_l34_3448

/-- Represents the number of games in each category -/
structure GameStats where
  won : ℕ
  lost : ℕ
  tied : ℕ

/-- Calculates the percentage of games lost -/
def percentLost (stats : GameStats) : ℚ :=
  stats.lost / (stats.won + stats.lost + stats.tied) * 100

/-- Theorem stating that for a team with a 7:3 win-to-loss ratio and 5 tied games,
    the percentage of games lost is 20% -/
theorem percent_lost_is_twenty {x : ℕ} (stats : GameStats)
    (h1 : stats.won = 7 * x)
    (h2 : stats.lost = 3 * x)
    (h3 : stats.tied = 5) :
  percentLost stats = 20 := by
  sorry

#eval percentLost ⟨7, 3, 5⟩

end percent_lost_is_twenty_l34_3448


namespace orange_pear_weight_equivalence_l34_3441

/-- Given that 7 oranges weigh the same as 5 pears, 
    prove that 49 oranges weigh the same as 35 pears. -/
theorem orange_pear_weight_equivalence :
  ∀ (orange_weight pear_weight : ℝ),
  orange_weight > 0 → pear_weight > 0 →
  7 * orange_weight = 5 * pear_weight →
  49 * orange_weight = 35 * pear_weight :=
by
  sorry

#check orange_pear_weight_equivalence

end orange_pear_weight_equivalence_l34_3441


namespace boat_fuel_cost_is_50_l34_3491

/-- The boat fuel cost per hour for Pat's shark hunting -/
def boat_fuel_cost_per_hour : ℚ :=
  let photo_earning : ℚ := 15
  let shark_interval : ℚ := 10 / 60  -- 10 minutes in hours
  let hunting_duration : ℚ := 5
  let expected_profit : ℚ := 200
  let total_sharks : ℚ := hunting_duration / shark_interval
  let total_earnings : ℚ := total_sharks * photo_earning
  let total_fuel_cost : ℚ := total_earnings - expected_profit
  total_fuel_cost / hunting_duration

/-- Theorem stating that the boat fuel cost per hour is $50 -/
theorem boat_fuel_cost_is_50 : boat_fuel_cost_per_hour = 50 := by
  sorry

end boat_fuel_cost_is_50_l34_3491


namespace min_value_sum_reciprocals_min_value_achieved_l34_3496

theorem min_value_sum_reciprocals (x y z : ℝ) 
  (h_pos : x > 0 ∧ y > 0 ∧ z > 0) 
  (h_sum : x + y + z = 1) : 
  (1 / x + 4 / y + 9 / z) ≥ 36 := by
sorry

theorem min_value_achieved (x y z : ℝ) 
  (h_pos : x > 0 ∧ y > 0 ∧ z > 0) 
  (h_sum : x + y + z = 1) : 
  ∃ (x₀ y₀ z₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ z₀ > 0 ∧ 
  x₀ + y₀ + z₀ = 1 ∧ 
  (1 / x₀ + 4 / y₀ + 9 / z₀) = 36 := by
sorry

end min_value_sum_reciprocals_min_value_achieved_l34_3496


namespace video_length_correct_l34_3459

/-- The length of each video in minutes -/
def video_length : ℝ := 7

/-- The number of videos watched per day -/
def videos_per_day : ℝ := 2

/-- The time spent watching ads in minutes -/
def ad_time : ℝ := 3

/-- The total time spent on Youtube in minutes -/
def total_time : ℝ := 17

/-- Theorem stating that the video length is correct given the conditions -/
theorem video_length_correct :
  videos_per_day * video_length + ad_time = total_time :=
by sorry

end video_length_correct_l34_3459


namespace movie_cost_ratio_l34_3442

/-- Proves that the ratio of the cost per minute of the new movie to the previous movie is 1/5 -/
theorem movie_cost_ratio :
  let previous_length : ℝ := 2 * 60  -- in minutes
  let new_length : ℝ := previous_length * 1.6
  let previous_cost_per_minute : ℝ := 50
  let total_new_cost : ℝ := 1920
  let new_cost_per_minute : ℝ := total_new_cost / new_length
  new_cost_per_minute / previous_cost_per_minute = 1 / 5 := by
sorry


end movie_cost_ratio_l34_3442


namespace unique_function_satisfying_equation_l34_3428

/-- A function satisfying the given functional equation -/
def SatisfiesEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x + f y + 1) = x + y + 1

/-- The theorem stating that there exists exactly one function satisfying the equation -/
theorem unique_function_satisfying_equation :
  ∃! f : ℝ → ℝ, SatisfiesEquation f :=
sorry

end unique_function_satisfying_equation_l34_3428


namespace initial_oil_fraction_l34_3478

/-- Proves that the initial fraction of oil in the cylinder was 3/4 -/
theorem initial_oil_fraction (total_capacity : ℕ) (added_bottles : ℕ) (final_fraction : ℚ) :
  total_capacity = 80 →
  added_bottles = 4 →
  final_fraction = 4/5 →
  (total_capacity : ℚ) * final_fraction - added_bottles = (3/4 : ℚ) * total_capacity := by
  sorry

end initial_oil_fraction_l34_3478


namespace quadratic_equation_solution_l34_3443

theorem quadratic_equation_solution : 
  ∀ x : ℝ, x^2 = 2*x ↔ x = 0 ∨ x = 2 := by sorry

end quadratic_equation_solution_l34_3443


namespace line_tangent_to_ellipse_l34_3426

/-- The value of m^2 for which the line y = mx + 2 is tangent to the ellipse x^2 + 9y^2 = 9 -/
theorem line_tangent_to_ellipse :
  ∃ (m : ℝ),
    (∀ (x y : ℝ), y = m * x + 2 → x^2 + 9 * y^2 = 9) →
    (∃! (x y : ℝ), y = m * x + 2 ∧ x^2 + 9 * y^2 = 9) →
    m^2 = 1/3 := by
  sorry

end line_tangent_to_ellipse_l34_3426


namespace expression_simplification_l34_3460

theorem expression_simplification :
  let a := 3
  let b := 4
  let c := 5
  let d := 6
  (Real.sqrt (a + b + c + d) / 3) + ((a * b + 10) / 4) = 5.5 + Real.sqrt 2 := by
  sorry

end expression_simplification_l34_3460


namespace females_over30_prefer_l34_3433

/-- Represents the survey data from WebStream --/
structure WebStreamSurvey where
  total_surveyed : ℕ
  total_prefer : ℕ
  males_prefer : ℕ
  females_under30_not_prefer : ℕ
  females_over30_not_prefer : ℕ

/-- Theorem stating the number of females over 30 who prefer WebStream --/
theorem females_over30_prefer (survey : WebStreamSurvey)
  (h1 : survey.total_surveyed = 420)
  (h2 : survey.total_prefer = 200)
  (h3 : survey.males_prefer = 80)
  (h4 : survey.females_under30_not_prefer = 90)
  (h5 : survey.females_over30_not_prefer = 70) :
  ∃ (females_over30_prefer : ℕ), females_over30_prefer = 110 := by
  sorry


end females_over30_prefer_l34_3433


namespace train_crossing_time_l34_3404

/-- The time taken for a train to cross a stationary point -/
theorem train_crossing_time (train_length : ℝ) (train_speed_kmh : ℝ) : 
  train_length = 180 → 
  train_speed_kmh = 108 → 
  (train_length / (train_speed_kmh * 1000 / 3600)) = 6 := by
  sorry

end train_crossing_time_l34_3404


namespace percentage_difference_l34_3475

theorem percentage_difference (x y : ℝ) (h : x = y * (1 - 0.45)) :
  y = x * (1 + 0.45) := by
  sorry

end percentage_difference_l34_3475


namespace trigonometric_inequality_and_supremum_l34_3403

theorem trigonometric_inequality_and_supremum 
  (x y z : ℝ) (m n : ℕ) (hm : m ≥ 2) (hn : n ≥ 2) :
  (Real.sin x)^m * (Real.cos y)^n + 
  (Real.sin y)^m * (Real.cos z)^n + 
  (Real.sin z)^m * (Real.cos x)^n ≤ 1 ∧ 
  ∃ (x₀ y₀ z₀ : ℝ), 
    (Real.sin x₀)^m * (Real.cos y₀)^n + 
    (Real.sin y₀)^m * (Real.cos z₀)^n + 
    (Real.sin z₀)^m * (Real.cos x₀)^n = 1 :=
by sorry

end trigonometric_inequality_and_supremum_l34_3403


namespace largest_angle_in_triangle_l34_3434

theorem largest_angle_in_triangle : ∀ (a b c : ℝ),
  -- Two angles sum to 7/5 of a right angle
  a + b = 7 / 5 * 90 →
  -- One angle is 40° larger than the other
  b = a + 40 →
  -- All angles are non-negative
  0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c →
  -- Sum of angles in a triangle is 180°
  a + b + c = 180 →
  -- The largest angle is 83°
  max a (max b c) = 83 := by
sorry

end largest_angle_in_triangle_l34_3434


namespace square_is_quadratic_l34_3414

/-- A quadratic function is of the form y = ax² + bx + c, where a, b, and c are constants, and a ≠ 0 -/
def IsQuadraticFunction (f : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The function f(x) = x² is a quadratic function -/
theorem square_is_quadratic : IsQuadraticFunction (λ x => x^2) := by
  sorry

end square_is_quadratic_l34_3414


namespace students_in_range_estimate_l34_3402

/-- Represents a normal distribution of scores -/
structure ScoreDistribution where
  mean : ℝ
  stdDev : ℝ
  isNormal : Bool

/-- Represents the student population and their score distribution -/
structure StudentPopulation where
  totalStudents : ℕ
  scoreDistribution : ScoreDistribution

/-- Calculates the number of students within a given score range -/
def studentsInRange (pop : StudentPopulation) (lowerBound upperBound : ℝ) : ℕ :=
  sorry

/-- The main theorem to prove -/
theorem students_in_range_estimate 
  (pop : StudentPopulation) 
  (h1 : pop.totalStudents = 3000) 
  (h2 : pop.scoreDistribution.isNormal = true) : 
  ∃ (ε : ℕ), ε ≤ 10 ∧ 
  (studentsInRange pop 70 80 = 408 + ε ∨ studentsInRange pop 70 80 = 408 - ε) :=
sorry

end students_in_range_estimate_l34_3402


namespace inscribed_right_triangle_exists_l34_3421

-- Define the circle
structure Circle where
  center : ℝ × ℝ
  radius : ℝ
  radius_pos : radius > 0

-- Define a point in 2D space
def Point := ℝ × ℝ

-- Define the condition for a point to be inside a circle
def inside_circle (p : Point) (c : Circle) : Prop :=
  let (x, y) := p
  let (cx, cy) := c.center
  (x - cx)^2 + (y - cy)^2 < c.radius^2

-- Define the condition for a point to be on the circumference of a circle
def on_circumference (p : Point) (c : Circle) : Prop :=
  let (x, y) := p
  let (cx, cy) := c.center
  (x - cx)^2 + (y - cy)^2 = c.radius^2

-- Define a right triangle
structure RightTriangle where
  A : Point
  B : Point
  C : Point
  right_angle : (C.1 - A.1) * (C.1 - B.1) + (C.2 - A.2) * (C.2 - B.2) = 0

-- Theorem statement
theorem inscribed_right_triangle_exists (c : Circle) (A B : Point)
  (h_A : inside_circle A c) (h_B : inside_circle B c) :
  ∃ (C : Point), on_circumference C c ∧
    ∃ (t : RightTriangle), t.A = A ∧ t.B = B ∧ t.C = C :=
sorry

end inscribed_right_triangle_exists_l34_3421


namespace value_of_a_l34_3418

theorem value_of_a (a b c d e : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) (he : e > 0)
  (hab : a * b = 2)
  (hbc : b * c = 3)
  (hcd : c * d = 4)
  (hde : d * e = 15)
  (hea : e * a = 10) :
  a = 4 / 3 := by
  sorry

end value_of_a_l34_3418


namespace square_difference_theorem_l34_3489

theorem square_difference_theorem : ∃ (n m : ℕ),
  (∀ k : ℕ, k^2 < 2018 → k ≤ n) ∧
  (n^2 < 2018) ∧
  (∀ k : ℕ, 2018 < k^2 → m ≤ k) ∧
  (2018 < m^2) ∧
  (m^2 - n^2 = 89) := by
sorry

end square_difference_theorem_l34_3489


namespace arithmetic_mean_problem_l34_3435

theorem arithmetic_mean_problem (x : ℚ) : 
  ((x + 10) + 20 + 3*x + 15 + (3*x + 6)) / 5 = 30 → x = 99 / 7 := by
  sorry

end arithmetic_mean_problem_l34_3435


namespace like_terms_exponent_sum_l34_3429

theorem like_terms_exponent_sum (m n : ℕ) : 
  (∃ (x y : ℝ), 2 * x^(3*n) * y^(m+4) = -3 * x^9 * y^(2*n)) → m + n = 5 := by
  sorry

end like_terms_exponent_sum_l34_3429


namespace quadratic_inequality_l34_3498

theorem quadratic_inequality (x : ℝ) : x^2 - 9*x + 18 ≤ 0 ↔ 3 ≤ x ∧ x ≤ 6 := by
  sorry

end quadratic_inequality_l34_3498


namespace womans_swimming_speed_l34_3477

/-- Given a woman who swims downstream and upstream with specific distances and times,
    this theorem proves her speed in still water. -/
theorem womans_swimming_speed
  (downstream_distance : ℝ)
  (upstream_distance : ℝ)
  (downstream_time : ℝ)
  (upstream_time : ℝ)
  (h_downstream : downstream_distance = 54)
  (h_upstream : upstream_distance = 6)
  (h_time : downstream_time = 6 ∧ upstream_time = 6)
  : ∃ (speed_still_water : ℝ) (stream_speed : ℝ),
    speed_still_water = 5 ∧
    downstream_distance / downstream_time = speed_still_water + stream_speed ∧
    upstream_distance / upstream_time = speed_still_water - stream_speed :=
by sorry

end womans_swimming_speed_l34_3477


namespace successive_integers_product_l34_3406

theorem successive_integers_product (n : ℕ) : 
  n * (n + 1) = 7832 → n = 88 := by sorry

end successive_integers_product_l34_3406


namespace least_number_divisor_l34_3409

theorem least_number_divisor (n : ℕ) (h1 : n % 5 = 3) (h2 : n % 67 = 3) (h3 : n % 8 = 3)
  (h4 : ∀ m : ℕ, m < n → (m % 5 = 3 ∧ m % 67 = 3 ∧ m % 8 = 3) → False)
  (h5 : n = 1683) :
  3 = Nat.gcd n (n - 3) :=
sorry

end least_number_divisor_l34_3409


namespace tribe_assignment_l34_3422

-- Define the two tribes
inductive Tribe
| Triussa
| La

-- Define a person as having a tribe
structure Person where
  tribe : Tribe

-- Define the three people
def person1 : Person := sorry
def person2 : Person := sorry
def person3 : Person := sorry

-- Define what it means for a statement to be true
def isTrueStatement (p : Person) (s : Prop) : Prop :=
  (p.tribe = Tribe.Triussa ∧ s) ∨ (p.tribe = Tribe.La ∧ ¬s)

-- Define the statements made by each person
def statement1 : Prop := 
  (person1.tribe = Tribe.Triussa ∧ person2.tribe = Tribe.La ∧ person3.tribe = Tribe.La) ∨
  (person1.tribe = Tribe.La ∧ person2.tribe = Tribe.Triussa ∧ person3.tribe = Tribe.La) ∨
  (person1.tribe = Tribe.La ∧ person2.tribe = Tribe.La ∧ person3.tribe = Tribe.Triussa)

def statement2 : Prop := person3.tribe = Tribe.La

def statement3 : Prop := person1.tribe = Tribe.La

-- Theorem to prove
theorem tribe_assignment :
  isTrueStatement person1 statement1 ∧
  isTrueStatement person2 statement2 ∧
  isTrueStatement person3 statement3 →
  person1.tribe = Tribe.La ∧ person2.tribe = Tribe.La ∧ person3.tribe = Tribe.Triussa :=
sorry

end tribe_assignment_l34_3422


namespace fair_products_l34_3485

/-- The number of recycled materials made by the group -/
def group_materials : ℕ := 65

/-- The number of recycled materials made by the teachers -/
def teacher_materials : ℕ := 28

/-- The total number of recycled products to sell at the fair -/
def total_products : ℕ := group_materials + teacher_materials

theorem fair_products : total_products = 93 := by
  sorry

end fair_products_l34_3485


namespace enumeration_pattern_correct_l34_3447

/-- Represents the number in a square of the enumerated grid -/
def square_number (m n : ℕ) : ℕ := Nat.choose (m + n - 1) 2 + n

/-- The enumeration pattern for the squared paper -/
def enumeration_pattern : ℕ → ℕ → ℕ := square_number

theorem enumeration_pattern_correct :
  ∀ (m n : ℕ), enumeration_pattern m n = square_number m n :=
by sorry

end enumeration_pattern_correct_l34_3447


namespace eighteenth_over_fortyfirst_415th_digit_l34_3468

def decimal_expansion (n d : ℕ) : List ℕ := sorry

def nth_digit (n : ℕ) (expansion : List ℕ) : ℕ := sorry

theorem eighteenth_over_fortyfirst_415th_digit :
  let expansion := decimal_expansion 18 41
  nth_digit 415 expansion = 3 := by sorry

end eighteenth_over_fortyfirst_415th_digit_l34_3468


namespace annual_pension_formula_l34_3412

/-- Represents an employee's pension calculation -/
structure PensionCalculation where
  x : ℝ  -- Years of service
  c : ℝ  -- Additional years scenario 1
  d : ℝ  -- Additional years scenario 2
  r : ℝ  -- Pension increase for scenario 1
  s : ℝ  -- Pension increase for scenario 2
  h1 : c ≠ d  -- Assumption that c and d are different

/-- The pension is proportional to years of service squared -/
def pension_proportional (p : PensionCalculation) (k : ℝ) : Prop :=
  ∃ (base_pension : ℝ), base_pension = k * p.x^2

/-- The pension increase after c more years of service -/
def pension_increase_c (p : PensionCalculation) (k : ℝ) : Prop :=
  k * (p.x + p.c)^2 - k * p.x^2 = p.r

/-- The pension increase after d more years of service -/
def pension_increase_d (p : PensionCalculation) (k : ℝ) : Prop :=
  k * (p.x + p.d)^2 - k * p.x^2 = p.s

/-- The theorem stating the formula for the annual pension -/
theorem annual_pension_formula (p : PensionCalculation) :
  ∃ (k : ℝ), 
    pension_proportional p k ∧ 
    pension_increase_c p k ∧ 
    pension_increase_d p k → 
    k = (p.s - p.r) / (2 * p.x * (p.d - p.c) + p.d^2 - p.c^2) :=
by sorry

end annual_pension_formula_l34_3412


namespace probability_of_white_ball_l34_3484

/-- The probability of drawing a white ball from a bag containing white and red balls -/
theorem probability_of_white_ball (num_white : ℕ) (num_red : ℕ) : 
  num_white = 6 → num_red = 14 → (num_white : ℚ) / (num_white + num_red : ℚ) = 3 / 10 := by
  sorry

end probability_of_white_ball_l34_3484


namespace cylinder_lateral_surface_area_l34_3401

/-- Given a cylinder with base area 4π and a lateral surface that unfolds into a square,
    prove that its lateral surface area is 16π. -/
theorem cylinder_lateral_surface_area (r h : ℝ) : 
  (π * r^2 = 4 * π) →  -- base area condition
  (2 * π * r = h) →    -- lateral surface unfolds into a square condition
  (2 * π * r * h = 16 * π) := by 
  sorry

end cylinder_lateral_surface_area_l34_3401


namespace f_xy_second_derivative_not_exists_l34_3473

noncomputable def f (x y : ℝ) : ℝ :=
  if x^2 + y^4 ≠ 0 then (x * y^2) / (x^2 + y^4) else 0

theorem f_xy_second_derivative_not_exists :
  ¬ ∃ (L : ℝ), ∀ ε > 0, ∃ δ > 0, ∀ x y : ℝ,
    x^2 + y^2 < δ^2 → |((f (x + y) y - f x y) / y - (f x y - f x 0) / y) / x - L| < ε :=
sorry

end f_xy_second_derivative_not_exists_l34_3473


namespace b_over_c_value_l34_3423

theorem b_over_c_value (a b c d e f : ℝ) 
  (h1 : a / b = 1 / 3)
  (h2 : a * b * c / (d * e * f) = 0.1875)
  (h3 : c / d = 1 / 2)
  (h4 : d / e = 3)
  (h5 : e / f = 1 / 8) :
  b / c = 3 := by
sorry

end b_over_c_value_l34_3423


namespace line_slope_is_two_l34_3419

/-- Given a line ax + 3my + 2a = 0 with m ≠ 0 and the sum of its intercepts on the coordinate axes is 2, prove that its slope is 2 -/
theorem line_slope_is_two (m a : ℝ) (hm : m ≠ 0) :
  (∃ (x y : ℝ), a * x + 3 * m * y + 2 * a = 0 ∧ 
   (a * 0 + 3 * m * y + 2 * a = 0 → y = -2 * a / (3 * m)) ∧
   (a * x + 3 * m * 0 + 2 * a = 0 → x = -2) ∧
   y + x = 2) →
  (∃ (k b : ℝ), ∀ x y, a * x + 3 * m * y + 2 * a = 0 ↔ y = k * x + b) ∧
  k = 2 :=
sorry

end line_slope_is_two_l34_3419


namespace binomial_cube_seven_l34_3440

theorem binomial_cube_seven : 7^3 + 3*(7^2) + 3*7 + 1 = 512 := by
  sorry

end binomial_cube_seven_l34_3440


namespace tuesday_sales_l34_3472

/-- Proves the number of bottles sold on Tuesday given inventory and sales information --/
theorem tuesday_sales (initial_inventory : ℕ) (monday_sales : ℕ) (daily_sales : ℕ) 
  (saturday_delivery : ℕ) (final_inventory : ℕ) : 
  initial_inventory = 4500 →
  monday_sales = 2445 →
  daily_sales = 50 →
  saturday_delivery = 650 →
  final_inventory = 1555 →
  initial_inventory + saturday_delivery - monday_sales - (daily_sales * 5) - final_inventory = 900 := by
  sorry

end tuesday_sales_l34_3472


namespace no_real_roots_l34_3457

theorem no_real_roots (a b c d : ℝ) 
  (h1 : ∀ x : ℝ, x^2 + a*x + b ≠ 0)
  (h2 : ∀ x : ℝ, x^2 + c*x + d ≠ 0) :
  ∀ x : ℝ, x^2 + ((a+c)/2)*x + ((b+d)/2) ≠ 0 := by
sorry

end no_real_roots_l34_3457


namespace carolyn_final_marbles_l34_3452

/-- Represents the number of marbles Carolyn has after sharing -/
def marbles_after_sharing (initial_marbles shared_marbles : ℕ) : ℕ :=
  initial_marbles - shared_marbles

/-- Theorem stating that Carolyn ends up with 5 marbles -/
theorem carolyn_final_marbles :
  marbles_after_sharing 47 42 = 5 := by
  sorry

end carolyn_final_marbles_l34_3452


namespace cos_squared_half_angle_minus_pi_fourth_l34_3479

theorem cos_squared_half_angle_minus_pi_fourth (α : Real) 
  (h : Real.sin α = 2/3) : 
  Real.cos (α/2 - π/4)^2 = 1/6 := by sorry

end cos_squared_half_angle_minus_pi_fourth_l34_3479


namespace ratio_calculation_l34_3470

theorem ratio_calculation (A B C : ℚ) (h : A / B = 3 / 2 ∧ B / C = 2 / 5) : 
  (4 * A + 3 * B) / (5 * C - 2 * A) = 18 / 19 := by sorry

end ratio_calculation_l34_3470


namespace fraction_equation_solution_l34_3487

theorem fraction_equation_solution (x : ℚ) :
  (x + 10) / (x - 4) = (x + 3) / (x - 6) → x = 48 / 5 := by
  sorry

end fraction_equation_solution_l34_3487


namespace smallest_factor_for_perfect_square_l34_3420

def y : ℕ := 2^3 * 3^4 * 5^6 * 7^8 * 8^9 * 9^10

theorem smallest_factor_for_perfect_square :
  (∀ m : ℕ, m > 0 ∧ m < 2 → ¬ ∃ k : ℕ, m * y = k^2) ∧
  ∃ k : ℕ, 2 * y = k^2 :=
sorry

end smallest_factor_for_perfect_square_l34_3420


namespace arithmetic_sequence_problem_l34_3444

/-- An arithmetic sequence is a sequence where the difference between any two consecutive terms is constant. -/
def ArithmeticSequence (a : ℕ → ℚ) : Prop :=
  ∃ d : ℚ, ∀ n : ℕ, a (n + 1) = a n + d

/-- Given an arithmetic sequence where the 4th term is 23 and the 9th term is 38, the 10th term is 41. -/
theorem arithmetic_sequence_problem (a : ℕ → ℚ) (h : ArithmeticSequence a) 
    (h4 : a 4 = 23) (h9 : a 9 = 38) : a 10 = 41 := by
  sorry


end arithmetic_sequence_problem_l34_3444


namespace drama_ticket_revenue_l34_3462

theorem drama_ticket_revenue (total_tickets : ℕ) (total_revenue : ℕ) 
  (h_total_tickets : total_tickets = 160)
  (h_total_revenue : total_revenue = 2400) : ∃ (full_price : ℕ) (half_price : ℕ) (price : ℕ),
  full_price + half_price = total_tickets ∧
  full_price * price + half_price * (price / 2) = total_revenue ∧
  full_price * price = 1600 :=
sorry

end drama_ticket_revenue_l34_3462
