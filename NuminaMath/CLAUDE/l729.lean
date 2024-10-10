import Mathlib

namespace product_mod_seventeen_l729_72952

theorem product_mod_seventeen : (3001 * 3002 * 3003 * 3004 * 3005) % 17 = 7 := by
  sorry

end product_mod_seventeen_l729_72952


namespace divisibility_implies_inequality_l729_72947

theorem divisibility_implies_inequality (a k : ℕ+) 
  (h : (a^2 + k) ∣ ((a - 1) * a * (a + 1))) : 
  k ≥ a := by
sorry

end divisibility_implies_inequality_l729_72947


namespace missing_number_is_four_l729_72984

theorem missing_number_is_four : 
  ∃ x : ℤ, (x + 3) + (8 - 3 - 1) = 11 ∧ x = 4 := by
sorry

end missing_number_is_four_l729_72984


namespace daughter_weight_l729_72992

/-- Proves that the weight of the daughter is 48 kg given the conditions of the problem -/
theorem daughter_weight (M D C : ℝ) 
  (total_weight : M + D + C = 120)
  (daughter_child_weight : D + C = 60)
  (child_grandmother_ratio : C = (1/5) * M) :
  D = 48 := by sorry

end daughter_weight_l729_72992


namespace tangent_line_at_1_1_l729_72930

-- Define the function
def f (x : ℝ) : ℝ := x^3 - 3*x^2 + 3

-- Define the derivative of the function
def f' (x : ℝ) : ℝ := 3*x^2 - 6*x

-- Theorem statement
theorem tangent_line_at_1_1 :
  let point : ℝ × ℝ := (1, 1)
  let slope : ℝ := f' point.1
  let tangent_line (x : ℝ) : ℝ := slope * (x - point.1) + point.2
  ∀ x, tangent_line x = -3 * x + 4 := by
sorry


end tangent_line_at_1_1_l729_72930


namespace middle_book_price_l729_72911

/-- A sequence of 49 numbers where each number differs by 5 from its adjacent numbers -/
def IncreasingSequence (a : ℕ → ℚ) : Prop :=
  (∀ n < 48, a (n + 1) = a n + 5) ∧ 
  (∀ n, n < 49)

theorem middle_book_price
  (a : ℕ → ℚ)
  (h1 : IncreasingSequence a)
  (h2 : a 48 = 2 * (a 23 + a 24 + a 25)) :
  a 24 = 24 := by
  sorry

end middle_book_price_l729_72911


namespace middle_card_is_six_l729_72944

def is_valid_set (a b c : ℕ) : Prop :=
  a < b ∧ b < c ∧ a + b + c = 20 ∧ a % 2 = 0 ∧ c % 2 = 0

def possible_after_aria (a b c : ℕ) : Prop :=
  is_valid_set a b c ∧ a ≠ 6

def possible_after_cece (a b c : ℕ) : Prop :=
  possible_after_aria a b c ∧ c ≠ 13

def possible_after_bruce (a b c : ℕ) : Prop :=
  possible_after_cece a b c ∧ (b ≠ 5 ∨ a ≠ 4)

theorem middle_card_is_six :
  ∀ a b c : ℕ, possible_after_bruce a b c → b = 6 :=
sorry

end middle_card_is_six_l729_72944


namespace cos_equality_proof_l729_72958

theorem cos_equality_proof (n : ℤ) : 
  0 ≤ n ∧ n ≤ 180 ∧ Real.cos (n * π / 180) = Real.cos (1534 * π / 180) → n = 154 := by
  sorry

end cos_equality_proof_l729_72958


namespace max_value_theorem_l729_72910

theorem max_value_theorem (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : 3 * x^2 - 2 * x * y + y^2 = 6) :
  ∃ (z : ℝ), z = 9 + 3 * Real.sqrt 3 ∧ 
  ∀ (w : ℝ), w = 3 * x^2 + 2 * x * y + y^2 → w ≤ z :=
sorry

end max_value_theorem_l729_72910


namespace f_continuous_at_1_l729_72929

def f (x : ℝ) : ℝ := -4 * x^2 - 6

theorem f_continuous_at_1 : 
  ∀ ε > 0, ∃ δ > 0, ∀ x, |x - 1| < δ → |f x - f 1| < ε :=
by sorry

end f_continuous_at_1_l729_72929


namespace tax_reduction_percentage_l729_72975

/-- Proves that if a tax rate is reduced by X%, consumption increases by 15%,
    and revenue decreases by 3.4%, then X = 16%. -/
theorem tax_reduction_percentage
  (T : ℝ)  -- Original tax rate (in percentage)
  (X : ℝ)  -- Percentage by which tax is reduced
  (h1 : T > 0)  -- Assumption that original tax rate is positive
  (h2 : X > 0)  -- Assumption that tax reduction is positive
  (h3 : X < T)  -- Assumption that tax reduction is less than original tax
  : ((T - X) / 100 * 115 = T / 100 * 96.6) → X = 16 :=
by sorry

end tax_reduction_percentage_l729_72975


namespace y_value_l729_72918

theorem y_value : ∀ y : ℚ, (2 / 5 - 1 / 7 : ℚ) = 14 / y → y = 490 / 9 := by
  sorry

end y_value_l729_72918


namespace rational_function_value_at_one_l729_72903

/-- A rational function with specific properties --/
structure RationalFunction where
  p : ℝ → ℝ
  q : ℝ → ℝ
  p_quadratic : ∃ a b c : ℝ, ∀ x, p x = a * x^2 + b * x + c
  q_quadratic : ∃ a b c : ℝ, ∀ x, q x = a * x^2 + b * x + c
  asymptote_minus_three : q (-3) = 0
  asymptote_two : q 2 = 0
  passes_origin : p 0 = 0 ∧ q 0 ≠ 0
  passes_one_two : p 1 = 2 * q 1 ∧ q 1 ≠ 0

/-- The main theorem --/
theorem rational_function_value_at_one (f : RationalFunction) : f.p 1 / f.q 1 = 2 := by
  sorry

end rational_function_value_at_one_l729_72903


namespace difference_of_squares_65_35_l729_72968

theorem difference_of_squares_65_35 : 65^2 - 35^2 = 3000 := by
  sorry

end difference_of_squares_65_35_l729_72968


namespace total_coins_is_32_l729_72967

/-- The number of dimes -/
def num_dimes : ℕ := 22

/-- The number of quarters -/
def num_quarters : ℕ := 10

/-- The total number of coins -/
def total_coins : ℕ := num_dimes + num_quarters

/-- Theorem: The total number of coins is 32 -/
theorem total_coins_is_32 : total_coins = 32 := by
  sorry

end total_coins_is_32_l729_72967


namespace bowtie_equation_solution_l729_72997

/-- The bowtie operation defined as a ⋈ b = a + √(b + √(b + √(b + ...))) -/
noncomputable def bowtie (a b : ℝ) : ℝ :=
  a + Real.sqrt (b + Real.sqrt (b + Real.sqrt b))

/-- Theorem: If 3 ⋈ z = 9, then z = 30 -/
theorem bowtie_equation_solution :
  ∃ z : ℝ, bowtie 3 z = 9 ∧ z = 30 := by
  sorry

end bowtie_equation_solution_l729_72997


namespace a_work_days_l729_72978

/-- The number of days B takes to finish the work alone -/
def b_days : ℝ := 8

/-- The number of days A and B work together -/
def together_days : ℝ := 2

/-- The number of days B works alone after A leaves -/
def b_alone_days : ℝ := 2

/-- The total amount of work to be done -/
def total_work : ℝ := 1

theorem a_work_days : ∃ (a : ℝ), 
  a > 0 ∧ 
  together_days * (1/a + 1/b_days) + b_alone_days * (1/b_days) = total_work ∧ 
  a = 4 := by
  sorry

end a_work_days_l729_72978


namespace sum_of_roots_l729_72927

theorem sum_of_roots (p q r s : ℝ) : 
  p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ q ≠ r ∧ q ≠ s ∧ r ≠ s →
  (∀ x, x^2 - 12*p*x + 14*q = 0 ↔ x = r ∨ x = s) →
  (∀ x, x^2 - 12*r*x - 14*s = 0 ↔ x = p ∨ x = q) →
  p + q + r + s = 2184 := by
sorry

end sum_of_roots_l729_72927


namespace sum_set_size_bounds_l729_72900

theorem sum_set_size_bounds (A : Finset ℕ) (S : Finset ℕ) : 
  A.card = 100 → 
  S = Finset.image (λ (p : ℕ × ℕ) => p.1 + p.2) (A.product A) → 
  199 ≤ S.card ∧ S.card ≤ 5050 := by
  sorry

end sum_set_size_bounds_l729_72900


namespace division_and_addition_l729_72907

theorem division_and_addition : (150 / (10 / 2)) + 5 = 35 := by
  sorry

end division_and_addition_l729_72907


namespace factorization_x4_minus_9_factorization_quadratic_in_a_and_b_l729_72973

-- Problem 1
theorem factorization_x4_minus_9 (x : ℝ) : 
  x^4 - 9 = (x^2 + 3) * (x + Real.sqrt 3) * (x - Real.sqrt 3) := by sorry

-- Problem 2
theorem factorization_quadratic_in_a_and_b (a b : ℝ) :
  -a^2*b + 2*a*b - b = -b*(a-1)^2 := by sorry

end factorization_x4_minus_9_factorization_quadratic_in_a_and_b_l729_72973


namespace price_reduction_achieves_target_profit_l729_72980

/-- Represents the daily sales and profit of a clothing store --/
structure ClothingStore where
  initialSales : ℕ
  initialProfit : ℝ
  priceReductionEffect : ℝ → ℕ
  targetProfit : ℝ

/-- Calculates the daily profit based on price reduction --/
def dailyProfit (store : ClothingStore) (priceReduction : ℝ) : ℝ :=
  let newSales := store.initialSales + store.priceReductionEffect priceReduction
  let newProfit := store.initialProfit - priceReduction
  newSales * newProfit

/-- Theorem stating that a $20 price reduction achieves the target profit --/
theorem price_reduction_achieves_target_profit (store : ClothingStore) 
  (h1 : store.initialSales = 20)
  (h2 : store.initialProfit = 40)
  (h3 : ∀ x, store.priceReductionEffect x = (8 / 4 : ℝ) * x)
  (h4 : store.targetProfit = 1200) :
  dailyProfit store 20 = store.targetProfit := by
  sorry


end price_reduction_achieves_target_profit_l729_72980


namespace lawn_mowing_total_l729_72933

theorem lawn_mowing_total (spring_mows summer_mows : ℕ) 
  (h1 : spring_mows = 6) 
  (h2 : summer_mows = 5) : 
  spring_mows + summer_mows = 11 := by
  sorry

end lawn_mowing_total_l729_72933


namespace mystery_number_multiple_of_four_l729_72951

def mystery_number (k : ℕ) : ℕ := (2*k+2)^2 - (2*k)^2

theorem mystery_number_multiple_of_four (k : ℕ) :
  ∃ m : ℕ, mystery_number k = 4 * m :=
sorry

end mystery_number_multiple_of_four_l729_72951


namespace shaded_fraction_of_square_l729_72917

theorem shaded_fraction_of_square (square_side : ℝ) (triangle_base : ℝ) (triangle_height : ℝ) :
  square_side = 4 →
  triangle_base = 3 →
  triangle_height = 2 →
  (square_side^2 - 2 * (triangle_base * triangle_height / 2)) / square_side^2 = 5/8 := by
  sorry

end shaded_fraction_of_square_l729_72917


namespace prob_three_even_out_of_six_l729_72914

/-- A fair 20-sided die -/
def Die : Type := Fin 20

/-- The probability of a single die showing an even number -/
def prob_even : ℚ := 1/2

/-- The number of dice rolled -/
def num_dice : ℕ := 6

/-- The number of dice we want to show even numbers -/
def target_even : ℕ := 3

/-- The probability of exactly three out of six fair 20-sided dice showing an even number -/
theorem prob_three_even_out_of_six : 
  (Nat.choose num_dice target_even : ℚ) * prob_even^target_even * (1 - prob_even)^(num_dice - target_even) = 5/16 := by
  sorry

end prob_three_even_out_of_six_l729_72914


namespace inclination_angle_range_l729_72963

/-- A line passing through a point -/
structure Line where
  slope : ℝ
  point : ℝ × ℝ

/-- A line segment defined by two endpoints -/
structure LineSegment where
  pointA : ℝ × ℝ
  pointB : ℝ × ℝ

/-- Checks if a line intersects a line segment -/
def intersects (l : Line) (seg : LineSegment) : Prop := sorry

/-- The inclination angle of a line -/
def inclinationAngle (l : Line) : ℝ := sorry

/-- The theorem statement -/
theorem inclination_angle_range 
  (l : Line) 
  (seg : LineSegment) :
  l.point = (0, -2) →
  seg.pointA = (1, -1) →
  seg.pointB = (2, -4) →
  intersects l seg →
  let α := inclinationAngle l
  (0 ≤ α ∧ α ≤ Real.pi / 4) ∨ (3 * Real.pi / 4 ≤ α ∧ α < Real.pi) := by
  sorry


end inclination_angle_range_l729_72963


namespace part_one_part_two_l729_72915

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- Theorem for part 1 of the problem -/
theorem part_one (t : Triangle) (h : 2 * t.a * Real.sin t.B = Real.sqrt 3 * t.b) :
  t.A = Real.pi / 3 ∨ t.A = 2 * Real.pi / 3 :=
sorry

/-- Theorem for part 2 of the problem -/
theorem part_two (t : Triangle) (h : t.a / 2 = t.b * Real.sin t.A) :
  (∀ x : Triangle, x.c / x.b + x.b / x.c ≤ 2 * Real.sqrt 2) ∧
  (∃ x : Triangle, x.c / x.b + x.b / x.c = 2 * Real.sqrt 2) :=
sorry

end part_one_part_two_l729_72915


namespace floor_sqrt_63_l729_72962

theorem floor_sqrt_63 : ⌊Real.sqrt 63⌋ = 7 := by
  sorry

end floor_sqrt_63_l729_72962


namespace correct_calculation_l729_72902

theorem correct_calculation (x y : ℝ) : -x^2*y + 3*x^2*y = 2*x^2*y := by
  sorry

end correct_calculation_l729_72902


namespace smallest_d_inequality_l729_72932

theorem smallest_d_inequality (x y z : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) (hz : z ≥ 0) :
  Real.sqrt (x * y * z) + (1/3) * |x^2 - y^2 + z^2| ≥ (x + y + z) / 3 ∧
  ∀ d : ℝ, d > 0 → d < 1/3 → ∃ a b c : ℝ, a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧
    Real.sqrt (a * b * c) + d * |a^2 - b^2 + c^2| < (a + b + c) / 3 :=
sorry

end smallest_d_inequality_l729_72932


namespace koi_fish_count_l729_72940

theorem koi_fish_count : ∃ k : ℕ, (2 * k - 14 = 64) ∧ (k = 39) := by
  sorry

end koi_fish_count_l729_72940


namespace hexagon_area_l729_72965

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- The hexagon defined by its vertices -/
def hexagon : List Point := [
  ⟨0, 3⟩, ⟨3, 3⟩, ⟨4, 0⟩, ⟨3, -3⟩, ⟨0, -3⟩, ⟨-1, 0⟩
]

/-- Calculate the area of a polygon given its vertices -/
def polygonArea (vertices : List Point) : ℝ := sorry

/-- Theorem: The area of the specified hexagon is 18 square units -/
theorem hexagon_area : polygonArea hexagon = 18 := by sorry

end hexagon_area_l729_72965


namespace pencils_left_l729_72925

def initial_pencils : ℕ := 142
def pencils_given_away : ℕ := 31

theorem pencils_left : initial_pencils - pencils_given_away = 111 := by
  sorry

end pencils_left_l729_72925


namespace probability_three_green_marbles_l729_72946

theorem probability_three_green_marbles : 
  let total_marbles : ℕ := 15
  let green_marbles : ℕ := 8
  let purple_marbles : ℕ := 7
  let total_trials : ℕ := 7
  let green_trials : ℕ := 3
  
  let prob_green : ℚ := green_marbles / total_marbles
  let prob_purple : ℚ := purple_marbles / total_marbles
  
  let ways_to_choose_green : ℕ := Nat.choose total_trials green_trials
  let prob_specific_outcome : ℚ := prob_green ^ green_trials * prob_purple ^ (total_trials - green_trials)
  
  ways_to_choose_green * prob_specific_outcome = 43079680 / 170859375 :=
by
  sorry

end probability_three_green_marbles_l729_72946


namespace part_to_whole_ratio_l729_72950

theorem part_to_whole_ratio (N : ℝ) (part : ℝ) : 
  (1/4 : ℝ) * part * (2/5 : ℝ) * N = 20 →
  (40/100 : ℝ) * N = 240 →
  part / ((2/5 : ℝ) * N) = 1/3 := by
sorry

end part_to_whole_ratio_l729_72950


namespace batsman_average_l729_72960

theorem batsman_average (total_innings : ℕ) (last_innings_score : ℕ) (average_increase : ℚ) :
  total_innings = 25 →
  last_innings_score = 95 →
  average_increase = 3.5 →
  (∃ (previous_average : ℚ),
    (previous_average * (total_innings - 1) + last_innings_score) / total_innings = 
    previous_average + average_increase) →
  (∃ (final_average : ℚ), final_average = 11) :=
by sorry

end batsman_average_l729_72960


namespace parabola_line_intersection_l729_72921

theorem parabola_line_intersection (α : Real) : 
  (∃! x, 3 * x^2 + 1 = 4 * Real.sin α * x) → α = Real.pi / 3 := by
  sorry

end parabola_line_intersection_l729_72921


namespace slope_product_sufficient_not_necessary_l729_72964

/-- A line in a 2D plane represented by its slope and y-intercept -/
structure Line where
  slope : ℝ
  y_intercept : ℝ

/-- Two lines are perpendicular -/
def perpendicular (l₁ l₂ : Line) : Prop :=
  sorry

/-- The product of slopes of two lines is -1 -/
def slope_product_negative_one (l₁ l₂ : Line) : Prop :=
  l₁.slope * l₂.slope = -1

/-- The product of slopes being -1 is sufficient but not necessary for perpendicularity -/
theorem slope_product_sufficient_not_necessary :
  (∀ l₁ l₂ : Line, slope_product_negative_one l₁ l₂ → perpendicular l₁ l₂) ∧
  ¬(∀ l₁ l₂ : Line, perpendicular l₁ l₂ → slope_product_negative_one l₁ l₂) :=
sorry

end slope_product_sufficient_not_necessary_l729_72964


namespace blind_box_probabilities_l729_72948

def total_boxes : ℕ := 7
def rabbit_boxes : ℕ := 4
def dog_boxes : ℕ := 3

theorem blind_box_probabilities :
  (∀ (n m : ℕ), n + m = total_boxes → n = rabbit_boxes → m = dog_boxes →
    (Nat.choose rabbit_boxes 1 * Nat.choose (total_boxes - 1) 1 ≠ 0 →
      (Nat.choose rabbit_boxes 1 * Nat.choose (rabbit_boxes - 1) 1 : ℚ) /
      (Nat.choose rabbit_boxes 1 * Nat.choose (total_boxes - 1) 1 : ℚ) = 1 / 2)) ∧
  (∀ (n m : ℕ), n + m = total_boxes → n = rabbit_boxes → m = dog_boxes →
    (Nat.choose total_boxes 1 ≠ 0 →
      (Nat.choose dog_boxes 1 : ℚ) / (Nat.choose total_boxes 1 : ℚ) = 3 / 7)) :=
sorry

end blind_box_probabilities_l729_72948


namespace simplify_expression_l729_72919

theorem simplify_expression (x : ℝ) : 3*x + 6*x + 9*x + 12*x + 15*x + 18 = 45*x + 18 := by
  sorry

end simplify_expression_l729_72919


namespace problem_solution_l729_72956

def A : Set ℝ := {x | x^2 - 3*x + 2 = 0}
def B (a : ℝ) : Set ℝ := {x | (x-1)*(x-a+1) = 0}
def C (m : ℝ) : Set ℝ := {x | x^2 - m*x + 2 = 0}

theorem problem_solution (a m : ℝ) 
  (h1 : A ∪ B a = A) 
  (h2 : A ∩ C m = C m) : 
  (a = 2 ∨ a = 3) ∧ 
  (m = 3 ∨ (-2 * Real.sqrt 2 < m ∧ m < 2 * Real.sqrt 2)) := by
  sorry

end problem_solution_l729_72956


namespace original_fraction_value_l729_72971

theorem original_fraction_value (n : ℚ) : 
  (n + 1) / (n + 6) = 7 / 12 → n / (n + 5) = 6 / 11 := by
  sorry

end original_fraction_value_l729_72971


namespace study_group_size_l729_72934

theorem study_group_size : ∃ n : ℕ, 
  n > 0 ∧ 
  n * (n - 1) = 90 ∧ 
  n = 10 := by
sorry

end study_group_size_l729_72934


namespace meatballs_stolen_l729_72982

/-- The number of meatballs Hayley initially had -/
def initial_meatballs : ℕ := 25

/-- The number of meatballs Hayley has now -/
def current_meatballs : ℕ := 11

/-- The number of meatballs Kirsten stole -/
def stolen_meatballs : ℕ := initial_meatballs - current_meatballs

theorem meatballs_stolen : stolen_meatballs = 14 := by
  sorry

end meatballs_stolen_l729_72982


namespace modulus_of_z_squared_l729_72988

theorem modulus_of_z_squared (i : ℂ) (h : i^2 = -1) : 
  let z := (2 - i)^2
  Complex.abs z = 5 := by
sorry

end modulus_of_z_squared_l729_72988


namespace sun_city_population_relation_l729_72943

/-- The population of Willowdale city -/
def willowdale_population : ℕ := 2000

/-- The population of Roseville city -/
def roseville_population : ℕ := 3 * willowdale_population - 500

/-- The population of Sun City -/
def sun_city_population : ℕ := 12000

/-- The multiple of Roseville City's population that Sun City has 1000 more than -/
def multiple : ℚ := (sun_city_population - 1000) / roseville_population

theorem sun_city_population_relation :
  sun_city_population = multiple * roseville_population + 1000 ∧ multiple = 2 := by sorry

end sun_city_population_relation_l729_72943


namespace no_rational_roots_l729_72904

-- Define the polynomial
def f (x : ℚ) : ℚ := 5 * x^3 - 4 * x^2 - 8 * x + 3

-- Theorem statement
theorem no_rational_roots : ∀ x : ℚ, f x ≠ 0 := by
  sorry

end no_rational_roots_l729_72904


namespace g20_asia_members_l729_72954

/-- Represents the continents in the G20 --/
inductive Continent
  | Asia
  | Europe
  | Africa
  | Oceania
  | America

/-- Structure representing the G20 membership distribution --/
structure G20 where
  members : Continent → ℕ
  total_twenty : (members Continent.Asia + members Continent.Europe + members Continent.Africa + 
                  members Continent.Oceania + members Continent.America) = 20
  asia_highest : ∀ c : Continent, members Continent.Asia ≥ members c
  africa_oceania_least : members Continent.Africa = members Continent.Oceania ∧ 
                         ∀ c : Continent, members c ≥ members Continent.Africa
  consecutive : ∃ x : ℕ, members Continent.America = x ∧ 
                         members Continent.Europe = x + 1 ∧ 
                         members Continent.Asia = x + 2

theorem g20_asia_members (g : G20) : g.members Continent.Asia = 7 := by
  sorry

end g20_asia_members_l729_72954


namespace binomial_2024_1_l729_72974

theorem binomial_2024_1 : Nat.choose 2024 1 = 2024 := by sorry

end binomial_2024_1_l729_72974


namespace largest_value_l729_72916

theorem largest_value : 
  (4^2 : ℝ) ≥ 4 * 2 ∧ 
  (4^2 : ℝ) ≥ 4 - 2 ∧ 
  (4^2 : ℝ) ≥ 4 / 2 ∧ 
  (4^2 : ℝ) ≥ 4 + 2 := by
  sorry

end largest_value_l729_72916


namespace remainder_of_12345678910_mod_101_l729_72901

theorem remainder_of_12345678910_mod_101 : 12345678910 % 101 = 31 := by
  sorry

end remainder_of_12345678910_mod_101_l729_72901


namespace necessary_but_not_sufficient_l729_72945

-- Define the condition for a hyperbola
def is_hyperbola (k : ℝ) : Prop :=
  (k - 2) * (k - 6) < 0

-- Define the condition given in the problem
def condition (k : ℝ) : Prop :=
  1 < k ∧ k < 7

-- Theorem statement
theorem necessary_but_not_sufficient :
  (∀ k, is_hyperbola k → condition k) ∧
  (∃ k, condition k ∧ ¬is_hyperbola k) :=
sorry

end necessary_but_not_sufficient_l729_72945


namespace weight_of_replaced_person_l729_72937

theorem weight_of_replaced_person 
  (n : ℕ) 
  (avg_increase : ℝ) 
  (new_person_weight : ℝ) 
  (h1 : n = 8)
  (h2 : avg_increase = 2.5)
  (h3 : new_person_weight = 60) :
  ∃ (replaced_weight : ℝ), replaced_weight = new_person_weight - n * avg_increase :=
by sorry

end weight_of_replaced_person_l729_72937


namespace sqrt_164_between_12_and_13_l729_72995

theorem sqrt_164_between_12_and_13 : 12 < Real.sqrt 164 ∧ Real.sqrt 164 < 13 := by
  sorry

end sqrt_164_between_12_and_13_l729_72995


namespace max_abs_u_for_unit_circle_l729_72993

theorem max_abs_u_for_unit_circle (z : ℂ) (h : Complex.abs z = 1) :
  Complex.abs (z^4 - z^3 - 3*z^2*Complex.I - z + 1) ≤ 5 ∧
  Complex.abs ((-1 : ℂ)^4 - (-1 : ℂ)^3 - 3*(-1 : ℂ)^2*Complex.I - (-1 : ℂ) + 1) = 5 :=
by sorry

end max_abs_u_for_unit_circle_l729_72993


namespace cubic_roots_difference_l729_72976

-- Define the cubic polynomial
def cubic_poly (x : ℝ) : ℝ := 27 * x^3 - 81 * x^2 + 63 * x - 14

-- Define a predicate for roots in geometric progression
def roots_in_geometric_progression (r₁ r₂ r₃ : ℝ) : Prop :=
  ∃ (a r : ℝ), r₁ = a ∧ r₂ = a * r ∧ r₃ = a * r^2

-- Theorem statement
theorem cubic_roots_difference (r₁ r₂ r₃ : ℝ) :
  cubic_poly r₁ = 0 ∧ cubic_poly r₂ = 0 ∧ cubic_poly r₃ = 0 →
  roots_in_geometric_progression r₁ r₂ r₃ →
  (max r₁ (max r₂ r₃))^2 - (min r₁ (min r₂ r₃))^2 = 5/3 :=
by sorry

end cubic_roots_difference_l729_72976


namespace staircase_perimeter_l729_72941

/-- Represents a staircase-shaped region with specific properties -/
structure StaircaseRegion where
  tickMarkSides : ℕ
  tickMarkLength : ℝ
  bottomBaseLength : ℝ
  totalArea : ℝ

/-- Calculates the perimeter of a StaircaseRegion -/
def perimeter (s : StaircaseRegion) : ℝ :=
  s.bottomBaseLength + s.tickMarkSides * s.tickMarkLength

theorem staircase_perimeter (s : StaircaseRegion) 
  (h1 : s.tickMarkSides = 12)
  (h2 : s.tickMarkLength = 1)
  (h3 : s.bottomBaseLength = 12)
  (h4 : s.totalArea = 78) :
  perimeter s = 34.5 := by
  sorry

end staircase_perimeter_l729_72941


namespace boys_neither_happy_nor_sad_l729_72908

theorem boys_neither_happy_nor_sad (total_children total_boys total_girls happy_children sad_children neither_children happy_boys sad_girls : ℕ) : 
  total_children = 60 →
  total_boys = 16 →
  total_girls = 44 →
  happy_children = 30 →
  sad_children = 10 →
  neither_children = 20 →
  happy_boys = 6 →
  sad_girls = 4 →
  total_children = total_boys + total_girls →
  happy_children + sad_children + neither_children = total_children →
  (total_boys - happy_boys - (sad_children - sad_girls) = 4) :=
by sorry

end boys_neither_happy_nor_sad_l729_72908


namespace arithmetic_mean_of_special_set_l729_72923

theorem arithmetic_mean_of_special_set (n : ℕ) (hn : n > 2) : 
  let set := [1 - 1 / n, 1 + 1 / n] ++ List.replicate (n - 2) 1
  (List.sum set) / n = 1 := by
sorry

end arithmetic_mean_of_special_set_l729_72923


namespace square_root_of_nine_l729_72972

-- Define the square root operation
def square_root (x : ℝ) : Set ℝ := {y : ℝ | y^2 = x}

-- Theorem statement
theorem square_root_of_nine : square_root 9 = {-3, 3} := by
  sorry

end square_root_of_nine_l729_72972


namespace sequence_ratio_l729_72981

/-- Given a sequence {a_n} where the sum of the first n terms S_n satisfies S_n = 2a_n - 2,
    prove that the ratio a_8 / a_6 = 4. -/
theorem sequence_ratio (a : ℕ → ℝ) (S : ℕ → ℝ) 
  (h : ∀ n, S n = 2 * a n - 2) : 
  a 8 / a 6 = 4 := by
  sorry

end sequence_ratio_l729_72981


namespace thermodynamic_expansion_l729_72994

/-- First law of thermodynamics --/
def first_law (Q Δu A : ℝ) : Prop := Q = Δu + A

/-- Ideal gas law --/
def ideal_gas_law (P V R T : ℝ) : Prop := P * V = R * T

theorem thermodynamic_expansion 
  (Q Δu A cᵥ T T₀ k x P S n R P₀ V₀ : ℝ) 
  (h_Q : Q = 0)
  (h_Δu : Δu = cᵥ * (T - T₀))
  (h_A : A = (k * x^2) / 2)
  (h_kx : k * x = P * S)
  (h_V : S * x = V₀ * (n - 1) / n)
  (h_first_law : first_law Q Δu A)
  (h_ideal_gas_initial : ideal_gas_law P₀ V₀ R T₀)
  (h_ideal_gas_final : ideal_gas_law P (n * V₀) R T)
  (h_positive : cᵥ > 0 ∧ n > 1 ∧ R > 0 ∧ T₀ > 0 ∧ P₀ > 0) :
  P = P₀ / (n * (1 + ((n - 1) * R) / (2 * n * cᵥ))) :=
sorry

end thermodynamic_expansion_l729_72994


namespace max_employees_l729_72931

theorem max_employees (x : ℝ) (h : x > 4) : 
  ∃ (n : ℕ), n = ⌊2 * (x / (2 * x - 8))⌋ ∧ 
  ∀ (m : ℕ), (∀ (i j : ℕ), i < m → j < m → i ≠ j → 
    ∃ (t : ℝ), 0 ≤ t ∧ t ≤ 8 ∧ t + x / 60 ≤ 8 ∧
    ∃ (ti tj : ℝ), 0 ≤ ti ∧ ti ≤ 8 ∧ 0 ≤ tj ∧ tj ≤ 8 ∧
    (t ≤ ti ∧ ti < t + x / 60) ∧ (t ≤ tj ∧ tj < t + x / 60)) →
  m ≤ n :=
sorry

end max_employees_l729_72931


namespace ellipse_equation_fixed_point_l729_72935

/-- Ellipse C with center at origin, foci on x-axis, and eccentricity 1/2 -/
structure EllipseC where
  equation : ℝ → ℝ → Prop
  center_origin : equation 0 0
  foci_on_x_axis : ∀ x y, equation x y → y = 0 → x ≠ 0
  eccentricity : (∀ x y, equation x y → x^2 + y^2 = 1) → 
                 (∃ c, c > 0 ∧ ∀ x y, equation x y → x^2 + y^2 = (1 - c^2) * x^2 + y^2)

/-- Parabola with equation x = 1/4 * y^2 -/
def parabola (x y : ℝ) : Prop := x = 1/4 * y^2

/-- One vertex of ellipse C coincides with the focus of the parabola -/
axiom vertex_coincides_focus (C : EllipseC) : 
  ∃ x y, C.equation x y ∧ x^2 + y^2 = 1 ∧ x = 1 ∧ y = 0

/-- Theorem: Standard equation of ellipse C -/
theorem ellipse_equation (C : EllipseC) : 
  ∀ x y, C.equation x y ↔ x^2 + 4/3 * y^2 = 1 :=
sorry

/-- Chord AB of ellipse C passing through (1, 0) -/
def chord (C : EllipseC) (m : ℝ) (x y : ℝ) : Prop :=
  C.equation x y ∧ y = m * (x - 1)

/-- A' is the reflection of A over the x-axis -/
def reflect_over_x (x y : ℝ) : ℝ × ℝ := (x, -y)

/-- Theorem: Line A'B passes through (1, 0) -/
theorem fixed_point (C : EllipseC) (m : ℝ) (h : m ≠ 0) :
  ∃ x₁ y₁ x₂ y₂, 
    chord C m x₁ y₁ ∧ 
    chord C m x₂ y₂ ∧ 
    (x₁ ≠ x₂ ∨ y₁ ≠ y₂) ∧
    let (x₁', y₁') := reflect_over_x x₁ y₁
    (y₁' - y₂) / (x₁' - x₂) = (0 - y₂) / (1 - x₂) :=
sorry

end ellipse_equation_fixed_point_l729_72935


namespace probability_two_aces_standard_deck_l729_72905

/-- Represents a standard deck of cards -/
structure Deck :=
  (total_cards : ℕ)
  (ace_count : ℕ)

/-- The probability of drawing two Aces as the top two cards from a randomly arranged deck -/
def probability_two_aces (d : Deck) : ℚ :=
  (d.ace_count : ℚ) / d.total_cards * (d.ace_count - 1) / (d.total_cards - 1)

/-- Theorem: The probability of drawing two Aces as the top two cards from a standard deck is 1/221 -/
theorem probability_two_aces_standard_deck :
  probability_two_aces ⟨52, 4⟩ = 1 / 221 := by
  sorry

#eval probability_two_aces ⟨52, 4⟩

end probability_two_aces_standard_deck_l729_72905


namespace worker_a_completion_time_l729_72959

/-- The number of days it takes for two workers to complete a job together -/
def combined_days : ℝ := 18

/-- The ratio of worker a's speed to worker b's speed -/
def speed_ratio : ℝ := 1.5

/-- The number of days it takes for worker a to complete the job alone -/
def days_a : ℝ := 30

theorem worker_a_completion_time : 
  1 / combined_days = 1 / days_a + 1 / (speed_ratio * days_a) :=
by sorry

end worker_a_completion_time_l729_72959


namespace turnip_bag_weights_l729_72987

def bag_weights : List ℕ := [13, 15, 16, 17, 21, 24]

def is_valid_turnip_weight (t : ℕ) : Prop :=
  t ∈ bag_weights ∧
  ∃ (onion_weight carrot_weight : ℕ),
    onion_weight + carrot_weight = (bag_weights.sum - t) ∧
    carrot_weight = 2 * onion_weight ∧
    ∃ (onion_bags carrot_bags : List ℕ),
      onion_bags ++ carrot_bags = bag_weights.filter (λ w => w ≠ t) ∧
      onion_bags.sum = onion_weight ∧
      carrot_bags.sum = carrot_weight

theorem turnip_bag_weights :
  ∀ t : ℕ, is_valid_turnip_weight t ↔ t = 13 ∨ t = 16 :=
by sorry

end turnip_bag_weights_l729_72987


namespace election_votes_l729_72936

theorem election_votes (total_votes : ℕ) 
  (h1 : total_votes > 0)
  (h2 : (62 * total_votes) / 100 - (38 * total_votes) / 100 = 384) :
  (62 * total_votes) / 100 = 992 := by
  sorry

end election_votes_l729_72936


namespace inequality_solution_l729_72939

theorem inequality_solution (x : ℝ) (h : x ≠ -1) :
  (x - 2) / (x + 1) ≤ 2 ↔ x ≤ -4 ∨ x > -1 := by
  sorry

end inequality_solution_l729_72939


namespace delta_nabla_equality_l729_72955

/-- Definition of the Δ operation -/
def delta (a b : ℕ) : ℕ := 3 * a + 2 * b

/-- Definition of the ∇ operation -/
def nabla (a b : ℕ) : ℕ := 2 * a + 3 * b

/-- Theorem stating that 3 Δ (2 ∇ 1) = 23 -/
theorem delta_nabla_equality : delta 3 (nabla 2 1) = 23 := by
  sorry

end delta_nabla_equality_l729_72955


namespace division_equality_l729_72969

theorem division_equality (h : 29.94 / 1.45 = 17.1) : 2994 / 14.5 = 171 := by
  sorry

end division_equality_l729_72969


namespace max_value_implies_a_l729_72977

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := x^3 - 3*x^2 + a

-- State the theorem
theorem max_value_implies_a (a : ℝ) :
  (∀ x ∈ Set.Icc (-1) 1, f a x ≤ 2) ∧ 
  (∃ x ∈ Set.Icc (-1) 1, f a x = 2) →
  a = 2 := by
  sorry


end max_value_implies_a_l729_72977


namespace roots_sum_reciprocal_l729_72942

theorem roots_sum_reciprocal (a b : ℝ) : 
  (a^2 + 10*a + 5 = 0) → 
  (b^2 + 10*b + 5 = 0) → 
  (a/b + b/a = 18) := by sorry

end roots_sum_reciprocal_l729_72942


namespace circle_center_radius_sum_l729_72913

/-- Given a circle with equation x^2 - 16x + y^2 + 6y = -75, 
    prove that the sum of its center coordinates and radius is 5 + √2 -/
theorem circle_center_radius_sum :
  ∃ (a b r : ℝ), 
    (∀ x y : ℝ, x^2 - 16*x + y^2 + 6*y = -75 ↔ (x - a)^2 + (y - b)^2 = r^2) ∧
    a + b + r = 5 + Real.sqrt 2 := by
  sorry

end circle_center_radius_sum_l729_72913


namespace river_speed_is_6_l729_72906

/-- Proves that the speed of the river is 6 km/h given the conditions of the boat problem -/
theorem river_speed_is_6 (total_distance : ℝ) (downstream_distance : ℝ) (still_water_speed : ℝ)
  (h1 : total_distance = 150)
  (h2 : downstream_distance = 90)
  (h3 : still_water_speed = 30)
  (h4 : downstream_distance / (still_water_speed + 6) = (total_distance - downstream_distance) / (still_water_speed - 6)) :
  6 = 6 := by
sorry

end river_speed_is_6_l729_72906


namespace company_capital_growth_l729_72990

/-- Calculates the final capital after n years given initial capital, growth rate, and yearly consumption --/
def finalCapital (initialCapital : ℝ) (growthRate : ℝ) (yearlyConsumption : ℝ) (years : ℕ) : ℝ :=
  match years with
  | 0 => initialCapital
  | n + 1 => (finalCapital initialCapital growthRate yearlyConsumption n * (1 + growthRate)) - yearlyConsumption

/-- The problem statement --/
theorem company_capital_growth (x : ℝ) : 
  finalCapital 1 0.5 x 3 = 2.9 ↔ x = 10 := by
  sorry

end company_capital_growth_l729_72990


namespace bench_wood_length_l729_72970

theorem bench_wood_length (num_long_pieces : ℕ) (long_piece_length : ℝ) (total_wood : ℝ) :
  num_long_pieces = 6 →
  long_piece_length = 4 →
  total_wood = 28 →
  total_wood - (num_long_pieces : ℝ) * long_piece_length = 4 := by
  sorry

end bench_wood_length_l729_72970


namespace prism_properties_l729_72949

/-- Represents a prism with n sides in its base. -/
structure Prism (n : ℕ) where
  base_sides : n ≥ 3

/-- Properties of a prism. -/
def Prism.properties (p : Prism n) : Prop :=
  let lateral_faces := n
  let lateral_edges := n
  let total_edges := 3 * n
  let total_faces := n + 2
  let total_vertices := 2 * n
  lateral_faces = lateral_edges ∧
  total_edges % 3 = 0 ∧
  (n ≥ 4 → Even total_faces) ∧
  Even total_vertices

/-- Theorem stating the properties of a prism. -/
theorem prism_properties (n : ℕ) (p : Prism n) : p.properties := by
  sorry

end prism_properties_l729_72949


namespace wednesday_rainfall_l729_72928

/-- Rainfall recorded over three days -/
def total_rainfall : ℝ := 0.67

/-- Rainfall recorded on Monday -/
def monday_rainfall : ℝ := 0.17

/-- Rainfall recorded on Tuesday -/
def tuesday_rainfall : ℝ := 0.42

/-- Theorem stating that the rainfall on Wednesday is 0.08 cm -/
theorem wednesday_rainfall : 
  total_rainfall - (monday_rainfall + tuesday_rainfall) = 0.08 := by
  sorry

end wednesday_rainfall_l729_72928


namespace sqrt_6_equality_l729_72999

theorem sqrt_6_equality : (3 : ℝ) / Real.sqrt 6 = Real.sqrt 6 / 2 := by sorry

end sqrt_6_equality_l729_72999


namespace swimming_pool_count_l729_72985

theorem swimming_pool_count (total : ℕ) (garage : ℕ) (both : ℕ) (neither : ℕ) : 
  total = 90 → garage = 50 → both = 35 → neither = 35 → 
  ∃ (pool : ℕ), pool = 40 ∧ 
    total = garage + pool - both + neither :=
by
  sorry

end swimming_pool_count_l729_72985


namespace yola_past_weight_l729_72924

/-- Proves Yola's weight from 2 years ago given current weights and differences -/
theorem yola_past_weight 
  (yola_current : ℕ) 
  (wanda_yola_diff : ℕ) 
  (wanda_yola_past_diff : ℕ) 
  (h1 : yola_current = 220)
  (h2 : wanda_yola_diff = 30)
  (h3 : wanda_yola_past_diff = 80) : 
  yola_current - (wanda_yola_past_diff - wanda_yola_diff) = 170 := by
  sorry

#check yola_past_weight

end yola_past_weight_l729_72924


namespace fresh_produce_to_soda_ratio_l729_72926

/-- Proves that the ratio of fresh produce weight to soda weight is 2:1 --/
theorem fresh_produce_to_soda_ratio :
  let empty_truck_weight : ℕ := 12000
  let soda_crates : ℕ := 20
  let soda_crate_weight : ℕ := 50
  let dryers : ℕ := 3
  let dryer_weight : ℕ := 3000
  let loaded_truck_weight : ℕ := 24000
  let soda_weight := soda_crates * soda_crate_weight
  let dryers_weight := dryers * dryer_weight
  let fresh_produce_weight := loaded_truck_weight - (empty_truck_weight + soda_weight + dryers_weight)
  (fresh_produce_weight : ℚ) / soda_weight = 2 := by
  sorry

end fresh_produce_to_soda_ratio_l729_72926


namespace smallest_n_square_and_cube_l729_72986

theorem smallest_n_square_and_cube : 
  ∃ (n : ℕ), n > 0 ∧ 
  (∃ (k : ℕ), 5 * n = k^2) ∧ 
  (∃ (m : ℕ), 3 * n = m^3) ∧
  (∀ (x : ℕ), x > 0 ∧ 
    (∃ (y : ℕ), 5 * x = y^2) ∧ 
    (∃ (z : ℕ), 3 * x = z^3) → 
    x ≥ n) ∧
  n = 45 :=
by sorry

end smallest_n_square_and_cube_l729_72986


namespace negation_of_existence_inequality_l729_72991

theorem negation_of_existence_inequality (p : Prop) :
  (¬ (∃ x : ℝ, Real.exp x - x - 1 ≤ 0)) ↔ (∀ x : ℝ, Real.exp x - x - 1 > 0) := by
  sorry

end negation_of_existence_inequality_l729_72991


namespace sufficient_condition_problem_l729_72922

theorem sufficient_condition_problem (p q r s : Prop) 
  (h1 : p → q)
  (h2 : s → q)
  (h3 : q → r)
  (h4 : r → s) :
  p → s := by
  sorry

end sufficient_condition_problem_l729_72922


namespace complex_magnitude_problem_l729_72938

theorem complex_magnitude_problem (z w : ℂ) 
  (h1 : Complex.abs (2 * z - w) = 29)
  (h2 : Complex.abs (z + 2 * w) = 7)
  (h3 : Complex.abs (z + w) = 3) :
  Complex.abs z = 11 := by sorry

end complex_magnitude_problem_l729_72938


namespace a_value_is_negative_six_l729_72912

/-- The coefficient of x^4 in the expansion of (2+ax)(1-x)^6 -/
def coefficient (a : ℝ) : ℝ := 30 - 20 * a

/-- The theorem stating that a = -6 given the coefficient of x^4 is 150 -/
theorem a_value_is_negative_six : 
  ∃ a : ℝ, coefficient a = 150 ∧ a = -6 :=
sorry

end a_value_is_negative_six_l729_72912


namespace expression_simplification_l729_72961

theorem expression_simplification (x : ℝ) (h : x = Real.sqrt 2) :
  (x + 3)^2 + (x + 2)*(x - 2) - x*(x + 6) = 7 := by
  sorry

end expression_simplification_l729_72961


namespace regular_pentagon_angle_l729_72909

theorem regular_pentagon_angle (n : ℕ) (h : n = 5) :
  let central_angle := 360 / n
  2 * central_angle = 144 := by
  sorry

end regular_pentagon_angle_l729_72909


namespace B_2_2_equals_9_l729_72979

def B : ℕ → ℕ → ℕ
  | 0, n => n + 1
  | m + 1, 0 => B m 2
  | m + 1, n + 1 => B m (B (m + 1) n)

theorem B_2_2_equals_9 : B 2 2 = 9 := by
  sorry

end B_2_2_equals_9_l729_72979


namespace soccer_team_uniform_numbers_l729_72998

/-- A predicate to check if a number is a two-digit prime -/
def isTwoDigitPrime (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100 ∧ Nat.Prime n

/-- The uniform numbers of Emily, Fiona, and Grace -/
structure UniformNumbers where
  emily : ℕ
  fiona : ℕ
  grace : ℕ

/-- The conditions of the soccer team uniform numbers problem -/
structure SoccerTeamConditions (u : UniformNumbers) : Prop where
  emily_prime : isTwoDigitPrime u.emily
  fiona_prime : isTwoDigitPrime u.fiona
  grace_prime : isTwoDigitPrime u.grace
  emily_fiona_sum : u.emily + u.fiona = 23
  emily_grace_sum : u.emily + u.grace = 31

theorem soccer_team_uniform_numbers (u : UniformNumbers) 
  (h : SoccerTeamConditions u) : u.grace = 19 := by
  sorry

end soccer_team_uniform_numbers_l729_72998


namespace range_of_a_for_second_quadrant_l729_72957

-- Define the complex number z as a function of a
def z (a : ℝ) : ℂ := (1 - Complex.I) * (a + Complex.I)

-- Define what it means for a complex number to be in the second quadrant
def in_second_quadrant (w : ℂ) : Prop := w.re < 0 ∧ w.im > 0

-- State the theorem
theorem range_of_a_for_second_quadrant :
  ∀ a : ℝ, in_second_quadrant (z a) ↔ a < -1 := by sorry

end range_of_a_for_second_quadrant_l729_72957


namespace monotonicity_not_algorithmic_l729_72966

-- Define the concept of an algorithm
def Algorithm : Type := Unit

-- Define the problems
def SumProblem : Type := Unit
def LinearSystemProblem : Type := Unit
def CircleAreaProblem : Type := Unit
def MonotonicityProblem : Type := Unit

-- Define solvability by algorithm
def SolvableByAlgorithm (p : Type) : Prop := ∃ (a : Algorithm), True

-- State the theorem
theorem monotonicity_not_algorithmic :
  SolvableByAlgorithm SumProblem ∧
  SolvableByAlgorithm LinearSystemProblem ∧
  SolvableByAlgorithm CircleAreaProblem ∧
  ¬SolvableByAlgorithm MonotonicityProblem :=
sorry

end monotonicity_not_algorithmic_l729_72966


namespace binary_51_l729_72983

/-- The binary representation of a natural number -/
def toBinary (n : ℕ) : List Bool :=
  if n = 0 then [false] else
  let rec go (m : ℕ) : List Bool :=
    if m = 0 then [] else (m % 2 = 1) :: go (m / 2)
  go n

/-- Convert a list of booleans to a natural number, interpreting it as binary -/
def fromBinary (l : List Bool) : ℕ :=
  l.foldl (fun acc b => 2 * acc + if b then 1 else 0) 0

theorem binary_51 :
  toBinary 51 = [true, true, false, false, true, true] :=
by sorry

end binary_51_l729_72983


namespace oak_willow_difference_l729_72953

theorem oak_willow_difference (total_trees : ℕ) (willow_percent oak_percent : ℚ) : 
  total_trees = 712 →
  willow_percent = 34 / 100 →
  oak_percent = 45 / 100 →
  ⌊oak_percent * total_trees⌋ - ⌊willow_percent * total_trees⌋ = 78 := by
  sorry

end oak_willow_difference_l729_72953


namespace min_disks_for_given_files_l729_72920

/-- Represents the minimum number of disks needed to store files --/
def min_disks (total_files : ℕ) (disk_space : ℚ) 
  (files_1_2MB : ℕ) (files_0_9MB : ℕ) (files_0_5MB : ℕ) : ℕ :=
  sorry

/-- The main theorem stating the minimum number of disks needed --/
theorem min_disks_for_given_files : 
  min_disks 40 2 5 15 20 = 16 := by sorry

end min_disks_for_given_files_l729_72920


namespace calculate_expression_l729_72989

theorem calculate_expression (a b : ℝ) (hb : b ≠ 0) :
  4 * a * (3 * a^2 * b) / (2 * a * b) = 6 * a^2 := by
  sorry

end calculate_expression_l729_72989


namespace bowling_ball_weight_calculation_l729_72996

-- Define the weight of a canoe
def canoe_weight : ℚ := 32

-- Define the number of canoes and bowling balls
def num_canoes : ℕ := 4
def num_bowling_balls : ℕ := 5

-- Define the total weight of canoes
def total_canoe_weight : ℚ := num_canoes * canoe_weight

-- Define the weight of one bowling ball
def bowling_ball_weight : ℚ := total_canoe_weight / num_bowling_balls

-- Theorem statement
theorem bowling_ball_weight_calculation :
  bowling_ball_weight = 128 / 5 := by sorry

end bowling_ball_weight_calculation_l729_72996
