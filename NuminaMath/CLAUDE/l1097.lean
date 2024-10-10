import Mathlib

namespace cube_painting_theorem_l1097_109776

/-- Represents a cube with side length n -/
structure Cube (n : ℕ) where
  side_length : n > 0

/-- Represents the number of cubelets with exactly one painted face for each color -/
def single_color_cubelets (c : Cube n) : ℕ :=
  4 * 4 * 2

/-- The total number of cubelets with exactly one painted face -/
def total_single_color_cubelets (c : Cube n) : ℕ :=
  3 * single_color_cubelets c

theorem cube_painting_theorem (c : Cube 6) :
  total_single_color_cubelets c = 96 := by
  sorry


end cube_painting_theorem_l1097_109776


namespace cathys_total_money_l1097_109753

def cathys_money (initial_balance dad_contribution : ℕ) : ℕ :=
  initial_balance + dad_contribution + 2 * dad_contribution

theorem cathys_total_money :
  cathys_money 12 25 = 87 := by
  sorry

end cathys_total_money_l1097_109753


namespace sqrt_meaningful_iff_geq_one_l1097_109791

theorem sqrt_meaningful_iff_geq_one (x : ℝ) : 
  (∃ y : ℝ, y^2 = x - 1) ↔ x ≥ 1 := by
sorry

end sqrt_meaningful_iff_geq_one_l1097_109791


namespace special_line_properties_l1097_109739

/-- A line passing through (5, 2) with x-intercept twice the y-intercept -/
def special_line (x y : ℝ) : Prop :=
  x + 2 * y - 9 = 0

theorem special_line_properties :
  (special_line 5 2) ∧ 
  (∃ (a : ℝ), a ≠ 0 ∧ special_line (2*a) 0 ∧ special_line 0 a) :=
by sorry

end special_line_properties_l1097_109739


namespace water_tank_capacity_l1097_109719

theorem water_tank_capacity (c : ℝ) (h1 : c > 0) : 
  (c / 4 : ℝ) / c = 1 / 4 ∧ 
  ((c / 4 + 5) : ℝ) / c = 1 / 3 → 
  c = 60 := by
sorry

end water_tank_capacity_l1097_109719


namespace total_paintable_area_l1097_109720

def num_bedrooms : ℕ := 4
def room_length : ℝ := 14
def room_width : ℝ := 11
def room_height : ℝ := 9
def unpaintable_area : ℝ := 70

def wall_area (length width height : ℝ) : ℝ :=
  2 * (length * height + width * height)

def paintable_area (total_area unpaintable_area : ℝ) : ℝ :=
  total_area - unpaintable_area

theorem total_paintable_area :
  (num_bedrooms : ℝ) * paintable_area (wall_area room_length room_width room_height) unpaintable_area = 1520 := by
  sorry

end total_paintable_area_l1097_109720


namespace n_div_f_n_equals_5_for_625_n_div_f_n_equals_1_solutions_l1097_109770

/-- Represents a three-digit number -/
structure ThreeDigitNumber where
  hundreds : Nat
  tens : Nat
  ones : Nat
  h_hundreds : hundreds ≥ 1 ∧ hundreds ≤ 9
  h_tens : tens ≥ 0 ∧ tens ≤ 9
  h_ones : ones ≥ 0 ∧ ones ≤ 9

/-- Converts a ThreeDigitNumber to its numerical value -/
def ThreeDigitNumber.toNat (n : ThreeDigitNumber) : Nat :=
  100 * n.hundreds + 10 * n.tens + n.ones

/-- Defines the function f as described in the problem -/
def f (n : ThreeDigitNumber) : Nat :=
  let a := n.hundreds
  let b := n.tens
  let c := n.ones
  a + b + c + a * b + b * c + c * a + a * b * c

theorem n_div_f_n_equals_5_for_625 :
  let n : ThreeDigitNumber := ⟨6, 2, 5, by simp, by simp, by simp⟩
  (n.toNat : ℚ) / f n = 5 := by sorry

theorem n_div_f_n_equals_1_solutions :
  {n : ThreeDigitNumber | (n.toNat : ℚ) / f n = 1} =
  {⟨1, 9, 9, by simp, by simp, by simp⟩,
   ⟨2, 9, 9, by simp, by simp, by simp⟩,
   ⟨3, 9, 9, by simp, by simp, by simp⟩,
   ⟨4, 9, 9, by simp, by simp, by simp⟩,
   ⟨5, 9, 9, by simp, by simp, by simp⟩,
   ⟨6, 9, 9, by simp, by simp, by simp⟩,
   ⟨7, 9, 9, by simp, by simp, by simp⟩,
   ⟨8, 9, 9, by simp, by simp, by simp⟩,
   ⟨9, 9, 9, by simp, by simp, by simp⟩} := by sorry

end n_div_f_n_equals_5_for_625_n_div_f_n_equals_1_solutions_l1097_109770


namespace square_sum_problem_l1097_109774

theorem square_sum_problem (x y : ℕ+) 
  (h1 : x.val * y.val + x.val + y.val = 35)
  (h2 : x.val * y.val * (x.val + y.val) = 360) :
  x.val^2 + y.val^2 = 185 := by
  sorry

end square_sum_problem_l1097_109774


namespace summer_discount_is_fifty_percent_l1097_109755

def original_price : ℝ := 49
def final_price : ℝ := 14.50
def additional_discount : ℝ := 10

def summer_discount_percentage (d : ℝ) : Prop :=
  original_price * (1 - d / 100) - additional_discount = final_price

theorem summer_discount_is_fifty_percent : 
  summer_discount_percentage 50 := by sorry

end summer_discount_is_fifty_percent_l1097_109755


namespace fraction_value_l1097_109700

theorem fraction_value (t k : ℚ) (f : ℚ) : 
  t = f * (k - 32) → t = 35 → k = 95 → f = 5/9 := by
  sorry

end fraction_value_l1097_109700


namespace multiply_whole_and_mixed_number_l1097_109718

theorem multiply_whole_and_mixed_number : 8 * (9 + 2/5) = 75 + 1/5 := by
  sorry

end multiply_whole_and_mixed_number_l1097_109718


namespace six_digit_number_concatenation_divisibility_l1097_109758

theorem six_digit_number_concatenation_divisibility :
  ∃ (A B : ℕ), 
    A ≠ B ∧
    100000 ≤ A ∧ A < 1000000 ∧
    100000 ≤ B ∧ B < 1000000 ∧
    (10^6 * B + A) % (A * B) = 0 :=
by sorry

end six_digit_number_concatenation_divisibility_l1097_109758


namespace price_reduction_equivalence_l1097_109768

theorem price_reduction_equivalence : 
  let first_reduction : ℝ := 0.25
  let second_reduction : ℝ := 0.20
  let equivalent_reduction : ℝ := 1 - (1 - first_reduction) * (1 - second_reduction)
  equivalent_reduction = 0.40
  := by sorry

end price_reduction_equivalence_l1097_109768


namespace car_distribution_l1097_109762

/-- The number of cars produced annually by American carmakers -/
def total_cars : ℕ := 5650000

/-- The number of car suppliers -/
def num_suppliers : ℕ := 5

/-- The number of cars received by the first supplier -/
def first_supplier : ℕ := 1000000

/-- The number of cars received by the second supplier -/
def second_supplier : ℕ := first_supplier + 500000

/-- The number of cars received by the third supplier -/
def third_supplier : ℕ := first_supplier + second_supplier

/-- The number of cars received by each of the fourth and fifth suppliers -/
def fourth_fifth_supplier : ℕ := (total_cars - (first_supplier + second_supplier + third_supplier)) / 2

theorem car_distribution :
  fourth_fifth_supplier = 325000 :=
sorry

end car_distribution_l1097_109762


namespace child_weight_l1097_109777

theorem child_weight (total_weight : ℝ) (weight_difference : ℝ) (dog_weight_ratio : ℝ)
  (hw_total : total_weight = 180)
  (hw_diff : weight_difference = 162)
  (hw_ratio : dog_weight_ratio = 0.3) :
  ∃ (father_weight child_weight dog_weight : ℝ),
    father_weight + child_weight + dog_weight = total_weight ∧
    father_weight + child_weight = weight_difference + dog_weight ∧
    dog_weight = dog_weight_ratio * child_weight ∧
    child_weight = 30 := by
  sorry

end child_weight_l1097_109777


namespace product_trailing_zeros_l1097_109799

/-- The number of trailing zeros in a natural number -/
def trailingZeros (n : ℕ) : ℕ := sorry

/-- The product of 45 and 800 -/
def product : ℕ := 45 * 800

theorem product_trailing_zeros :
  trailingZeros product = 3 := by sorry

end product_trailing_zeros_l1097_109799


namespace number_of_girls_l1097_109790

/-- The number of girls in the group -/
def n : ℕ := sorry

/-- The average weight of the group before the new girl arrives -/
def A : ℝ := sorry

/-- The weight of the new girl -/
def W : ℝ := 80

theorem number_of_girls :
  (n * A = n * A - 55 + W) ∧ (n * (A + 1) = n * A - 55 + W) → n = 25 := by
  sorry

end number_of_girls_l1097_109790


namespace digit_sum_theorem_l1097_109749

theorem digit_sum_theorem (A B C D : ℕ) : 
  A ≠ 0 →
  A < 10 → B < 10 → C < 10 → D < 10 →
  1000 * A + 100 * B + 10 * C + D = (10 * C + D)^2 - (10 * A + B)^2 →
  A + B + C + D = 21 := by
sorry

end digit_sum_theorem_l1097_109749


namespace sqrt_three_irrational_sqrt_three_only_irrational_l1097_109736

theorem sqrt_three_irrational :
  ¬ ∃ (p q : ℤ), q ≠ 0 ∧ Real.sqrt 3 = (p : ℚ) / (q : ℚ) :=
by
  sorry

-- Definitions for rational numbers in the problem
def zero_rational : ℚ := 0
def one_point_five_rational : ℚ := 3/2
def negative_two_rational : ℚ := -2

-- Theorem stating that √3 is the only irrational number among the given options
theorem sqrt_three_only_irrational :
  ¬ (∃ (p q : ℤ), q ≠ 0 ∧ Real.sqrt 3 = (p : ℚ) / (q : ℚ)) ∧
  (∃ (p q : ℤ), q ≠ 0 ∧ zero_rational = (p : ℚ) / (q : ℚ)) ∧
  (∃ (p q : ℤ), q ≠ 0 ∧ one_point_five_rational = (p : ℚ) / (q : ℚ)) ∧
  (∃ (p q : ℤ), q ≠ 0 ∧ negative_two_rational = (p : ℚ) / (q : ℚ)) :=
by
  sorry

end sqrt_three_irrational_sqrt_three_only_irrational_l1097_109736


namespace boat_speed_in_still_water_boat_speed_is_16_l1097_109709

/-- The speed of a boat in still water, given downstream travel information and stream speed. -/
theorem boat_speed_in_still_water
  (stream_speed : ℝ)
  (downstream_distance : ℝ)
  (downstream_time : ℝ)
  (h1 : stream_speed = 4)
  (h2 : downstream_distance = 60)
  (h3 : downstream_time = 3)
  : ℝ :=
  let downstream_speed := downstream_distance / downstream_time
  let boat_speed := downstream_speed - stream_speed
  16

/-- Proof that the boat's speed in still water is 16 km/hr -/
theorem boat_speed_is_16
  (stream_speed : ℝ)
  (downstream_distance : ℝ)
  (downstream_time : ℝ)
  (h1 : stream_speed = 4)
  (h2 : downstream_distance = 60)
  (h3 : downstream_time = 3)
  : boat_speed_in_still_water stream_speed downstream_distance downstream_time h1 h2 h3 = 16 := by
  sorry

end boat_speed_in_still_water_boat_speed_is_16_l1097_109709


namespace alpha_beta_sum_l1097_109787

theorem alpha_beta_sum (α β : ℝ) : 
  (∀ x : ℝ, x ≠ 30 → (x - α) / (x + β) = (x^2 - 120*x + 3600) / (x^2 + 70*x - 2300)) →
  α + β = 137 := by
sorry

end alpha_beta_sum_l1097_109787


namespace inverse_function_problem_l1097_109754

-- Define the function f and its inverse
def f : ℝ → ℝ := sorry

-- State the theorem
theorem inverse_function_problem (h : ∀ x > 0, f⁻¹ x = x^2) : f 4 = 2 := by
  sorry

end inverse_function_problem_l1097_109754


namespace square_of_difference_l1097_109733

theorem square_of_difference (y : ℝ) (h : y^2 ≥ 49) :
  (7 - Real.sqrt (y^2 - 49))^2 = y^2 - 14 * Real.sqrt (y^2 - 49) := by
  sorry

end square_of_difference_l1097_109733


namespace roots_sum_product_l1097_109761

theorem roots_sum_product (α' β' : ℝ) : 
  (α' + β' = 5) → (α' * β' = 6) → 3 * α'^3 + 4 * β'^2 = 271 := by
  sorry

end roots_sum_product_l1097_109761


namespace fixed_point_theorem_a_value_theorem_minimum_point_theorem_l1097_109728

noncomputable section

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x - a * x^2 - 2 * x

-- Define the derivative of f
def f_deriv (a : ℝ) (x : ℝ) : ℝ := Real.exp x - 2 * a * x - 2

theorem fixed_point_theorem (a : ℝ) :
  f a 0 = 1 := by sorry

theorem a_value_theorem (a : ℝ) :
  (∀ x, f_deriv a x ≥ -a * x - 1) → a = 1 := by sorry

theorem minimum_point_theorem :
  ∃ x₀, (∀ x, f 1 x ≥ f 1 x₀) ∧ -2 < f 1 x₀ ∧ f 1 x₀ < -1/4 := by sorry

end

end fixed_point_theorem_a_value_theorem_minimum_point_theorem_l1097_109728


namespace geometric_sequence_property_l1097_109722

/-- Given a geometric sequence {a_n} where a₄ + a₆ = 3, prove that a₅(a₃ + 2a₅ + a₇) = 9 -/
theorem geometric_sequence_property (a : ℕ → ℝ) (q : ℝ) :
  (∀ n, a (n + 1) = a n * q) →  -- a_n is a geometric sequence with common ratio q
  a 4 + a 6 = 3 →               -- given condition
  a 5 * (a 3 + 2 * a 5 + a 7) = 9 := by
sorry

end geometric_sequence_property_l1097_109722


namespace f_value_at_sqrt3_over_2_main_theorem_l1097_109734

-- Define the function f
def f (x : ℝ) : ℝ := 1 - 2 * x^2

-- Theorem statement
theorem f_value_at_sqrt3_over_2 : f (Real.sqrt 3 / 2) = -1/2 := by
  sorry

-- The main theorem that corresponds to the original problem
theorem main_theorem : 
  (∀ x, f (Real.sin x) = 1 - 2 * (Real.sin x)^2) → f (Real.sqrt 3 / 2) = -1/2 := by
  sorry

end f_value_at_sqrt3_over_2_main_theorem_l1097_109734


namespace draw_all_red_probability_l1097_109737

-- Define the number of red and green chips
def num_red : ℕ := 3
def num_green : ℕ := 2

-- Define the total number of chips
def total_chips : ℕ := num_red + num_green

-- Define the probability of drawing all red chips before both green chips
def prob_all_red : ℚ := 3 / 10

-- Theorem statement
theorem draw_all_red_probability :
  prob_all_red = (num_red * (num_red - 1) * (num_red - 2)) / 
    (total_chips * (total_chips - 1) * (total_chips - 2)) :=
by sorry

end draw_all_red_probability_l1097_109737


namespace rate_increase_factor_l1097_109702

/-- Reaction rate equation -/
def reaction_rate (k : ℝ) (C_CO : ℝ) (C_O2 : ℝ) : ℝ :=
  k * C_CO^2 * C_O2

/-- Theorem: When concentrations triple, rate increases by factor of 27 -/
theorem rate_increase_factor (k : ℝ) (C_CO : ℝ) (C_O2 : ℝ) :
  reaction_rate k (3 * C_CO) (3 * C_O2) = 27 * reaction_rate k C_CO C_O2 := by
  sorry


end rate_increase_factor_l1097_109702


namespace mango_purchase_l1097_109724

/-- The amount of grapes purchased in kg -/
def grapes : ℕ := 8

/-- The price of grapes per kg -/
def grape_price : ℕ := 70

/-- The price of mangoes per kg -/
def mango_price : ℕ := 60

/-- The total amount paid to the shopkeeper -/
def total_paid : ℕ := 1100

/-- The amount of mangoes purchased in kg -/
def mangoes : ℕ := (total_paid - grapes * grape_price) / mango_price

theorem mango_purchase : mangoes = 9 := by
  sorry

end mango_purchase_l1097_109724


namespace point_between_l1097_109726

theorem point_between (a b c : ℚ) (h1 : a ≠ b) (h2 : b ≠ c) (h3 : a ≠ c) 
  (h4 : |a - b| + |b - c| = |a - c|) : 
  (a < b ∧ b < c) ∨ (c < b ∧ b < a) :=
sorry

end point_between_l1097_109726


namespace growth_rate_is_25_percent_l1097_109723

/-- The average monthly growth rate of new 5G physical base stations -/
def average_growth_rate : ℝ := 0.25

/-- The number of new 5G physical base stations opened in January -/
def january_stations : ℕ := 1600

/-- The number of new 5G physical base stations opened in March -/
def march_stations : ℕ := 2500

/-- Theorem stating that the average monthly growth rate is 25% -/
theorem growth_rate_is_25_percent :
  january_stations * (1 + average_growth_rate)^2 = march_stations := by
  sorry

#check growth_rate_is_25_percent

end growth_rate_is_25_percent_l1097_109723


namespace line_equation_and_distance_l1097_109785

-- Define the point P
def P : ℝ × ℝ := (-1, 4)

-- Define line l₂
def l₂ (x y : ℝ) : Prop := 2 * x - y + 5 = 0

-- Define line l₁
def l₁ (x y : ℝ) : Prop := 2 * x - y + 6 = 0

-- Define line l₃
def l₃ (x y m : ℝ) : Prop := 4 * x - 2 * y + m = 0

-- State the theorem
theorem line_equation_and_distance (m : ℝ) : 
  (∀ x y, l₁ x y ↔ l₂ (x + 1/2) (y + 1/2)) ∧ -- l₁ is parallel to l₂
  l₁ P.1 P.2 ∧ -- P lies on l₁
  (∃ d, d = 2 * Real.sqrt 5 ∧ 
   d = |m - 12| / Real.sqrt (4^2 + (-2)^2)) → -- Distance between l₁ and l₃
  (m = -8 ∨ m = 32) := by
sorry

end line_equation_and_distance_l1097_109785


namespace investment_problem_l1097_109725

/-- Investment problem -/
theorem investment_problem (x y : ℕ) (profit_ratio : Rat) (y_investment : ℕ) : 
  profit_ratio = 2 / 6 →
  y_investment = 15000 →
  x = 5000 :=
by
  sorry

end investment_problem_l1097_109725


namespace tina_july_savings_l1097_109789

/-- Represents Tina's savings and spending --/
structure TinaSavings where
  june : ℕ
  july : ℕ
  august : ℕ
  books_spent : ℕ
  shoes_spent : ℕ
  remaining : ℕ

/-- Theorem stating that Tina saved $14 in July --/
theorem tina_july_savings (s : TinaSavings) 
  (h1 : s.june = 27)
  (h2 : s.august = 21)
  (h3 : s.books_spent = 5)
  (h4 : s.shoes_spent = 17)
  (h5 : s.remaining = 40)
  (h6 : s.june + s.july + s.august = s.books_spent + s.shoes_spent + s.remaining) :
  s.july = 14 := by
  sorry


end tina_july_savings_l1097_109789


namespace only_students_far_from_school_not_set_l1097_109731

-- Define the groups of objects
def right_angled_triangles : Set (Set ℝ) := sorry
def points_on_unit_circle : Set (ℝ × ℝ) := sorry
def students_far_from_school : Set String := sorry
def homeroom_teachers : Set String := sorry

-- Define a predicate for well-defined sets
def is_well_defined_set (S : Set α) : Prop := sorry

-- Theorem statement
theorem only_students_far_from_school_not_set :
  is_well_defined_set right_angled_triangles ∧
  is_well_defined_set points_on_unit_circle ∧
  ¬ is_well_defined_set students_far_from_school ∧
  is_well_defined_set homeroom_teachers :=
sorry

end only_students_far_from_school_not_set_l1097_109731


namespace no_prime_roots_l1097_109705

/-- A prime number is a natural number greater than 1 that has no positive divisors other than 1 and itself -/
def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

/-- The quadratic equation x^2 - 108x + k = 0 -/
def quadraticEquation (x k : ℝ) : Prop := x^2 - 108*x + k = 0

/-- Both roots of the quadratic equation are prime numbers -/
def bothRootsPrime (k : ℝ) : Prop :=
  ∃ (p q : ℕ), isPrime p ∧ isPrime q ∧ 
    (∀ x : ℝ, quadraticEquation x k ↔ x = p ∨ x = q)

/-- There are no values of k for which both roots of the quadratic equation are prime -/
theorem no_prime_roots : ¬∃ k : ℝ, bothRootsPrime k := by sorry

end no_prime_roots_l1097_109705


namespace one_third_of_seven_times_nine_l1097_109706

theorem one_third_of_seven_times_nine : (1 / 3 : ℚ) * (7 * 9) = 21 := by
  sorry

end one_third_of_seven_times_nine_l1097_109706


namespace opinion_change_difference_l1097_109769

theorem opinion_change_difference (initial_yes initial_no final_yes final_no : ℝ) :
  initial_yes = 30 →
  initial_no = 70 →
  final_yes = 60 →
  final_no = 40 →
  initial_yes + initial_no = 100 →
  final_yes + final_no = 100 →
  ∃ (min_change max_change : ℝ),
    (min_change ≤ max_change) ∧
    (∀ (change : ℝ), change ≥ min_change ∧ change ≤ max_change →
      ∃ (yes_to_no no_to_yes : ℝ),
        yes_to_no ≥ 0 ∧
        no_to_yes ≥ 0 ∧
        yes_to_no + no_to_yes = change ∧
        initial_yes - yes_to_no + no_to_yes = final_yes) ∧
    (max_change - min_change = 30) :=
by sorry

end opinion_change_difference_l1097_109769


namespace positive_difference_of_solutions_l1097_109742

-- Define the equation
def equation (x : ℝ) : Prop := (9 - x^2 / 3)^(1/3) = 3

-- Define the set of solutions
def solutions : Set ℝ := {x : ℝ | equation x}

-- Theorem statement
theorem positive_difference_of_solutions :
  ∃ (x y : ℝ), x ∈ solutions ∧ y ∈ solutions ∧ x ≠ y ∧ |x - y| = 18 * Real.sqrt 2 :=
sorry

end positive_difference_of_solutions_l1097_109742


namespace lassis_from_twenty_fruit_l1097_109730

/-- The number of lassis that can be made given a certain number of fruit units -/
def lassis_from_fruit (fruit_units : ℕ) : ℚ :=
  (9 : ℚ) / 4 * fruit_units

/-- Theorem stating that 45 lassis can be made from 20 fruit units -/
theorem lassis_from_twenty_fruit : lassis_from_fruit 20 = 45 := by
  sorry

end lassis_from_twenty_fruit_l1097_109730


namespace buddy_cards_l1097_109716

/-- Calculates the number of baseball cards Buddy has on Saturday --/
def saturday_cards (initial : ℕ) : ℕ :=
  let tuesday := initial - (initial * 30 / 100)
  let wednesday := tuesday + (tuesday * 20 / 100)
  let thursday := wednesday - (wednesday / 4)
  let friday := thursday + (thursday / 3)
  friday + (friday * 2)

/-- Theorem stating that Buddy will have 252 cards on Saturday --/
theorem buddy_cards : saturday_cards 100 = 252 := by
  sorry

end buddy_cards_l1097_109716


namespace intersection_of_A_and_B_l1097_109712

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | 1 ≤ x ∧ x < 4}
def B : Set ℝ := {x : ℝ | -2 ≤ x ∧ x < 2}

-- State the theorem
theorem intersection_of_A_and_B :
  A ∩ B = {x : ℝ | 1 ≤ x ∧ x < 2} :=
by sorry

end intersection_of_A_and_B_l1097_109712


namespace circle_tangent_to_line_l1097_109792

theorem circle_tangent_to_line (r : ℝ) (h : r = Real.sqrt 5) :
  ∃ (c1 c2 : ℝ × ℝ),
    c1.2 = 0 ∧ c2.2 = 0 ∧
    (∀ (x y : ℝ), (x - c1.1)^2 + (y - c1.2)^2 = r^2 ↔ (x + 2*y)^2 = 5) ∧
    (∀ (x y : ℝ), (x - c2.1)^2 + (y - c2.2)^2 = r^2 ↔ (x + 2*y)^2 = 5) ∧
    c1 = (5, 0) ∧ c2 = (-5, 0) := by
  sorry

end circle_tangent_to_line_l1097_109792


namespace gcf_of_lcms_l1097_109738

def GCF (a b : ℕ) : ℕ := Nat.gcd a b

def LCM (c d : ℕ) : ℕ := Nat.lcm c d

theorem gcf_of_lcms : GCF (LCM 18 30) (LCM 10 45) = 90 := by
  sorry

end gcf_of_lcms_l1097_109738


namespace sum_seventh_eighth_l1097_109717

/-- A geometric sequence with the given properties -/
structure GeometricSequence where
  a : ℕ → ℝ
  is_geometric : ∀ n : ℕ, a (n + 1) / a n = a 2 / a 1
  sum_first_two : a 1 + a 2 = 16
  sum_third_fourth : a 3 + a 4 = 32

/-- The sum of the 7th and 8th terms is 128 -/
theorem sum_seventh_eighth (seq : GeometricSequence) : seq.a 7 + seq.a 8 = 128 := by
  sorry

end sum_seventh_eighth_l1097_109717


namespace polynomial_coefficient_problem_l1097_109708

theorem polynomial_coefficient_problem (a b : ℝ) : 
  (∀ x : ℝ, (x^2 + a*x + b) * (2*x^2 - 3*x - 1) = 
    2*x^4 + (-5)*x^3 + (-6)*x^2 + ((-3*b - a)*x - b)) → 
  a = -1 ∧ b = -4 := by
  sorry

end polynomial_coefficient_problem_l1097_109708


namespace stamps_per_page_l1097_109783

theorem stamps_per_page (book1 book2 book3 : ℕ) 
  (h1 : book1 = 924) 
  (h2 : book2 = 1386) 
  (h3 : book3 = 1848) : 
  Nat.gcd book1 (Nat.gcd book2 book3) = 462 := by
  sorry

end stamps_per_page_l1097_109783


namespace d_must_be_positive_l1097_109740

theorem d_must_be_positive
  (a b c d e f : ℤ)
  (h1 : a * b + c * d * e * f < 0)
  (h2 : a < 0)
  (h3 : b < 0)
  (h4 : c < 0)
  (h5 : e < 0)
  (h6 : f < 0) :
  d > 0 := by
sorry

end d_must_be_positive_l1097_109740


namespace age_difference_l1097_109701

/-- Given three people A, B, and C, where C is 12 years younger than A,
    prove that the total age of A and B is 12 years more than the total age of B and C. -/
theorem age_difference (A B C : ℕ) (h : C = A - 12) :
  (A + B) - (B + C) = 12 :=
sorry

end age_difference_l1097_109701


namespace mirror_pieces_l1097_109773

theorem mirror_pieces (total : ℕ) (swept : ℕ) (stolen : ℕ) (picked : ℕ) : 
  total = 60 →
  swept = total / 2 →
  stolen = 3 →
  picked = (total - swept - stolen) / 3 →
  picked = 9 := by
sorry

end mirror_pieces_l1097_109773


namespace min_value_of_squares_l1097_109788

theorem min_value_of_squares (a b t : ℝ) (h : 2 * a + b = 2 * t) :
  ∃ (min : ℝ), min = (4 * t^2) / 5 ∧ ∀ (x y : ℝ), 2 * x + y = 2 * t → x^2 + y^2 ≥ min :=
sorry

end min_value_of_squares_l1097_109788


namespace samantha_birth_year_proof_l1097_109715

/-- The year of the first AMC 8 -/
def first_amc8_year : ℕ := 1983

/-- The number of AMC 8 contests Samantha has taken -/
def samantha_amc8_count : ℕ := 9

/-- Samantha's age when she took her last AMC 8 -/
def samantha_age : ℕ := 13

/-- The year Samantha was born -/
def samantha_birth_year : ℕ := 1978

theorem samantha_birth_year_proof :
  samantha_birth_year = first_amc8_year + samantha_amc8_count - 1 - samantha_age :=
by sorry

end samantha_birth_year_proof_l1097_109715


namespace mark_and_carolyn_money_l1097_109786

theorem mark_and_carolyn_money (mark_money : ℚ) (carolyn_money : ℚ) : 
  mark_money = 5/8 → carolyn_money = 2/5 → mark_money + carolyn_money = 1.025 := by
  sorry

end mark_and_carolyn_money_l1097_109786


namespace allan_balloons_l1097_109765

/-- The number of balloons Allan initially brought to the park -/
def initial_balloons : ℕ := 5

/-- The number of balloons Allan bought at the park -/
def bought_balloons : ℕ := 3

/-- The total number of balloons Allan brought to the park -/
def total_balloons : ℕ := initial_balloons + bought_balloons

theorem allan_balloons : total_balloons = 8 := by
  sorry

end allan_balloons_l1097_109765


namespace twenty_five_percent_less_than_80_l1097_109756

theorem twenty_five_percent_less_than_80 (x : ℝ) : x + (1/4) * x = 80 - (1/4) * 80 → x = 48 := by
  sorry

end twenty_five_percent_less_than_80_l1097_109756


namespace baking_cookies_theorem_l1097_109743

/-- The number of pans of cookies that can be baked in a given time -/
def pans_of_cookies (total_time minutes_per_pan : ℕ) : ℕ :=
  total_time / minutes_per_pan

theorem baking_cookies_theorem (total_time minutes_per_pan : ℕ) 
  (h1 : total_time = 28) (h2 : minutes_per_pan = 7) : 
  pans_of_cookies total_time minutes_per_pan = 4 := by
  sorry

end baking_cookies_theorem_l1097_109743


namespace power_of_product_l1097_109771

theorem power_of_product (a : ℝ) : (3 * a)^2 = 9 * a^2 := by
  sorry

end power_of_product_l1097_109771


namespace can_identification_theorem_l1097_109721

-- Define the type for our weighing results
inductive WeighResult
| Heavy
| Medium
| Light

def WeighSequence := List WeighResult

theorem can_identification_theorem (n : ℕ) (weights : Fin n → ℝ) 
  (h_n : n = 80) (h_distinct : ∀ i j : Fin n, i ≠ j → weights i ≠ weights j) :
  (∃ (f : Fin n → WeighSequence), 
    (∀ seq, (∃ i, f i = seq) → seq.length ≤ 4) ∧ 
    (∀ i j : Fin n, i ≠ j → f i ≠ f j)) ∧ 
  (¬ ∃ (f : Fin n → WeighSequence), 
    (∀ seq, (∃ i, f i = seq) → seq.length ≤ 3) ∧ 
    (∀ i j : Fin n, i ≠ j → f i ≠ f j)) := by
  sorry


end can_identification_theorem_l1097_109721


namespace area_of_right_triangle_abc_l1097_109796

/-- Right triangle ABC with specific properties -/
structure RightTriangleABC where
  -- A, B, C are points in the plane
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  -- ABC is a right triangle with right angle at C
  is_right_triangle : (A.1 - C.1) * (B.1 - C.1) + (A.2 - C.2) * (B.2 - C.2) = 0
  -- Length of AB is 50
  hypotenuse_length : (A.1 - B.1)^2 + (A.2 - B.2)^2 = 50^2
  -- Median through A lies on y = x + 2
  median_A : ∃ t : ℝ, (A.1 + C.1) / 2 = t ∧ (A.2 + C.2) / 2 = t + 2
  -- Median through B lies on y = 3x + 1
  median_B : ∃ t : ℝ, (B.1 + C.1) / 2 = t ∧ (B.2 + C.2) / 2 = 3 * t + 1

/-- The area of the right triangle ABC is 250/3 -/
theorem area_of_right_triangle_abc (t : RightTriangleABC) : 
  abs ((t.A.1 - t.C.1) * (t.B.2 - t.C.2) - (t.B.1 - t.C.1) * (t.A.2 - t.C.2)) / 2 = 250 / 3 := by
  sorry

end area_of_right_triangle_abc_l1097_109796


namespace lindas_savings_l1097_109704

theorem lindas_savings (savings : ℝ) (tv_cost : ℝ) : 
  tv_cost = 240 →
  (1 / 4 : ℝ) * savings = tv_cost →
  savings = 960 := by
  sorry

end lindas_savings_l1097_109704


namespace range_of_f_l1097_109784

-- Define the function f
def f (x : ℝ) : ℝ := 2 * x^2 - 4 * x + 12

-- Statement to prove
theorem range_of_f :
  Set.range f = Set.Ici 10 := by
  sorry

end range_of_f_l1097_109784


namespace min_sum_of_digits_of_sum_l1097_109767

-- Define a function to calculate the sum of digits
def sumOfDigits (n : ℕ) : ℕ := sorry

-- Define the theorem
theorem min_sum_of_digits_of_sum (A B : ℕ) 
  (hA : sumOfDigits A = 59) 
  (hB : sumOfDigits B = 77) : 
  ∃ (C : ℕ), C = A + B ∧ sumOfDigits C = 1 ∧ 
  ∀ (D : ℕ), D = A + B → sumOfDigits D ≥ 1 := by
  sorry

end min_sum_of_digits_of_sum_l1097_109767


namespace max_value_complex_l1097_109729

theorem max_value_complex (z : ℂ) (h : Complex.abs z = 2) :
  Complex.abs ((z - 2)^3 * (z + 1)) ≤ 8 * Real.sqrt 5 := by sorry

end max_value_complex_l1097_109729


namespace rachels_father_age_rachels_father_age_at_25_l1097_109747

/-- Rachel's age problem -/
theorem rachels_father_age (rachel_age : ℕ) (grandfather_age_multiplier : ℕ) 
  (father_age_difference : ℕ) (rachel_future_age : ℕ) : ℕ :=
  let grandfather_age := rachel_age * grandfather_age_multiplier
  let mother_age := grandfather_age / 2
  let father_age := mother_age + father_age_difference
  let years_passed := rachel_future_age - rachel_age
  father_age + years_passed

/-- Proof of Rachel's father's age when she is 25 -/
theorem rachels_father_age_at_25 : 
  rachels_father_age 12 7 5 25 = 60 := by
  sorry

end rachels_father_age_rachels_father_age_at_25_l1097_109747


namespace derivative_of_sin_minus_cos_l1097_109775

theorem derivative_of_sin_minus_cos (α : ℝ) :
  let f : ℝ → ℝ := λ x => Real.sin α - Real.cos x
  (deriv f) α = Real.sin α :=
by
  sorry

end derivative_of_sin_minus_cos_l1097_109775


namespace mary_added_four_peanuts_l1097_109757

/-- The number of peanuts Mary added to the box -/
def peanuts_added (initial final : ℕ) : ℕ := final - initial

/-- Theorem stating that Mary added 4 peanuts to the box -/
theorem mary_added_four_peanuts :
  peanuts_added 4 8 = 4 := by sorry

end mary_added_four_peanuts_l1097_109757


namespace abs_neg_sqrt_16_plus_9_l1097_109748

theorem abs_neg_sqrt_16_plus_9 : |-(Real.sqrt 16) + 9| = 5 := by sorry

end abs_neg_sqrt_16_plus_9_l1097_109748


namespace number_puzzle_l1097_109779

theorem number_puzzle : ∃! x : ℝ, 3 * (2 * x + 9) = 69 := by sorry

end number_puzzle_l1097_109779


namespace roots_of_g_l1097_109751

def f (a b x : ℝ) : ℝ := a * x - b

def g (a b x : ℝ) : ℝ := b * x^2 + 3 * a * x

theorem roots_of_g (a b : ℝ) (h : f a b 3 = 0) :
  {x : ℝ | g a b x = 0} = {-1, 0} := by sorry

end roots_of_g_l1097_109751


namespace circle_chord_distance_l1097_109778

theorem circle_chord_distance (r : ℝ) (AB AC BC : ℝ) : 
  r = 10 →
  AB = 2 * r →
  AC = 12 →
  AB^2 = AC^2 + BC^2 →
  BC = 16 := by
sorry

end circle_chord_distance_l1097_109778


namespace score_order_l1097_109713

/-- Represents the scores of contestants in a math competition. -/
structure Scores where
  alice : ℕ
  brian : ℕ
  cindy : ℕ
  donna : ℕ

/-- Conditions for the math competition scores. -/
def valid_scores (s : Scores) : Prop :=
  -- Brian + Donna = Alice + Cindy
  s.brian + s.donna = s.alice + s.cindy ∧
  -- If Brian and Cindy were swapped, Alice + Cindy > Brian + Donna + 10
  s.alice + s.brian > s.cindy + s.donna + 10 ∧
  -- Donna > Brian + Cindy + 20
  s.donna > s.brian + s.cindy + 20 ∧
  -- Total score is 200
  s.alice + s.brian + s.cindy + s.donna = 200

/-- The theorem to prove -/
theorem score_order (s : Scores) (h : valid_scores s) :
  s.donna > s.alice ∧ s.alice > s.brian ∧ s.brian > s.cindy := by
  sorry

end score_order_l1097_109713


namespace series_sum_equals_three_fourths_l1097_109735

/-- The sum of the series Σ(k=0 to ∞) (3^(2^k) / (6^(2^k) - 2)) is equal to 3/4 -/
theorem series_sum_equals_three_fourths : 
  ∑' k : ℕ, (3 : ℝ)^(2^k) / ((6 : ℝ)^(2^k) - 2) = 3/4 := by sorry

end series_sum_equals_three_fourths_l1097_109735


namespace polynomial_division_l1097_109750

theorem polynomial_division (x : ℝ) :
  x^6 + 3 = (x - 2) * (x^5 + 2*x^4 + 4*x^3 + 8*x^2 + 16*x + 32) + 67 := by
  sorry

end polynomial_division_l1097_109750


namespace greatest_npmm_l1097_109772

/-- Represents a three-digit number with equal digits -/
def ThreeEqualDigits (n : ℕ) : Prop :=
  ∃ d : ℕ, d < 10 ∧ n = d * 100 + d * 10 + d

/-- Represents a one-digit number -/
def OneDigit (n : ℕ) : Prop := n < 10 ∧ n > 0

/-- Represents a four-digit number -/
def FourDigits (n : ℕ) : Prop := n ≥ 1000 ∧ n < 10000

/-- The main theorem -/
theorem greatest_npmm :
  ∀ MMM M NPMM : ℕ,
    ThreeEqualDigits MMM →
    OneDigit M →
    FourDigits NPMM →
    MMM * M = NPMM →
    NPMM ≤ 3996 :=
by
  sorry

#check greatest_npmm

end greatest_npmm_l1097_109772


namespace roots_of_quadratic_and_quartic_l1097_109707

theorem roots_of_quadratic_and_quartic (α β p q : ℝ) : 
  (α^2 - 3*α + 1 = 0) ∧ 
  (β^2 - 3*β + 1 = 0) ∧ 
  (α^4 - p*α^2 + q = 0) ∧ 
  (β^4 - p*β^2 + q = 0) →
  p = 7 ∧ q = 1 := by
sorry

end roots_of_quadratic_and_quartic_l1097_109707


namespace complex_multiplication_result_l1097_109752

theorem complex_multiplication_result : 
  (2 + 2 * Complex.I) * (1 - 2 * Complex.I) = 6 - 2 * Complex.I := by
  sorry

end complex_multiplication_result_l1097_109752


namespace maximum_mark_calculation_l1097_109759

def passing_threshold (max_mark : ℝ) : ℝ := 0.33 * max_mark

theorem maximum_mark_calculation (student_marks : ℝ) (failed_by : ℝ) 
  (h1 : student_marks = 125)
  (h2 : failed_by = 40)
  (h3 : passing_threshold (student_marks + failed_by) = student_marks + failed_by) :
  student_marks + failed_by = 500 := by
  sorry

end maximum_mark_calculation_l1097_109759


namespace geometric_sequence_sum_l1097_109745

/-- A geometric sequence with negative terms -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_sum (a : ℕ → ℝ) :
  GeometricSequence a →
  (∀ n, a n < 0) →
  a 2 * a 4 + 2 * a 3 * a 5 + a 4 * a 6 = 25 →
  a 3 + a 5 = -5 := by
sorry

end geometric_sequence_sum_l1097_109745


namespace germs_left_is_thirty_percent_l1097_109727

/-- The percentage of germs killed by spray A -/
def spray_a_kill_rate : ℝ := 50

/-- The percentage of germs killed by spray B -/
def spray_b_kill_rate : ℝ := 25

/-- The percentage of germs killed by both sprays -/
def overlap_kill_rate : ℝ := 5

/-- The percentage of germs left after using both sprays -/
def germs_left : ℝ := 100 - (spray_a_kill_rate + spray_b_kill_rate - overlap_kill_rate)

theorem germs_left_is_thirty_percent :
  germs_left = 30 := by sorry

end germs_left_is_thirty_percent_l1097_109727


namespace arithmetic_progression_cubic_coeff_conditions_l1097_109793

/-- A cubic polynomial with coefficients a, b, c whose roots form an arithmetic progression -/
structure ArithmeticProgressionCubic where
  a : ℝ
  b : ℝ
  c : ℝ
  roots_in_ap : ∃ (r₁ r₂ r₃ : ℝ), r₁ < r₂ ∧ r₂ < r₃ ∧
    r₂ - r₁ = r₃ - r₂ ∧
    r₁ + r₂ + r₃ = -a ∧
    r₁ * r₂ + r₁ * r₃ + r₂ * r₃ = b ∧
    r₁ * r₂ * r₃ = -c

/-- The coefficients of a cubic polynomial with roots in arithmetic progression satisfy specific conditions -/
theorem arithmetic_progression_cubic_coeff_conditions (p : ArithmeticProgressionCubic) :
  27 * p.c = 3 * p.a * p.b - 2 * p.a^3 ∧ 3 * p.b ≤ p.a^2 := by
  sorry

end arithmetic_progression_cubic_coeff_conditions_l1097_109793


namespace f_negative_three_l1097_109714

-- Define f as a function from ℝ to ℝ
variable (f : ℝ → ℝ)

-- f is an even function
def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

-- For any positive number x, f(2+x) = -2f(2-x)
def satisfies_condition (f : ℝ → ℝ) : Prop :=
  ∀ x > 0, f (2 + x) = -2 * f (2 - x)

-- Main theorem
theorem f_negative_three (h1 : is_even f) (h2 : satisfies_condition f) (h3 : f (-1) = 4) :
  f (-3) = -8 := by
  sorry


end f_negative_three_l1097_109714


namespace probability_at_least_one_of_each_color_l1097_109798

theorem probability_at_least_one_of_each_color (white red yellow drawn : ℕ) 
  (hw : white = 5) (hr : red = 4) (hy : yellow = 3) (hd : drawn = 4) :
  let total := white + red + yellow
  let total_ways := Nat.choose total drawn
  let favorable_ways := 
    Nat.choose white 2 * Nat.choose red 1 * Nat.choose yellow 1 +
    Nat.choose white 1 * Nat.choose red 2 * Nat.choose yellow 1 +
    Nat.choose white 1 * Nat.choose red 1 * Nat.choose yellow 2
  (favorable_ways : ℚ) / total_ways = 6 / 11 := by
sorry

end probability_at_least_one_of_each_color_l1097_109798


namespace parallel_transitive_l1097_109732

-- Define the type for lines in space
structure Line3D where
  -- We don't need to specify the internal structure of a line
  -- as we're only concerned with their relationships

-- Define the parallelism relation
def parallel (l1 l2 : Line3D) : Prop :=
  sorry  -- The actual definition is not important for this statement

-- State the theorem
theorem parallel_transitive (a b c : Line3D) 
  (hab : parallel a b) (hbc : parallel b c) : 
  parallel a c :=
sorry

end parallel_transitive_l1097_109732


namespace max_value_of_objective_function_l1097_109766

-- Define the constraint set
def ConstraintSet (x y : ℝ) : Prop :=
  y ≥ x ∧ x + 3 * y ≤ 4 ∧ x ≥ -2

-- Define the objective function
def ObjectiveFunction (x y : ℝ) : ℝ :=
  |x - 3 * y|

-- Theorem statement
theorem max_value_of_objective_function :
  ∃ (max : ℝ), max = 4 ∧
  ∀ (x y : ℝ), ConstraintSet x y →
  ObjectiveFunction x y ≤ max :=
sorry

end max_value_of_objective_function_l1097_109766


namespace f_properties_l1097_109741

noncomputable def f (x : ℝ) := Real.exp x - Real.exp (-x)

theorem f_properties :
  (∀ x, f (-x) = -f x) ∧
  (∀ x, deriv f x > 0) ∧
  (∀ k, (∀ x, f (x^2) + f (k*x + 1) > 0) ↔ -2 < k ∧ k < 2) :=
sorry

end f_properties_l1097_109741


namespace floor_ceil_sum_l1097_109780

theorem floor_ceil_sum : ⌊(-3.67 : ℝ)⌋ + ⌈(30.95 : ℝ)⌉ = 27 := by
  sorry

end floor_ceil_sum_l1097_109780


namespace quadratic_root_implies_positive_triangle_l1097_109746

theorem quadratic_root_implies_positive_triangle (a b c : ℝ) 
  (h_root : ∃ (α β : ℝ), α > 0 ∧ β ≠ 0 ∧ Complex.I * Complex.I = -1 ∧ 
    (α + Complex.I * β) ^ 2 - (a + b + c) * (α + Complex.I * β) + (a * b + b * c + c * a) = 0) :
  (a > 0 ∧ b > 0 ∧ c > 0) ∧ 
  (Real.sqrt a + Real.sqrt b > Real.sqrt c ∧ 
   Real.sqrt b + Real.sqrt c > Real.sqrt a ∧ 
   Real.sqrt c + Real.sqrt a > Real.sqrt b) := by
  sorry

end quadratic_root_implies_positive_triangle_l1097_109746


namespace function_not_in_third_quadrant_l1097_109711

theorem function_not_in_third_quadrant 
  (a b : ℝ) (ha : 0 < a) (ha' : a < 1) (hb : 0 < b) (hb' : b < 1) :
  ∀ x : ℝ, x < 0 → a^x + b - 1 > 0 := by
  sorry

end function_not_in_third_quadrant_l1097_109711


namespace lcm_from_hcf_and_product_l1097_109794

theorem lcm_from_hcf_and_product (a b : ℕ+) : 
  Nat.gcd a b = 21 → a * b = 138567 → Nat.lcm a b = 6603 := by
  sorry

end lcm_from_hcf_and_product_l1097_109794


namespace binary_conversion_and_subtraction_l1097_109781

/-- Converts a binary number represented as a list of bits to its decimal equivalent -/
def binaryToDecimal (bits : List Bool) : ℕ :=
  bits.enum.foldl (fun acc (i, b) => acc + if b then 2^i else 0) 0

/-- The binary representation of 101101₂ -/
def binaryNumber : List Bool := [true, false, true, true, false, true]

/-- The main theorem to prove -/
theorem binary_conversion_and_subtraction :
  (binaryToDecimal binaryNumber) - 5 = 40 := by sorry

end binary_conversion_and_subtraction_l1097_109781


namespace valid_outfit_choices_l1097_109797

def num_items : ℕ := 4
def num_colors : ℕ := 8

def total_combinations : ℕ := num_colors ^ num_items

def same_color_two_items : ℕ := (num_items.choose 2) * num_colors * (num_colors - 1) * (num_colors - 2)

def same_color_three_items : ℕ := (num_items.choose 3) * num_colors * (num_colors - 1)

def same_color_four_items : ℕ := num_colors

def two_pairs_same_color : ℕ := (num_items.choose 2) * 1 * num_colors * (num_colors - 1)

def invalid_combinations : ℕ := same_color_two_items + same_color_three_items + same_color_four_items + two_pairs_same_color

theorem valid_outfit_choices : 
  total_combinations - invalid_combinations = 1512 :=
sorry

end valid_outfit_choices_l1097_109797


namespace non_multiples_count_is_546_l1097_109760

/-- The count of three-digit numbers that are not multiples of 3 or 11 -/
def non_multiples_count : ℕ :=
  let total_three_digit := 999 - 100 + 1
  let multiples_of_3 := (999 - 100) / 3 + 1
  let multiples_of_11 := (990 - 110) / 11 + 1
  let multiples_of_33 := (990 - 132) / 33 + 1
  total_three_digit - (multiples_of_3 + multiples_of_11 - multiples_of_33)

theorem non_multiples_count_is_546 : non_multiples_count = 546 := by
  sorry

end non_multiples_count_is_546_l1097_109760


namespace square_ratio_sum_l1097_109764

theorem square_ratio_sum (area_ratio : ℚ) (a b c : ℕ) : 
  area_ratio = 300 / 75 →
  (a : ℚ) * Real.sqrt b / c = Real.sqrt area_ratio →
  a + b + c = 4 := by
sorry

end square_ratio_sum_l1097_109764


namespace unique_function_theorem_l1097_109744

open Real

-- Define the function type
def FunctionType := (x : ℝ) → x > 0 → ℝ

-- State the theorem
theorem unique_function_theorem (f : FunctionType) 
  (h1 : f 2009 (by norm_num) = 1)
  (h2 : ∀ (x y : ℝ) (hx : x > 0) (hy : y > 0), 
    f x hx * f y hy + f (2009 / x) (by positivity) * f (2009 / y) (by positivity) = 2 * f (x * y) (by positivity)) :
  ∀ (x : ℝ) (hx : x > 0), f x hx = 1 := by
sorry

end unique_function_theorem_l1097_109744


namespace problem_solution_l1097_109795

theorem problem_solution (y₁ y₂ y₃ y₄ y₅ y₆ y₇ : ℝ) 
  (h₁ : y₁ + 3*y₂ + 5*y₃ + 7*y₄ + 9*y₅ + 11*y₆ + 13*y₇ = 0)
  (h₂ : 3*y₁ + 5*y₂ + 7*y₃ + 9*y₄ + 11*y₅ + 13*y₆ + 15*y₇ = 10)
  (h₃ : 5*y₁ + 7*y₂ + 9*y₃ + 11*y₄ + 13*y₅ + 15*y₆ + 17*y₇ = 104) :
  7*y₁ + 9*y₂ + 11*y₃ + 13*y₄ + 15*y₅ + 17*y₆ + 19*y₇ = 282 := by
sorry

end problem_solution_l1097_109795


namespace new_person_weight_l1097_109782

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

theorem new_person_weight (initial_people : ℕ) (replaced_weight : ℕ) (avg_increase : ℕ) (total_weight : ℕ) :
  initial_people = 4 →
  replaced_weight = 70 →
  avg_increase = 3 →
  total_weight = 390 →
  ∃ (new_weight : ℕ), 
    is_prime new_weight ∧
    new_weight = total_weight - (initial_people * replaced_weight + initial_people * avg_increase) :=
by sorry

end new_person_weight_l1097_109782


namespace solution_product_l1097_109763

theorem solution_product (p q : ℝ) : 
  (p - 4) * (3 * p + 11) = p^2 - 19 * p + 72 →
  (q - 4) * (3 * q + 11) = q^2 - 19 * q + 72 →
  p ≠ q →
  (p + 4) * (q + 4) = -78 :=
by
  sorry

end solution_product_l1097_109763


namespace smallest_m_for_nth_roots_in_T_l1097_109710

def T : Set ℂ := {z | ∃ x y : ℝ, z = x + y * Complex.I ∧ 1/2 ≤ x ∧ x ≤ Real.sqrt 2 / 2}

theorem smallest_m_for_nth_roots_in_T : 
  ∃ m : ℕ+, (∀ n : ℕ+, n ≥ m → ∃ z ∈ T, z^(n:ℕ) = 1) ∧ 
  (∀ k : ℕ+, k < m → ∃ n : ℕ+, n ≥ k ∧ ∀ z ∈ T, z^(n:ℕ) ≠ 1) ∧
  m = 12 :=
sorry

end smallest_m_for_nth_roots_in_T_l1097_109710


namespace movie_ticket_cost_l1097_109703

/-- The cost of movie tickets for a family --/
theorem movie_ticket_cost (C : ℝ) : 
  (∃ (A : ℝ), 
    A = C + 3.25 ∧ 
    2 * A + 4 * C - 2 = 30) → 
  C = 4.25 := by
sorry

end movie_ticket_cost_l1097_109703
