import Mathlib

namespace census_suitable_for_class_spirit_awareness_only_class_spirit_awareness_census_suitable_l3315_331516

/-- Represents a survey scenario --/
inductive SurveyScenario
  | ShellLethalRadius
  | TVViewershipRating
  | YellowRiverFishSpecies
  | ClassSpiritAwareness

/-- Determines if a census method is suitable for a given survey scenario --/
def isCensusSuitable (scenario : SurveyScenario) : Prop :=
  match scenario with
  | SurveyScenario.ClassSpiritAwareness => True
  | _ => False

/-- Theorem: The survey to ascertain the awareness rate of the "Shanxi Spirit" 
    among the students of a certain class is suitable for using a census method --/
theorem census_suitable_for_class_spirit_awareness :
  isCensusSuitable SurveyScenario.ClassSpiritAwareness :=
by sorry

/-- Theorem: The survey to ascertain the awareness rate of the "Shanxi Spirit" 
    among the students of a certain class is the only one suitable for using a census method --/
theorem only_class_spirit_awareness_census_suitable :
  ∀ (scenario : SurveyScenario), 
    isCensusSuitable scenario ↔ scenario = SurveyScenario.ClassSpiritAwareness :=
by sorry

end census_suitable_for_class_spirit_awareness_only_class_spirit_awareness_census_suitable_l3315_331516


namespace work_completion_time_l3315_331514

/-- The time taken to complete a work given the rates of two workers and their working schedule -/
theorem work_completion_time
  (p_completion_time q_completion_time : ℝ)
  (p_solo_time : ℝ)
  (hp : p_completion_time = 20)
  (hq : q_completion_time = 12)
  (hp_solo : p_solo_time = 4)
  : ∃ (total_time : ℝ), total_time = 10 :=
by sorry

end work_completion_time_l3315_331514


namespace sheila_work_hours_l3315_331584

/-- Represents Sheila's work schedule and earnings -/
structure WorkSchedule where
  hourly_rate : ℕ
  weekly_earnings : ℕ
  tue_thu_hours : ℕ
  mon_wed_fri_hours : ℕ

/-- Theorem stating that given Sheila's work conditions, she works 24 hours on Mon, Wed, Fri -/
theorem sheila_work_hours (schedule : WorkSchedule) 
  (h1 : schedule.hourly_rate = 7)
  (h2 : schedule.weekly_earnings = 252)
  (h3 : schedule.tue_thu_hours = 6 * 2)
  (h4 : schedule.weekly_earnings = 
        schedule.hourly_rate * (schedule.tue_thu_hours + schedule.mon_wed_fri_hours)) :
  schedule.mon_wed_fri_hours = 24 := by
  sorry


end sheila_work_hours_l3315_331584


namespace quadratic_complex_roots_l3315_331540

theorem quadratic_complex_roots : ∃ (z₁ z₂ : ℂ),
  z₁ = Complex.mk (Real.sqrt 7 - 1) ((Real.sqrt 7) / 2) ∧
  z₂ = Complex.mk (-(Real.sqrt 7) - 1) (-(Real.sqrt 7) / 2) ∧
  z₁^2 + 2*z₁ = Complex.mk 3 7 ∧
  z₂^2 + 2*z₂ = Complex.mk 3 7 :=
by sorry

end quadratic_complex_roots_l3315_331540


namespace max_payment_is_31_l3315_331587

def is_valid_number (n : ℕ) : Prop :=
  2000 ≤ n ∧ n ≤ 2099

def divisibility_payment (n : ℕ) : ℕ :=
  (if n % 1 = 0 then 1 else 0) +
  (if n % 3 = 0 then 3 else 0) +
  (if n % 5 = 0 then 5 else 0) +
  (if n % 7 = 0 then 7 else 0) +
  (if n % 9 = 0 then 9 else 0) +
  (if n % 11 = 0 then 11 else 0)

theorem max_payment_is_31 :
  ∃ n : ℕ, is_valid_number n ∧
    divisibility_payment n = 31 ∧
    ∀ m : ℕ, is_valid_number m → divisibility_payment m ≤ 31 :=
by sorry

end max_payment_is_31_l3315_331587


namespace calzone_ratio_l3315_331500

def calzone_problem (onion_time garlic_time knead_time rest_time assemble_time total_time : ℕ) : Prop :=
  let pepper_time := garlic_time
  onion_time = 20 ∧
  garlic_time = onion_time / 4 ∧
  knead_time = 30 ∧
  rest_time = 2 * knead_time ∧
  total_time = 124 ∧
  total_time = onion_time + garlic_time + pepper_time + knead_time + rest_time + assemble_time

theorem calzone_ratio (onion_time garlic_time knead_time rest_time assemble_time total_time : ℕ) :
  calzone_problem onion_time garlic_time knead_time rest_time assemble_time total_time →
  (assemble_time : ℚ) / (knead_time + rest_time : ℚ) = 1 / 10 :=
by sorry

end calzone_ratio_l3315_331500


namespace f_neg_l3315_331577

-- Define an odd function f
def f (x : ℝ) : ℝ := sorry

-- Define the properties of f
axiom f_odd : ∀ x, f (-x) = -f x
axiom f_pos : ∀ x > 0, f x = -x * (1 + x)

-- Theorem to prove
theorem f_neg : ∀ x < 0, f x = -x * (1 - x) := by sorry

end f_neg_l3315_331577


namespace perimeter_of_special_isosceles_triangle_l3315_331513

-- Define the real numbers m and n
variable (m n : ℝ)

-- Define the condition |m-2| + √(n-4) = 0
def condition (m n : ℝ) : Prop := abs (m - 2) + Real.sqrt (n - 4) = 0

-- Define an isosceles triangle with sides m and n
structure IsoscelesTriangle :=
  (side1 : ℝ)
  (side2 : ℝ)
  (base : ℝ)
  (is_isosceles : side1 = side2)

-- Define the perimeter of a triangle
def perimeter (t : IsoscelesTriangle) : ℝ := t.side1 + t.side2 + t.base

-- State the theorem
theorem perimeter_of_special_isosceles_triangle :
  ∀ m n : ℝ, condition m n →
  ∃ t : IsoscelesTriangle, (t.side1 = m ∨ t.side1 = n) ∧ (t.base = m ∨ t.base = n) →
  perimeter t = 10 :=
sorry

end perimeter_of_special_isosceles_triangle_l3315_331513


namespace product_of_ab_l3315_331596

theorem product_of_ab (a b : ℝ) (h1 : a - b = 3) (h2 : a^2 + b^2 = 31) : a * b = -11 := by
  sorry

end product_of_ab_l3315_331596


namespace m_equals_2_sufficient_not_necessary_l3315_331518

def M (m : ℝ) : Set ℝ := {-1, m^2}
def N : Set ℝ := {2, 4}

theorem m_equals_2_sufficient_not_necessary :
  ∃ m : ℝ, (M m ∩ N = {4} ∧ m ≠ 2) ∧
  ∀ m : ℝ, m = 2 → M m ∩ N = {4} :=
sorry

end m_equals_2_sufficient_not_necessary_l3315_331518


namespace square_side_length_l3315_331595

theorem square_side_length 
  (x y : ℕ+) 
  (h1 : Nat.gcd x.val y.val = 5)
  (h2 : ∃ (s : ℝ), s > 0 ∧ x.val^2 + y.val^2 = 2 * s^2)
  (h3 : (169 : ℝ) / 6 * Nat.lcm x.val y.val = 2 * s^2) :
  s = 65 * Real.sqrt 2 :=
sorry

end square_side_length_l3315_331595


namespace smallest_k_with_given_remainders_l3315_331551

theorem smallest_k_with_given_remainders : ∃! k : ℕ,
  k > 1 ∧
  k % 13 = 1 ∧
  k % 8 = 1 ∧
  k % 4 = 1 ∧
  ∀ m : ℕ, m > 1 ∧ m % 13 = 1 ∧ m % 8 = 1 ∧ m % 4 = 1 → k ≤ m :=
by
  -- Proof goes here
  sorry

end smallest_k_with_given_remainders_l3315_331551


namespace vectors_form_basis_l3315_331565

theorem vectors_form_basis (a b : ℝ × ℝ) : 
  a = (2, 6) → b = (-1, 3) → LinearIndependent ℝ ![a, b] := by sorry

end vectors_form_basis_l3315_331565


namespace inequality_equivalence_l3315_331557

theorem inequality_equivalence (x : ℝ) : 3 - 1 / (3 * x + 2) < 5 ↔ x < -7/6 ∨ x > -2/3 := by
  sorry

end inequality_equivalence_l3315_331557


namespace inverse_true_implies_negation_true_l3315_331537

theorem inverse_true_implies_negation_true (P : Prop) : 
  (¬P → ¬(¬P)) → (¬P) := by
  sorry

end inverse_true_implies_negation_true_l3315_331537


namespace intersection_M_N_l3315_331578

def M : Set ℕ := {1, 2}

def N : Set ℕ := {x | ∃ a ∈ M, x = 2*a - 1}

theorem intersection_M_N : M ∩ N = {1} := by sorry

end intersection_M_N_l3315_331578


namespace intersection_of_A_and_B_l3315_331580

open Set

-- Define the sets A and B
def A : Set ℝ := {x | 0 < x ∧ x < 2}
def B : Set ℝ := {x | x^2 ≤ 1}

-- State the theorem
theorem intersection_of_A_and_B : A ∩ B = {x | 0 < x ∧ x ≤ 1} := by
  sorry

end intersection_of_A_and_B_l3315_331580


namespace pie_fraction_to_percentage_l3315_331512

theorem pie_fraction_to_percentage : 
  let apple_fraction : ℚ := 1/5
  let cherry_fraction : ℚ := 3/4
  let total_fraction : ℚ := apple_fraction + cherry_fraction
  (total_fraction * 100 : ℚ) = 95 := by sorry

end pie_fraction_to_percentage_l3315_331512


namespace specific_ellipse_area_l3315_331520

/-- An ellipse with given properties -/
structure Ellipse where
  major_axis_endpoint1 : ℝ × ℝ
  major_axis_endpoint2 : ℝ × ℝ
  point_on_ellipse : ℝ × ℝ

/-- The area of an ellipse with the given properties -/
def ellipse_area (e : Ellipse) : ℝ :=
  sorry

/-- The theorem stating that the area of the specific ellipse is 42π -/
theorem specific_ellipse_area :
  let e : Ellipse := {
    major_axis_endpoint1 := (-5, -1),
    major_axis_endpoint2 := (15, -1),
    point_on_ellipse := (12, 2)
  }
  ellipse_area e = 42 * Real.pi := by sorry

end specific_ellipse_area_l3315_331520


namespace factorization_proof_l3315_331572

theorem factorization_proof (x y : ℝ) : 75 * x^10 * y^3 - 150 * x^20 * y^6 = 75 * x^10 * y^3 * (1 - 2 * x^10 * y^3) := by
  sorry

end factorization_proof_l3315_331572


namespace cuboid_surface_area_example_l3315_331532

/-- The surface area of a cuboid given its dimensions -/
def cuboidSurfaceArea (length width height : ℝ) : ℝ :=
  2 * (length * width + width * height + height * length)

/-- Theorem: The surface area of a cuboid with length 4, width 5, and height 6 is 148 -/
theorem cuboid_surface_area_example : cuboidSurfaceArea 4 5 6 = 148 := by
  sorry

end cuboid_surface_area_example_l3315_331532


namespace charlie_won_two_games_l3315_331525

/-- Represents a player in the tournament -/
inductive Player : Type
| Alice : Player
| Bob : Player
| Charlie : Player

/-- The number of games won by a player -/
def games_won (p : Player) : ℕ :=
  match p with
  | Player.Alice => 2
  | Player.Bob => 1
  | Player.Charlie => sorry  -- To be proven

/-- The number of games lost by a player -/
def games_lost (p : Player) : ℕ :=
  match p with
  | Player.Alice => 1
  | Player.Bob => 2
  | Player.Charlie => 2

/-- The total number of games played in the tournament -/
def total_games : ℕ := 3

theorem charlie_won_two_games :
  games_won Player.Charlie = 2 := by sorry

end charlie_won_two_games_l3315_331525


namespace percentage_ratio_l3315_331563

theorem percentage_ratio (x : ℝ) (a b : ℝ) (ha : a = 0.08 * x) (hb : b = 0.16 * x) :
  a / b = 0.5 := by sorry

end percentage_ratio_l3315_331563


namespace total_pamphlets_is_10700_l3315_331592

-- Define the printing rates and durations
def mike_initial_rate : ℕ := 600
def mike_initial_duration : ℕ := 9
def mike_final_duration : ℕ := 2

def leo_initial_rate : ℕ := 2 * mike_initial_rate
def leo_initial_duration : ℕ := mike_initial_duration / 3

def sally_initial_rate : ℕ := 3 * mike_initial_rate
def sally_initial_duration : ℕ := leo_initial_duration / 2
def sally_final_duration : ℕ := 1

-- Define the function to calculate total pamphlets
def calculate_total_pamphlets : ℕ :=
  -- Mike's pamphlets
  let mike_pamphlets := mike_initial_rate * mike_initial_duration + 
                        (mike_initial_rate / 3) * mike_final_duration

  -- Leo's pamphlets
  let leo_pamphlets := leo_initial_rate * 1 + 
                       (leo_initial_rate / 2) * 1 + 
                       (leo_initial_rate / 4) * 1

  -- Sally's pamphlets
  let sally_pamphlets := sally_initial_rate * sally_initial_duration + 
                         (leo_initial_rate / 2) * sally_final_duration

  mike_pamphlets + leo_pamphlets + sally_pamphlets

-- Theorem statement
theorem total_pamphlets_is_10700 :
  calculate_total_pamphlets = 10700 := by
  sorry

end total_pamphlets_is_10700_l3315_331592


namespace roses_equation_initial_roses_count_l3315_331552

/-- The number of roses initially in the vase -/
def initial_roses : ℕ := sorry

/-- The number of roses Jessica added to the vase -/
def added_roses : ℕ := 16

/-- The final number of roses in the vase -/
def final_roses : ℕ := 23

/-- Theorem stating that the initial number of roses plus the added roses equals the final number of roses -/
theorem roses_equation : initial_roses + added_roses = final_roses := by sorry

/-- Theorem proving that the initial number of roses is 7 -/
theorem initial_roses_count : initial_roses = 7 := by sorry

end roses_equation_initial_roses_count_l3315_331552


namespace solve_equation_l3315_331543

theorem solve_equation (x : ℝ) (h : x - 3*x + 4*x = 140) : x = 70 := by
  sorry

end solve_equation_l3315_331543


namespace sin_pi_fourth_plus_alpha_l3315_331585

theorem sin_pi_fourth_plus_alpha (α : Real) (h1 : α ∈ Set.Ioo 0 π) 
  (h2 : Real.tan (α - π/4) = 1/3) : Real.sin (π/4 + α) = 3 * Real.sqrt 10 / 10 := by
  sorry

end sin_pi_fourth_plus_alpha_l3315_331585


namespace vertical_translation_of_linear_function_l3315_331531

/-- Represents a linear function of the form y = mx + b -/
structure LinearFunction where
  m : ℝ
  b : ℝ

/-- Translates a linear function vertically by a given amount -/
def translate_vertical (f : LinearFunction) (k : ℝ) : LinearFunction :=
  { m := f.m, b := f.b + k }

/-- The original function y = -3x -/
def original_function : LinearFunction :=
  { m := -3, b := 0 }

/-- The amount of vertical translation -/
def translation_amount : ℝ := 2

theorem vertical_translation_of_linear_function :
  translate_vertical original_function translation_amount =
  { m := -3, b := 2 } :=
sorry

end vertical_translation_of_linear_function_l3315_331531


namespace parabola_coefficient_sum_l3315_331589

/-- A parabola passing through (2, 3) and (0, 7) has coefficients a, b, c such that a + b + c = 4 -/
theorem parabola_coefficient_sum (a b c : ℝ) : 
  (∀ x, a * x^2 + b * x + c = a * (x - 2)^2 + 3) → -- Vertex form condition
  (a * 0^2 + b * 0 + c = 7) →                      -- Passes through (0, 7)
  (a + b + c = 4) := by
sorry

end parabola_coefficient_sum_l3315_331589


namespace camping_products_costs_l3315_331517

/-- The wholesale cost of a sleeping bag -/
def sleeping_bag_cost : ℚ := 560 / 23

/-- The wholesale cost of a tent -/
def tent_cost : ℚ := 200 / 3

/-- The selling price of a sleeping bag -/
def sleeping_bag_price : ℚ := 28

/-- The selling price of a tent -/
def tent_price : ℚ := 80

/-- The gross profit percentage for sleeping bags -/
def sleeping_bag_profit_percent : ℚ := 15 / 100

/-- The gross profit percentage for tents -/
def tent_profit_percent : ℚ := 20 / 100

theorem camping_products_costs :
  (sleeping_bag_cost * (1 + sleeping_bag_profit_percent) = sleeping_bag_price) ∧
  (tent_cost * (1 + tent_profit_percent) = tent_price) := by
  sorry

end camping_products_costs_l3315_331517


namespace max_value_of_t_l3315_331576

open Real

noncomputable def f (x : ℝ) : ℝ := log x

noncomputable def t (m : ℝ) : ℝ := (2 * m + log m / m - m * log m) / 2

theorem max_value_of_t :
  ∃ (m : ℝ), m > 1 ∧ ∀ (x : ℝ), x > 1 → t x ≤ t m ∧ t m = (exp 2 + 1) / (2 * exp 1) := by
  sorry

end max_value_of_t_l3315_331576


namespace product_of_difference_and_sum_of_squares_l3315_331547

theorem product_of_difference_and_sum_of_squares (a b : ℝ) 
  (h1 : a - b = 5) 
  (h2 : a^2 + b^2 = 29) : 
  a * b = 2 := by
sorry

end product_of_difference_and_sum_of_squares_l3315_331547


namespace sarah_flour_total_l3315_331590

/-- The total amount of flour Sarah has -/
def total_flour (rye whole_wheat chickpea pastry : ℕ) : ℕ :=
  rye + whole_wheat + chickpea + pastry

/-- Theorem: Sarah has 20 pounds of flour in total -/
theorem sarah_flour_total :
  total_flour 5 10 3 2 = 20 := by
  sorry

end sarah_flour_total_l3315_331590


namespace incorrect_statement_about_parallelogram_l3315_331582

-- Define a parallelogram
structure Parallelogram :=
  (diagonals_bisect : Bool)
  (diagonals_perpendicular : Bool)

-- Define the properties of a parallelogram
def parallelogram_properties : Parallelogram :=
  { diagonals_bisect := true,
    diagonals_perpendicular := false }

-- Theorem to prove
theorem incorrect_statement_about_parallelogram :
  ¬(parallelogram_properties.diagonals_bisect ∧ parallelogram_properties.diagonals_perpendicular) :=
by sorry

end incorrect_statement_about_parallelogram_l3315_331582


namespace transform_sine_to_cosine_l3315_331567

/-- Given a function f(x) = √3 * sin(2x), prove that translating it right by π/4 
    and then compressing its x-coordinates by half results in g(x) = -√3 * cos(4x) -/
theorem transform_sine_to_cosine (x : ℝ) : 
  let f : ℝ → ℝ := λ x => Real.sqrt 3 * Real.sin (2 * x)
  let g : ℝ → ℝ := λ x => -Real.sqrt 3 * Real.cos (4 * x)
  let h : ℝ → ℝ := λ x => f (x / 2 + π / 4)
  h x = g x := by
  sorry

end transform_sine_to_cosine_l3315_331567


namespace triangle_base_length_l3315_331524

theorem triangle_base_length 
  (height : ℝ) 
  (area : ℝ) 
  (h1 : height = 8) 
  (h2 : area = 24) 
  (h3 : area = (1/2) * height * base) : 
  base = 6 := by
  sorry

end triangle_base_length_l3315_331524


namespace count_integers_with_fourth_power_between_negative_hundred_and_hundred_l3315_331504

theorem count_integers_with_fourth_power_between_negative_hundred_and_hundred :
  (∃ (S : Finset Int), (∀ x : Int, x ∈ S ↔ -100 < x^4 ∧ x^4 < 100) ∧ Finset.card S = 7) := by
  sorry

end count_integers_with_fourth_power_between_negative_hundred_and_hundred_l3315_331504


namespace barn_paint_area_l3315_331535

/-- Represents the dimensions of a rectangular barn -/
structure BarnDimensions where
  width : ℝ
  length : ℝ
  height : ℝ

/-- Calculates the total area to be painted for a barn with given dimensions -/
def totalPaintArea (d : BarnDimensions) : ℝ :=
  let wallArea1 := 2 * d.width * d.height
  let wallArea2 := 2 * d.length * d.height
  let ceilingArea := d.width * d.length
  let roofArea := d.width * d.length
  2 * wallArea1 + 2 * wallArea2 + ceilingArea + roofArea

/-- The theorem stating that the total paint area for the given barn dimensions is 1116 sq yd -/
theorem barn_paint_area :
  let barn := BarnDimensions.mk 12 15 7
  totalPaintArea barn = 1116 := by sorry

end barn_paint_area_l3315_331535


namespace mult_B_not_binomial_square_or_diff_squares_other_mults_are_diff_squares_l3315_331562

-- Define the multiplications
def mult_A (x y : ℝ) := (3*x + 7*y) * (3*x - 7*y)
def mult_B (m n : ℝ) := (5*m - n) * (n - 5*m)
def mult_C (x : ℝ) := (-0.2*x - 0.3) * (-0.2*x + 0.3)
def mult_D (m n : ℝ) := (-3*n - m*n) * (3*n - m*n)

-- Define the square of binomial form
def square_of_binomial (a b : ℝ) := (a + b)^2

-- Define the difference of squares form
def diff_of_squares (a b : ℝ) := a^2 - b^2

theorem mult_B_not_binomial_square_or_diff_squares :
  ∀ m n : ℝ, ¬∃ a b : ℝ, mult_B m n = square_of_binomial a b ∨ mult_B m n = diff_of_squares a b :=
sorry

theorem other_mults_are_diff_squares :
  (∀ x y : ℝ, ∃ a b : ℝ, mult_A x y = diff_of_squares a b) ∧
  (∀ x : ℝ, ∃ a b : ℝ, mult_C x = diff_of_squares a b) ∧
  (∀ m n : ℝ, ∃ a b : ℝ, mult_D m n = diff_of_squares a b) :=
sorry

end mult_B_not_binomial_square_or_diff_squares_other_mults_are_diff_squares_l3315_331562


namespace square_side_length_l3315_331560

theorem square_side_length (s : ℝ) (h : s > 0) : s ^ 2 = 2 * (4 * s) → s = 8 := by
  sorry

end square_side_length_l3315_331560


namespace coin_stack_count_l3315_331529

/-- Thickness of a 2p coin in millimeters -/
def thickness_2p : ℚ := 205/100

/-- Thickness of a 10p coin in millimeters -/
def thickness_10p : ℚ := 195/100

/-- Total height of the stack in millimeters -/
def stack_height : ℚ := 19

/-- The number of coins in the stack -/
def total_coins : ℕ := 10

/-- Theorem stating that the total number of coins in a stack of 19 mm height,
    consisting only of 2p and 10p coins, is 10 -/
theorem coin_stack_count :
  ∃ (x y : ℕ), x + y = total_coins ∧
  x * thickness_2p + y * thickness_10p = stack_height :=
sorry

end coin_stack_count_l3315_331529


namespace imaginary_part_of_complex_fraction_l3315_331542

theorem imaginary_part_of_complex_fraction : 
  Complex.im ((1 - Complex.I) / (1 + Complex.I)) = -1 := by
  sorry

end imaginary_part_of_complex_fraction_l3315_331542


namespace sin_4theta_l3315_331545

theorem sin_4theta (θ : ℝ) (h : Complex.exp (θ * Complex.I) = (4 + Complex.I * Real.sqrt 3) / 5) : 
  Real.sin (4 * θ) = 208 * Real.sqrt 3 / 625 := by
  sorry

end sin_4theta_l3315_331545


namespace count_valid_numbers_l3315_331536

def is_valid_number (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000 ∧ 
  (n % 1000 = n / 6)

theorem count_valid_numbers : 
  ∃ (s : Finset ℕ), (∀ n ∈ s, is_valid_number n) ∧ s.card = 4 :=
sorry

end count_valid_numbers_l3315_331536


namespace goldfish_equal_at_11_months_l3315_331546

/-- The number of months it takes for Brent and Gretel to have the same number of goldfish -/
def months_until_equal : ℕ := 11

/-- Brent's initial number of goldfish -/
def brent_initial : ℕ := 6

/-- Gretel's initial number of goldfish -/
def gretel_initial : ℕ := 150

/-- Brent's goldfish growth rate per month -/
def brent_growth_rate : ℝ := 2

/-- Gretel's goldfish growth rate per month -/
def gretel_growth_rate : ℝ := 1.5

/-- Brent's number of goldfish after n months -/
def brent_goldfish (n : ℕ) : ℝ := brent_initial * brent_growth_rate ^ n

/-- Gretel's number of goldfish after n months -/
def gretel_goldfish (n : ℕ) : ℝ := gretel_initial * gretel_growth_rate ^ n

theorem goldfish_equal_at_11_months :
  brent_goldfish months_until_equal = gretel_goldfish months_until_equal :=
sorry

end goldfish_equal_at_11_months_l3315_331546


namespace max_n_with_divisor_condition_l3315_331501

theorem max_n_with_divisor_condition (N : ℕ) : 
  (∃ d₁ d₂ d₃ : ℕ, 
    d₁ ∣ N ∧ d₂ ∣ N ∧ d₃ ∣ N ∧
    d₁ < d₂ ∧ 
    (∀ d : ℕ, d ∣ N → d ≤ d₁ ∨ d ≥ d₂) ∧
    (∀ d : ℕ, d ∣ N → d ≤ d₃ ∨ d > N / d₃) ∧
    d₃ = 21 * d₂) →
  N ≤ 441 := by
sorry

end max_n_with_divisor_condition_l3315_331501


namespace count_integer_solutions_l3315_331538

theorem count_integer_solutions : ∃! A : ℕ, 
  A = (Finset.filter (fun p : ℕ × ℕ => 
    p.1 + p.2 ≥ A ∧ 
    p.1 ≤ 6 ∧ 
    p.2 ≤ 7
  ) (Finset.product (Finset.range 7) (Finset.range 8))).card ∧
  A = 10 := by
  sorry

end count_integer_solutions_l3315_331538


namespace number_of_men_l3315_331591

theorem number_of_men (max_handshakes : ℕ) : max_handshakes = 153 → ∃ n : ℕ, n = 18 ∧ max_handshakes = n * (n - 1) / 2 := by
  sorry

end number_of_men_l3315_331591


namespace committee_formation_l3315_331534

theorem committee_formation (n : ℕ) (k : ℕ) (h : n = 8 ∧ k = 4) : 
  (Nat.choose (n - 1) (k - 1)) * (Nat.choose (n - 1) k) = 1225 := by
  sorry

end committee_formation_l3315_331534


namespace multiply_and_simplify_l3315_331539

theorem multiply_and_simplify (x y : ℝ) :
  (3 * x^2 - 4 * y^3) * (9 * x^4 + 12 * x^2 * y^3 + 16 * y^6) = 27 * x^6 - 64 * y^9 := by
  sorry

end multiply_and_simplify_l3315_331539


namespace sin_negative_1290_degrees_l3315_331521

theorem sin_negative_1290_degrees (θ : ℝ) :
  (∀ k : ℤ, Real.sin (θ + k * (2 * π)) = Real.sin θ) →
  (∀ θ : ℝ, Real.sin (π - θ) = Real.sin θ) →
  Real.sin (π / 6) = 1 / 2 →
  Real.sin (-1290 * π / 180) = 1 / 2 := by
  sorry

end sin_negative_1290_degrees_l3315_331521


namespace intersection_implies_sum_l3315_331553

-- Define the functions
def f (a b x : ℝ) : ℝ := -2 * abs (x - a) + b
def g (c d x : ℝ) : ℝ := 2 * abs (x - c) + d

-- State the theorem
theorem intersection_implies_sum (a b c d : ℝ) : 
  (f a b 1 = 4 ∧ f a b 7 = 0 ∧ g c d 1 = 4 ∧ g c d 7 = 0) → a + c = 10 := by
  sorry

end intersection_implies_sum_l3315_331553


namespace integral_value_l3315_331503

theorem integral_value : ∫ (x : ℝ) in (0)..(1), (Real.sqrt (1 - (x - 1)^2) - x^2) = π/4 - 1/3 := by
  sorry

end integral_value_l3315_331503


namespace square_perimeter_relation_l3315_331593

theorem square_perimeter_relation (perimeter_A : ℝ) (area_ratio : ℝ) : 
  perimeter_A = 36 →
  area_ratio = 1/3 →
  let side_A := perimeter_A / 4
  let area_A := side_A ^ 2
  let area_B := area_ratio * area_A
  let side_B := Real.sqrt area_B
  let perimeter_B := 4 * side_B
  perimeter_B = 12 * Real.sqrt 3 := by
sorry

end square_perimeter_relation_l3315_331593


namespace range_of_a_l3315_331502

/-- Proposition p: For all real x, x²-2x > a -/
def p (a : ℝ) : Prop := ∀ x : ℝ, x^2 - 2*x > a

/-- Proposition q: There exists a real x₀ such that x₀²+2ax₀+2-a=0 -/
def q (a : ℝ) : Prop := ∃ x₀ : ℝ, x₀^2 + 2*a*x₀ + 2 - a = 0

/-- The range of a given the conditions on p and q -/
theorem range_of_a : ∀ a : ℝ, (p a ∨ q a) ∧ ¬(p a ∧ q a) → a ∈ Set.Ioo (-2) (-1) ∪ Set.Ici 1 :=
sorry

end range_of_a_l3315_331502


namespace method1_saves_more_l3315_331569

/-- Represents the price of a badminton racket in yuan -/
def racket_price : ℕ := 20

/-- Represents the price of a shuttlecock in yuan -/
def shuttlecock_price : ℕ := 5

/-- Represents the number of rackets to be purchased -/
def num_rackets : ℕ := 4

/-- Represents the number of shuttlecocks to be purchased -/
def num_shuttlecocks : ℕ := 30

/-- Calculates the cost using discount method ① -/
def cost_method1 : ℕ := racket_price * num_rackets + shuttlecock_price * (num_shuttlecocks - num_rackets)

/-- Calculates the cost using discount method ② -/
def cost_method2 : ℚ := (racket_price * num_rackets + shuttlecock_price * num_shuttlecocks) * 92 / 100

/-- Theorem stating that discount method ① saves more money than method ② -/
theorem method1_saves_more : cost_method1 < cost_method2 := by
  sorry


end method1_saves_more_l3315_331569


namespace min_value_fraction_l3315_331522

theorem min_value_fraction (x : ℝ) (h : x > 9) :
  x^2 / (x - 9) ≥ 36 ∧ ∃ y > 9, y^2 / (y - 9) = 36 := by
  sorry

end min_value_fraction_l3315_331522


namespace sqrt_difference_comparison_l3315_331571

theorem sqrt_difference_comparison (m : ℝ) (h : m > 1) :
  Real.sqrt (m + 1) - Real.sqrt m < Real.sqrt m - Real.sqrt (m - 1) := by
  sorry

end sqrt_difference_comparison_l3315_331571


namespace sequence_sum_l3315_331550

theorem sequence_sum (a : ℕ → ℕ) (h1 : a 1 = 1) (h2 : ∀ n > 1, a n - a (n-1) = n) :
  ∀ n : ℕ, n ≥ 1 → a n = n * (n + 1) / 2 :=
sorry

end sequence_sum_l3315_331550


namespace blithe_toy_count_l3315_331558

/-- The number of toys Blithe has after losing some and finding some -/
def finalToyCount (initial lost found : ℕ) : ℕ :=
  initial - lost + found

/-- Theorem: Given Blithe's initial toy count, the number of toys lost, and the number of toys found,
    the final toy count is equal to the initial count minus the lost toys plus the found toys -/
theorem blithe_toy_count : finalToyCount 40 6 9 = 43 := by
  sorry

end blithe_toy_count_l3315_331558


namespace unique_solution_l3315_331566

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x < 1 then 2^(-x) else Real.log x / Real.log 81

-- State the theorem
theorem unique_solution :
  ∃! x, f x = 1/4 ∧ x = 3 := by sorry

end unique_solution_l3315_331566


namespace solve_equation_l3315_331544

theorem solve_equation (x : ℤ) (h : 9873 + x = 13200) : x = 3327 := by
  sorry

end solve_equation_l3315_331544


namespace tangent_line_equation_l3315_331583

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 + Real.log x

theorem tangent_line_equation (a : ℝ) :
  (∀ ε > 0, ∃ δ > 0, ∀ x, 0 < |x - 1| ∧ |x - 1| < δ → 
    |((f a x - f a 1) / (x - 1)) - 3| < ε) →
  ∃ b c : ℝ, ∀ x y : ℝ, y = f a x → (x = 1 ∧ y = f a 1) → 
    3 * x - y - 2 = 0 := by sorry

end tangent_line_equation_l3315_331583


namespace quadratic_roots_expression_l3315_331511

theorem quadratic_roots_expression (m n : ℝ) : 
  m ^ 2 + 2015 * m - 1 = 0 ∧ n ^ 2 + 2015 * n - 1 = 0 → m ^ 2 * n + m * n ^ 2 - m * n = 2016 := by
  sorry

end quadratic_roots_expression_l3315_331511


namespace valid_subcommittee_count_l3315_331505

def total_members : ℕ := 12
def teacher_count : ℕ := 6
def subcommittee_size : ℕ := 5
def min_teachers : ℕ := 2

def subcommittee_count : ℕ := 696

theorem valid_subcommittee_count :
  (total_members.choose subcommittee_size) -
  ((teacher_count.choose 0) * ((total_members - teacher_count).choose subcommittee_size) +
   (teacher_count.choose 1) * ((total_members - teacher_count).choose (subcommittee_size - 1)))
  = subcommittee_count :=
by sorry

end valid_subcommittee_count_l3315_331505


namespace range_of_q_l3315_331549

def q (x : ℝ) : ℝ := x^4 + 4*x^2 + 4

theorem range_of_q :
  {y : ℝ | ∃ x ≥ 0, q x = y} = {y : ℝ | y ≥ 4} := by sorry

end range_of_q_l3315_331549


namespace square_difference_forty_thirtynine_l3315_331568

theorem square_difference_forty_thirtynine : (40 : ℕ)^2 - (39 : ℕ)^2 = 79 := by
  sorry

end square_difference_forty_thirtynine_l3315_331568


namespace min_value_abs_sum_min_value_achievable_l3315_331574

theorem min_value_abs_sum (x : ℝ) : |x - 1| + |x - 4| ≥ 3 := by sorry

theorem min_value_achievable : ∃ x : ℝ, |x - 1| + |x - 4| = 3 := by sorry

end min_value_abs_sum_min_value_achievable_l3315_331574


namespace line_properties_l3315_331570

-- Define the line l
def line_l (x y : ℝ) : Prop := Real.sqrt 3 * x + y + 3 = 0

-- Define the point (0, -3)
def point : ℝ × ℝ := (0, -3)

-- Define the other line
def other_line (x y : ℝ) : Prop := x + (Real.sqrt 3 / 3) * y + Real.sqrt 3 = 0

theorem line_properties :
  (∀ x y, line_l x y ↔ other_line x y) ∧
  line_l point.1 point.2 ∧
  (∀ x y, line_l x y → y / x ≠ Real.tan (60 * π / 180)) ∧
  (∃ x, line_l x 0 ∧ x = -Real.sqrt 3) :=
sorry

end line_properties_l3315_331570


namespace ivy_cupcakes_l3315_331581

def morning_cupcakes : ℕ := 20
def afternoon_difference : ℕ := 15

def total_cupcakes : ℕ := morning_cupcakes + (morning_cupcakes + afternoon_difference)

theorem ivy_cupcakes : total_cupcakes = 55 := by
  sorry

end ivy_cupcakes_l3315_331581


namespace correct_matching_probability_l3315_331523

/-- The number of celebrities, recent photos, and baby photos -/
def n : ℕ := 4

/-- The probability of correctly matching all celebrities to their recent photos -/
def prob_recent : ℚ := 1 / (n.factorial : ℚ)

/-- The probability of correctly matching all recent photos to baby photos -/
def prob_baby : ℚ := 1 / (n.factorial : ℚ)

/-- The overall probability of correctly matching all celebrities to their baby photos through recent photos -/
def prob_total : ℚ := prob_recent * prob_baby

theorem correct_matching_probability :
  prob_total = 1 / 576 := by sorry

end correct_matching_probability_l3315_331523


namespace scaling_transformation_cosine_curve_l3315_331533

/-- The scaling transformation applied to the curve y = cos 6x results in y' = 2cos 2x' -/
theorem scaling_transformation_cosine_curve :
  ∀ (x y x' y' : ℝ),
  y = Real.cos (6 * x) →
  x' = 3 * x →
  y' = 2 * y →
  y' = 2 * Real.cos (2 * x') := by
sorry

end scaling_transformation_cosine_curve_l3315_331533


namespace smallest_lcm_three_digit_gcd_five_l3315_331506

theorem smallest_lcm_three_digit_gcd_five :
  ∃ (m n : ℕ), 
    100 ≤ m ∧ m < 1000 ∧
    100 ≤ n ∧ n < 1000 ∧
    Nat.gcd m n = 5 ∧
    Nat.lcm m n = 2100 ∧
    ∀ (p q : ℕ), 
      100 ≤ p ∧ p < 1000 ∧
      100 ≤ q ∧ q < 1000 ∧
      Nat.gcd p q = 5 →
      Nat.lcm p q ≥ 2100 :=
by sorry

end smallest_lcm_three_digit_gcd_five_l3315_331506


namespace angie_salary_is_80_l3315_331594

/-- Represents Angie's monthly finances -/
structure MonthlyFinances where
  necessities : ℕ
  taxes : ℕ
  leftover : ℕ

/-- Calculates the monthly salary based on expenses and leftover amount -/
def calculate_salary (finances : MonthlyFinances) : ℕ :=
  finances.necessities + finances.taxes + finances.leftover

/-- Theorem stating that Angie's monthly salary is $80 -/
theorem angie_salary_is_80 (angie : MonthlyFinances) 
  (h1 : angie.necessities = 42)
  (h2 : angie.taxes = 20)
  (h3 : angie.leftover = 18) :
  calculate_salary angie = 80 := by
  sorry

#eval calculate_salary { necessities := 42, taxes := 20, leftover := 18 }

end angie_salary_is_80_l3315_331594


namespace cylinder_surface_area_l3315_331564

/-- The surface area of a cylinder with height 4 and base circumference 2π is 10π. -/
theorem cylinder_surface_area :
  ∀ (h r : ℝ), h = 4 ∧ 2 * π * r = 2 * π →
  2 * π * r * h + 2 * π * r^2 = 10 * π :=
by sorry

end cylinder_surface_area_l3315_331564


namespace keaton_yearly_earnings_l3315_331556

/-- Represents Keaton's farm earnings -/
def farm_earnings (orange_harvest_interval : ℕ) (orange_price : ℕ) (apple_harvest_interval : ℕ) (apple_price : ℕ) : ℕ :=
  let months_in_year := 12
  let orange_harvests := months_in_year / orange_harvest_interval
  let apple_harvests := months_in_year / apple_harvest_interval
  orange_harvests * orange_price + apple_harvests * apple_price

/-- Keaton's yearly earnings from his farm of oranges and apples -/
theorem keaton_yearly_earnings :
  farm_earnings 2 50 3 30 = 420 :=
by sorry

end keaton_yearly_earnings_l3315_331556


namespace fraction_sum_is_one_equation_no_solution_l3315_331548

-- Problem 1
theorem fraction_sum_is_one (a b : ℝ) (h : a ≠ b) :
  a / (a - b) + b / (b - a) = 1 := by sorry

-- Problem 2
theorem equation_no_solution :
  ¬∃ x : ℝ, (1 / (x - 2) = (1 - x) / (2 - x) - 3) := by sorry

end fraction_sum_is_one_equation_no_solution_l3315_331548


namespace A_intersect_B_l3315_331515

def A : Set ℝ := {-1, 0, 1, 2}
def B : Set ℝ := {x | 0 < 2 - x ∧ 2 - x < 3}

theorem A_intersect_B : A ∩ B = {0, 1} := by
  sorry

end A_intersect_B_l3315_331515


namespace isosceles_triangle_area_l3315_331599

/-- An isosceles triangle with given properties has an area of 54 square centimeters -/
theorem isosceles_triangle_area (a b : ℝ) (h_isosceles : a = b) (h_perimeter : 2 * a + b = 36)
  (h_base_angles : 2 * Real.arccos ((a^2 - b^2/4) / a^2) = 130 * π / 180)
  (h_inradius : (a * b) / (a + b + (a^2 - b^2/4).sqrt) = 3) : 
  a * b * Real.sin (Real.arccos ((a^2 - b^2/4) / a^2)) / 2 = 54 := by
  sorry

end isosceles_triangle_area_l3315_331599


namespace ball_distribution_theorem_l3315_331555

/-- Represents the number of ways to distribute balls into boxes -/
def distribute_balls (total_balls : ℕ) (num_boxes : ℕ) (min_balls : ℕ → ℕ) : ℕ :=
  sorry

/-- The main theorem stating that there are 15 ways to distribute 10 balls into 3 boxes -/
theorem ball_distribution_theorem :
  distribute_balls 10 3 (fun i => i) = 15 := by sorry

end ball_distribution_theorem_l3315_331555


namespace nearly_regular_polyhedra_theorem_l3315_331586

/-- A structure representing a polyhedron -/
structure Polyhedron where
  faces : ℕ
  edges : ℕ
  vertices : ℕ

/-- Definition of a nearly regular polyhedron -/
def NearlyRegularPolyhedron (p : Polyhedron) : Prop := sorry

/-- Intersection of two polyhedra -/
def intersect (p1 p2 : Polyhedron) : Polyhedron := sorry

/-- Tetrahedron -/
def Tetrahedron : Polyhedron := ⟨4, 6, 4⟩

/-- Octahedron -/
def Octahedron : Polyhedron := ⟨8, 12, 6⟩

/-- Cube -/
def Cube : Polyhedron := ⟨6, 12, 8⟩

/-- Dodecahedron -/
def Dodecahedron : Polyhedron := ⟨12, 30, 20⟩

/-- Icosahedron -/
def Icosahedron : Polyhedron := ⟨20, 30, 12⟩

/-- The set of nearly regular polyhedra -/
def NearlyRegularPolyhedra : Set Polyhedron := sorry

theorem nearly_regular_polyhedra_theorem :
  ∃ (p1 p2 p3 p4 p5 : Polyhedron),
    p1 ∈ NearlyRegularPolyhedra ∧
    p2 ∈ NearlyRegularPolyhedra ∧
    p3 ∈ NearlyRegularPolyhedra ∧
    p4 ∈ NearlyRegularPolyhedra ∧
    p5 ∈ NearlyRegularPolyhedra ∧
    p1 = intersect Tetrahedron Octahedron ∧
    p2 = intersect Cube Octahedron ∧
    p3 = intersect Dodecahedron Icosahedron ∧
    NearlyRegularPolyhedron p4 ∧
    NearlyRegularPolyhedron p5 :=
  sorry

end nearly_regular_polyhedra_theorem_l3315_331586


namespace solve_equation_l3315_331527

theorem solve_equation : ∃ y : ℝ, (60 / 100 = Real.sqrt ((y + 20) / 100)) ∧ y = 16 := by
  sorry

end solve_equation_l3315_331527


namespace continued_fraction_sum_l3315_331510

theorem continued_fraction_sum (x y z : ℕ+) : 
  (151 : ℚ) / 44 = 3 + 1 / (x.val + 1 / (y.val + 1 / z.val)) → 
  x.val + y.val + z.val = 11 := by
sorry

end continued_fraction_sum_l3315_331510


namespace georgia_muffins_per_batch_l3315_331526

/-- Calculates the number of muffins per batch given the total number of students,
    total batches made, and the number of months. -/
def muffins_per_batch (students : ℕ) (total_batches : ℕ) (months : ℕ) : ℕ :=
  students * months / total_batches

/-- Proves that given 24 students and 36 batches of muffins made in 9 months,
    the number of muffins per batch is 6. -/
theorem georgia_muffins_per_batch :
  muffins_per_batch 24 36 9 = 6 := by
  sorry

end georgia_muffins_per_batch_l3315_331526


namespace absolute_value_calculation_l3315_331573

theorem absolute_value_calculation : |-2| - Real.sqrt 4 + 3^2 = 9 := by
  sorry

end absolute_value_calculation_l3315_331573


namespace exactly_one_hit_probability_l3315_331561

/-- The probability that both A and B hit the target -/
def prob_both_hit : ℝ := 0.6

/-- The probability that A hits the target -/
def prob_A_hit : ℝ := prob_both_hit

/-- The probability that B hits the target -/
def prob_B_hit : ℝ := prob_both_hit

/-- The probability that exactly one of A and B hits the target -/
def prob_exactly_one_hit : ℝ := prob_A_hit * (1 - prob_B_hit) + (1 - prob_A_hit) * prob_B_hit

theorem exactly_one_hit_probability :
  prob_exactly_one_hit = 0.48 :=
by sorry

end exactly_one_hit_probability_l3315_331561


namespace initial_machines_count_l3315_331598

/-- The number of machines in the initial group -/
def initial_machines : ℕ := 15

/-- The number of bags produced per minute by the initial group -/
def initial_production_rate : ℕ := 45

/-- The number of machines in the larger group -/
def larger_group_machines : ℕ := 150

/-- The number of bags produced by the larger group -/
def larger_group_production : ℕ := 3600

/-- The time taken by the larger group to produce the bags (in minutes) -/
def production_time : ℕ := 8

theorem initial_machines_count :
  initial_machines = 15 ∧
  initial_production_rate = 45 ∧
  larger_group_machines = 150 ∧
  larger_group_production = 3600 ∧
  production_time = 8 →
  initial_machines * larger_group_production = initial_production_rate * larger_group_machines * production_time :=
by sorry

end initial_machines_count_l3315_331598


namespace dans_remaining_potatoes_l3315_331579

/-- Given an initial number of potatoes and a number of eaten potatoes,
    calculate the remaining number of potatoes. -/
def remaining_potatoes (initial : ℕ) (eaten : ℕ) : ℕ :=
  initial - eaten

/-- Theorem stating that Dan's remaining potatoes is 3 given the initial conditions. -/
theorem dans_remaining_potatoes :
  remaining_potatoes 7 4 = 3 := by sorry

end dans_remaining_potatoes_l3315_331579


namespace area_ratio_square_to_rectangle_l3315_331554

/-- The ratio of the area of a square with side length 48 cm to the area of a rectangle with dimensions 56 cm by 63 cm is 2/3. -/
theorem area_ratio_square_to_rectangle : 
  let square_side : ℝ := 48
  let rect_width : ℝ := 56
  let rect_height : ℝ := 63
  let square_area := square_side ^ 2
  let rect_area := rect_width * rect_height
  square_area / rect_area = 2 / 3 := by sorry

end area_ratio_square_to_rectangle_l3315_331554


namespace find_C_value_l3315_331530

-- Define the structure of the 8-digit numbers
def FirstNumber (A B : ℕ) : ℕ := 85000000 + A * 100000 + 73000 + B * 100 + 20
def SecondNumber (A B C : ℕ) : ℕ := 41000000 + 700000 + A * 10000 + B * 1000 + 500 + C * 10 + 9

-- Define the condition for being a multiple of 5
def IsMultipleOf5 (n : ℕ) : Prop := ∃ k : ℕ, n = 5 * k

-- State the theorem
theorem find_C_value (A B : ℕ) (h1 : IsMultipleOf5 (FirstNumber A B)) 
  (h2 : ∃ C : ℕ, IsMultipleOf5 (SecondNumber A B C)) : 
  ∃ C : ℕ, C = 1 ∧ IsMultipleOf5 (SecondNumber A B C) :=
sorry

end find_C_value_l3315_331530


namespace sum_of_digits_product_of_nines_l3315_331509

/-- 
Given a natural number n, define a function that calculates the product:
9 × 99 × 9999 × ⋯ × (99...99) where the number of nines doubles in each factor
and the last factor has 2^n nines.
-/
def productOfNines (n : ℕ) : ℕ :=
  (List.range n).foldl (λ acc i => acc * (10^(2^i) - 1)) 9

/-- 
Sum of digits function
-/
def sumOfDigits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sumOfDigits (n / 10)

/-- 
Theorem: The sum of the digits of the product of nines is equal to 9 * 2^n
-/
theorem sum_of_digits_product_of_nines (n : ℕ) :
  sumOfDigits (productOfNines n) = 9 * 2^n := by
  sorry

end sum_of_digits_product_of_nines_l3315_331509


namespace system_one_solution_system_two_solution_l3315_331575

-- System 1
theorem system_one_solution (x : ℝ) : 
  (2 * x > 1 - x ∧ x + 2 < 4 * x - 1) ↔ x > 1 :=
sorry

-- System 2
theorem system_two_solution (x : ℝ) : 
  ((2 / 3) * x + 5 > 1 - x ∧ x - 1 ≤ (3 / 4) * x - (1 / 8)) ↔ 
  (-12 / 5 < x ∧ x ≤ 7 / 2) :=
sorry

end system_one_solution_system_two_solution_l3315_331575


namespace function_value_at_2004_l3315_331541

/-- Given a function f(x) = a*sin(πx + α) + b*cos(πx + β) + 4,
    where α, β, a, and b are non-zero real numbers, and f(2003) = 6,
    prove that f(2004) = 2. -/
theorem function_value_at_2004 
  (α β a b : ℝ) 
  (ha : a ≠ 0) (hb : b ≠ 0) 
  (f : ℝ → ℝ) 
  (hf : ∀ x, f x = a * Real.sin (Real.pi * x + α) + b * Real.cos (Real.pi * x + β) + 4) 
  (h2003 : f 2003 = 6) : 
  f 2004 = 2 := by
  sorry

end function_value_at_2004_l3315_331541


namespace area_ratio_inscribed_squares_l3315_331597

/-- A square inscribed in a circle -/
structure InscribedSquare :=
  (center : ℝ × ℝ)
  (radius : ℝ)
  (side : ℝ)

/-- A square with two vertices on a side of another square and two vertices on a circle -/
structure PartiallyInscribedSquare :=
  (center : ℝ × ℝ)
  (radius : ℝ)
  (side : ℝ)

/-- The theorem stating the relationship between the areas of the two squares -/
theorem area_ratio_inscribed_squares 
  (ABCD : InscribedSquare) 
  (EFGH : PartiallyInscribedSquare) 
  (h1 : ABCD.center = EFGH.center) 
  (h2 : ABCD.radius = EFGH.radius) 
  (h3 : ABCD.side ^ 2 = 1) : 
  EFGH.side ^ 2 = 1 / 25 := by
  sorry

end area_ratio_inscribed_squares_l3315_331597


namespace parabola_intersection_points_l3315_331588

/-- Given a quadratic equation with specific roots, prove the intersection points of a related parabola with the x-axis -/
theorem parabola_intersection_points 
  (a m : ℝ) 
  (h1 : a * (-1 + m)^2 = 3) 
  (h2 : a * (3 + m)^2 = 3) :
  let f (x : ℝ) := a * (x + m - 2)^2 - 3
  ∃ (x1 x2 : ℝ), x1 = 5 ∧ x2 = 1 ∧ f x1 = 0 ∧ f x2 = 0 :=
by sorry

end parabola_intersection_points_l3315_331588


namespace consecutive_substring_perfect_square_l3315_331508

/-- A type representing a 16-digit positive integer -/
def SixteenDigitInteger := { n : ℕ // 10^15 ≤ n ∧ n < 10^16 }

/-- A function that checks if a natural number is a perfect square -/
def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

/-- A function that returns the product of digits in a substring of a number -/
def substring_product (n : ℕ) (start finish : ℕ) : ℕ := sorry

/-- The main theorem: For any 16-digit positive integer, there exists a consecutive
    substring of digits whose product is a perfect square -/
theorem consecutive_substring_perfect_square (A : SixteenDigitInteger) :
  ∃ start finish : ℕ, start ≤ finish ∧ finish ≤ 16 ∧
    is_perfect_square (substring_product A.val start finish) := by
  sorry

end consecutive_substring_perfect_square_l3315_331508


namespace min_distance_between_curves_l3315_331528

/-- The minimum squared distance between a point on y = x^2 + 3ln(x) and a point on y = x + 2 -/
theorem min_distance_between_curves : ∀ (a b c d : ℝ),
  b = a^2 + 3 * Real.log a →  -- P(a,b) is on y = x^2 + 3ln(x)
  d = c + 2 →                 -- Q(c,d) is on y = x + 2
  (∀ x y z w : ℝ, 
    y = x^2 + 3 * Real.log x → 
    w = z + 2 → 
    (a - c)^2 + (b - d)^2 ≤ (x - z)^2 + (y - w)^2) →
  (a - c)^2 + (b - d)^2 = 8 := by
sorry

end min_distance_between_curves_l3315_331528


namespace ratio_problem_l3315_331507

theorem ratio_problem (a b c : ℝ) (h1 : b/a = 4) (h2 : c/b = 2) :
  (a + b + c) / (a + b) = 13/5 := by
  sorry

end ratio_problem_l3315_331507


namespace greatest_possible_area_l3315_331559

/-- A convex equilateral pentagon with side length 2 and two right angles -/
structure ConvexEquilateralPentagon where
  side_length : ℝ
  has_two_right_angles : Prop
  is_convex : Prop
  is_equilateral : Prop
  side_length_eq_two : side_length = 2

/-- The area of a ConvexEquilateralPentagon -/
def area (p : ConvexEquilateralPentagon) : ℝ := sorry

theorem greatest_possible_area (p : ConvexEquilateralPentagon) :
  area p ≤ 4 + Real.sqrt 7 :=
sorry

end greatest_possible_area_l3315_331559


namespace find_y_value_l3315_331519

theorem find_y_value (y : ℝ) (h : (15^2 * 8^3) / y = 450) : y = 256 := by
  sorry

end find_y_value_l3315_331519
