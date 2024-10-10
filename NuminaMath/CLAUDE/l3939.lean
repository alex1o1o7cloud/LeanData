import Mathlib

namespace complement_intersection_theorem_intersection_equality_range_l3939_393981

-- Define the sets A and B
def A : Set ℝ := {x | (x - 5) / (x - 2) ≤ 0}
def B (a : ℝ) : Set ℝ := {x | x^2 - 2*a*x + a^2 - 1 < 0}

-- Theorem for part 1
theorem complement_intersection_theorem :
  (Set.univ \ A) ∩ (Set.univ \ B 2) = {x | x ≤ 1 ∨ x > 5} := by sorry

-- Theorem for part 2
theorem intersection_equality_range :
  {a : ℝ | A ∩ B a = B a} = {a : ℝ | 3 ≤ a ∧ a ≤ 4} := by sorry

end complement_intersection_theorem_intersection_equality_range_l3939_393981


namespace orchard_sections_count_l3939_393943

/-- The number of sacks harvested from each section daily -/
def sacks_per_section : ℕ := 45

/-- The total number of sacks harvested daily -/
def total_sacks : ℕ := 360

/-- The number of sections in the orchard -/
def num_sections : ℕ := total_sacks / sacks_per_section

theorem orchard_sections_count :
  num_sections = 8 :=
sorry

end orchard_sections_count_l3939_393943


namespace max_value_of_f_l3939_393986

-- Define the function f(x) = ln(x) / x
noncomputable def f (x : ℝ) : ℝ := (Real.log x) / x

-- State the theorem
theorem max_value_of_f :
  ∃ (c : ℝ), c > 0 ∧ 
  (∀ x > 0, f x ≤ f c) ∧
  f c = 1 / Real.exp 1 := by
  sorry

end max_value_of_f_l3939_393986


namespace sum_of_digits_of_B_is_seven_l3939_393971

-- Define the function to calculate the sum of digits
def sumOfDigits (n : ℕ) : ℕ := sorry

-- Define A as the sum of digits of 4444^4144
def A : ℕ := sumOfDigits (4444^4144)

-- Define B as the sum of digits of A
def B : ℕ := sumOfDigits A

-- Theorem to prove
theorem sum_of_digits_of_B_is_seven : sumOfDigits B = 7 := by sorry

end sum_of_digits_of_B_is_seven_l3939_393971


namespace triangle_side_and_area_l3939_393916

/-- Given a triangle ABC with side lengths a, b, c and angle A, prove the length of side a and the area of the triangle. -/
theorem triangle_side_and_area 
  (b c : ℝ) 
  (A : ℝ) 
  (hb : b = 4) 
  (hc : c = 2) 
  (hA : Real.cos A = 1/4) :
  ∃ (a : ℝ), 
    a = 4 ∧ 
    (1/2 * b * c * Real.sin A : ℝ) = Real.sqrt 15 :=
by sorry

end triangle_side_and_area_l3939_393916


namespace simplify_expression_l3939_393918

theorem simplify_expression :
  4 * (12 / 9) * (36 / -45) = -12 / 5 := by
  sorry

end simplify_expression_l3939_393918


namespace expression_evaluation_l3939_393977

theorem expression_evaluation : 4 * (9 - 6) / 2 - 3 = 3 := by
  sorry

end expression_evaluation_l3939_393977


namespace quadratic_function_properties_l3939_393984

/-- A quadratic function with given vertex and y-intercept -/
def quadratic_function (x : ℝ) : ℝ := 2 * (x - 2)^2 - 4

theorem quadratic_function_properties :
  (∀ x, quadratic_function x = 2 * (x - 2)^2 - 4) ∧
  (quadratic_function 2 = -4) ∧
  (quadratic_function 0 = 4) ∧
  (quadratic_function 3 ≠ 5) :=
sorry

#check quadratic_function_properties

end quadratic_function_properties_l3939_393984


namespace supplement_quadruple_complement_30_l3939_393953

/-- The degree measure of the supplement of the quadruple of the complement of a 30-degree angle is 120 degrees. -/
theorem supplement_quadruple_complement_30 : 
  let initial_angle : ℝ := 30
  let complement := 90 - initial_angle
  let quadruple := 4 * complement
  let supplement := if quadruple ≤ 180 then 180 - quadruple else 360 - quadruple
  supplement = 120 := by sorry

end supplement_quadruple_complement_30_l3939_393953


namespace opposite_face_of_y_l3939_393997

-- Define a cube net
structure CubeNet where
  faces : Finset Char
  y_face : Char
  foldable : Bool

-- Define a property for opposite faces in a cube
def opposite_faces (net : CubeNet) (face1 face2 : Char) : Prop :=
  face1 ∈ net.faces ∧ face2 ∈ net.faces ∧ face1 ≠ face2

-- Theorem statement
theorem opposite_face_of_y (net : CubeNet) :
  net.faces = {'W', 'X', 'Y', 'Z', 'V', net.y_face} →
  net.foldable = true →
  net.y_face ≠ 'V' →
  opposite_faces net net.y_face 'V' :=
sorry

end opposite_face_of_y_l3939_393997


namespace subset_implies_C_C_complete_l3939_393970

def A (a : ℝ) : Set ℝ := {x | a * x - 2 = 0}
def B : Set ℝ := {x | x^2 - 3*x + 2 = 0}
def C : Set ℝ := {1, 2}

theorem subset_implies_C (a : ℝ) (h : A a ⊆ B) : a ∈ C := by
  sorry

theorem C_complete : ∀ a ∈ C, A a ⊆ B := by
  sorry

end subset_implies_C_C_complete_l3939_393970


namespace weight_of_new_person_l3939_393908

/-- Given a group of 8 people, if replacing a 65 kg person with a new person
    increases the average weight by 1.5 kg, then the weight of the new person is 77 kg. -/
theorem weight_of_new_person
  (initial_count : ℕ)
  (weight_replaced : ℝ)
  (avg_increase : ℝ)
  (h_count : initial_count = 8)
  (h_replaced : weight_replaced = 65)
  (h_increase : avg_increase = 1.5) :
  weight_replaced + initial_count * avg_increase = 77 :=
by sorry

end weight_of_new_person_l3939_393908


namespace complement_of_union_l3939_393927

-- Define the sets A and B
def A : Set ℝ := {x | x < 0}
def B : Set ℝ := {x | x ≥ 2}

-- Define the set C as the complement of A ∪ B in ℝ
def C : Set ℝ := (A ∪ B)ᶜ

-- Theorem statement
theorem complement_of_union :
  C = {x : ℝ | 0 ≤ x ∧ x < 2} :=
sorry

end complement_of_union_l3939_393927


namespace largest_difference_of_three_digit_numbers_l3939_393992

/-- A function that represents a 3-digit number given its digits -/
def threeDigitNumber (a b c : Nat) : Nat := 100 * a + 10 * b + c

/-- The set of valid digits -/
def validDigits : Finset Nat := Finset.range 9

theorem largest_difference_of_three_digit_numbers :
  ∃ (U V W X Y Z : Nat),
    U ∈ validDigits ∧ V ∈ validDigits ∧ W ∈ validDigits ∧
    X ∈ validDigits ∧ Y ∈ validDigits ∧ Z ∈ validDigits ∧
    U ≠ V ∧ U ≠ W ∧ U ≠ X ∧ U ≠ Y ∧ U ≠ Z ∧
    V ≠ W ∧ V ≠ X ∧ V ≠ Y ∧ V ≠ Z ∧
    W ≠ X ∧ W ≠ Y ∧ W ≠ Z ∧
    X ≠ Y ∧ X ≠ Z ∧
    Y ≠ Z ∧
    threeDigitNumber U V W - threeDigitNumber X Y Z = 864 ∧
    ∀ (A B C D E F : Nat),
      A ∈ validDigits → B ∈ validDigits → C ∈ validDigits →
      D ∈ validDigits → E ∈ validDigits → F ∈ validDigits →
      A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ A ≠ E ∧ A ≠ F ∧
      B ≠ C ∧ B ≠ D ∧ B ≠ E ∧ B ≠ F ∧
      C ≠ D ∧ C ≠ E ∧ C ≠ F ∧
      D ≠ E ∧ D ≠ F ∧
      E ≠ F →
      threeDigitNumber A B C - threeDigitNumber D E F ≤ 864 :=
by
  sorry


end largest_difference_of_three_digit_numbers_l3939_393992


namespace four_people_handshakes_l3939_393966

/-- The number of handshakes in a group where each person shakes hands with every other person exactly once -/
def num_handshakes (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem: In a group of 4 people, where each person shakes hands with every other person exactly once, the total number of handshakes is 6. -/
theorem four_people_handshakes : num_handshakes 4 = 6 := by
  sorry

end four_people_handshakes_l3939_393966


namespace sum_three_x_square_y_correct_l3939_393973

/-- The sum of three times x and the square of y -/
def sum_three_x_square_y (x y : ℝ) : ℝ := 3 * x + y^2

theorem sum_three_x_square_y_correct (x y : ℝ) : 
  sum_three_x_square_y x y = 3 * x + y^2 := by
  sorry

end sum_three_x_square_y_correct_l3939_393973


namespace arithmetic_progression_problem_l3939_393922

theorem arithmetic_progression_problem (a d : ℝ) : 
  (a - 2*d)^3 + (a - d)^3 + a^3 + (a + d)^3 + (a + 2*d)^3 = 0 ∧
  (a - 2*d)^4 + (a - d)^4 + a^4 + (a + d)^4 + (a + 2*d)^4 = 136 →
  a - 2*d = -2 * Real.sqrt 2 :=
by sorry

end arithmetic_progression_problem_l3939_393922


namespace mode_is_180_l3939_393934

/-- Represents the electricity consumption data for households -/
structure ElectricityData where
  consumption : List Nat
  frequency : List Nat
  total_households : Nat

/-- Calculates the mode of a list of numbers -/
def mode (data : ElectricityData) : Nat :=
  let paired_data := data.consumption.zip data.frequency
  let max_frequency := paired_data.map Prod.snd |>.maximum?
  match max_frequency with
  | none => 0  -- Default value if the list is empty
  | some max => 
      (paired_data.filter (fun p => p.2 = max)).map Prod.fst |>.head!

/-- The electricity consumption survey data -/
def survey_data : ElectricityData := {
  consumption := [120, 140, 160, 180, 200],
  frequency := [5, 5, 3, 6, 1],
  total_households := 20
}

theorem mode_is_180 : mode survey_data = 180 := by
  sorry

end mode_is_180_l3939_393934


namespace book_selling_price_l3939_393936

theorem book_selling_price (CP : ℝ) : 
  (0.9 * CP = CP - 0.1 * CP) →  -- 10% loss condition
  (1.1 * CP = 990) →            -- 10% gain condition
  (0.9 * CP = 810) :=           -- Original selling price
by sorry

end book_selling_price_l3939_393936


namespace M_subset_N_l3939_393928

def M : Set ℕ := {x | ∃ a : ℕ, x = a^2 + 2*a + 2}
def N : Set ℕ := {y | ∃ b : ℕ, y = b^2 - 4*b + 5}

theorem M_subset_N : M ⊆ N := by sorry

end M_subset_N_l3939_393928


namespace min_additional_marbles_correct_l3939_393923

/-- The number of friends Lisa has -/
def num_friends : ℕ := 10

/-- The initial number of marbles Lisa has -/
def initial_marbles : ℕ := 34

/-- The sum of the first n natural numbers -/
def sum_first_n (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The minimum number of additional marbles needed -/
def min_additional_marbles : ℕ := sum_first_n num_friends - initial_marbles

theorem min_additional_marbles_correct :
  min_additional_marbles = 21 ∧
  sum_first_n num_friends ≥ initial_marbles + min_additional_marbles ∧
  ∀ k : ℕ, k < min_additional_marbles →
    sum_first_n num_friends > initial_marbles + k :=
by sorry

end min_additional_marbles_correct_l3939_393923


namespace intersection_A_B_union_A_C_equals_C_l3939_393989

-- Define the sets A, B, and C
def A : Set ℝ := {x | -2 ≤ x ∧ x < 5}
def B : Set ℝ := {x | 3 * x - 5 ≥ x - 1}
def C (m : ℝ) : Set ℝ := {x | -x + m > 0}

-- Theorem for part 1
theorem intersection_A_B : A ∩ B = {x | 2 ≤ x ∧ x < 5} := by sorry

-- Theorem for part 2
theorem union_A_C_equals_C (m : ℝ) : A ∪ C m = C m ↔ m ≥ 5 := by sorry

end intersection_A_B_union_A_C_equals_C_l3939_393989


namespace quadratic_equation_solution_l3939_393950

theorem quadratic_equation_solution :
  let a : ℚ := -2
  let b : ℚ := 1
  let c : ℚ := 3
  let x₁ : ℚ := -1
  let x₂ : ℚ := 3/2
  (a * x₁^2 + b * x₁ + c = 0) ∧ (a * x₂^2 + b * x₂ + c = 0) :=
by sorry

end quadratic_equation_solution_l3939_393950


namespace max_sales_on_day_40_l3939_393972

def salesVolume (t : ℕ) : ℝ := -t + 110

def price (t : ℕ) : ℝ :=
  if t ≤ 40 then t + 8 else -0.5 * t + 69

def salesAmount (t : ℕ) : ℝ := salesVolume t * price t

theorem max_sales_on_day_40 :
  ∀ t : ℕ, 1 ≤ t ∧ t ≤ 100 → salesAmount t ≤ salesAmount 40 ∧ salesAmount 40 = 3360 :=
by sorry

end max_sales_on_day_40_l3939_393972


namespace prob_sum_not_greater_than_4_prob_first_less_than_second_plus_2_l3939_393994

-- Define the sample space for a single die throw
def Die : Type := Fin 6

-- Define the sample space for two dice throws
def TwoDice : Type := Die × Die

-- Define the probability measure on TwoDice
noncomputable def P : Set TwoDice → ℝ := sorry

-- Define the event where the sum of dice is not greater than 4
def SumNotGreaterThan4 : Set TwoDice :=
  {x | x.1.val + x.2.val ≤ 4}

-- Define the event where the first die is less than the second die plus 2
def FirstLessThanSecondPlus2 : Set TwoDice :=
  {x | x.1.val < x.2.val + 2}

-- Theorem 1: Probability that the sum of dice is not greater than 4 is 1/6
theorem prob_sum_not_greater_than_4 :
  P SumNotGreaterThan4 = 1/6 := by sorry

-- Theorem 2: Probability that the first die is less than the second die plus 2 is 13/18
theorem prob_first_less_than_second_plus_2 :
  P FirstLessThanSecondPlus2 = 13/18 := by sorry

end prob_sum_not_greater_than_4_prob_first_less_than_second_plus_2_l3939_393994


namespace prime_equation_solutions_l3939_393911

theorem prime_equation_solutions (p : ℕ) (hp : Prime p) (hp_mod : p % 8 = 3) :
  ∀ (x y : ℚ), p^2 * x^4 - 6*p*x^2 + 1 = y^2 ↔ (x = 0 ∧ (y = 1 ∨ y = -1)) :=
by sorry

end prime_equation_solutions_l3939_393911


namespace deepak_age_l3939_393931

theorem deepak_age (arun_age deepak_age : ℕ) : 
  (arun_age : ℚ) / deepak_age = 2 / 5 →
  arun_age + 10 = 30 →
  deepak_age = 50 := by
sorry

end deepak_age_l3939_393931


namespace dallas_age_l3939_393919

theorem dallas_age (dexter_age : ℕ) (darcy_age : ℕ) (dallas_age_last_year : ℕ) :
  dexter_age = 8 →
  darcy_age = 2 * dexter_age →
  dallas_age_last_year = 3 * (darcy_age - 1) →
  dallas_age_last_year + 1 = 46 :=
by
  sorry

end dallas_age_l3939_393919


namespace quadratic_real_roots_condition_l3939_393983

theorem quadratic_real_roots_condition (k : ℝ) : 
  (∃ x : ℝ, k * x^2 - 2 * x + 3 = 0) ↔ (k ≤ 1/3 ∧ k ≠ 0) :=
by sorry

end quadratic_real_roots_condition_l3939_393983


namespace scarf_cost_proof_l3939_393913

def sweater_cost : ℕ := 30
def num_items : ℕ := 6
def total_savings : ℕ := 500
def remaining_savings : ℕ := 200

theorem scarf_cost_proof :
  ∃ (scarf_cost : ℕ),
    scarf_cost * num_items = total_savings - remaining_savings - (sweater_cost * num_items) :=
by sorry

end scarf_cost_proof_l3939_393913


namespace range_of_m_l3939_393960

-- Define the function y in terms of x, k, and m
def y (x k m : ℝ) : ℝ := k * x - k + m

-- State the theorem
theorem range_of_m (k m : ℝ) : 
  (∃ x, y x k m = 3 ∧ x = -2) →  -- When x = -2, y = 3
  (k ≠ 0) →  -- k is non-zero (implied by direct proportionality)
  (k < 0) →  -- Slope is negative (passes through 2nd, 3rd, and 4th quadrants)
  (-k + m < 0) →  -- y-intercept is negative (passes through 2nd, 3rd, and 4th quadrants)
  m < -3/2 := by
sorry

end range_of_m_l3939_393960


namespace absolute_difference_of_solution_l3939_393955

theorem absolute_difference_of_solution (x y : ℝ) : 
  (Int.floor x + (y - Int.floor y) = 3.7) →
  ((x - Int.floor x) + Int.floor y = 6.2) →
  |x - y| = 3.5 := by
sorry

end absolute_difference_of_solution_l3939_393955


namespace reciprocal_power_2014_l3939_393912

theorem reciprocal_power_2014 (a : ℚ) (h : a ≠ 0) : (a = a⁻¹) → a^2014 = 1 := by
  sorry

end reciprocal_power_2014_l3939_393912


namespace zero_points_of_f_l3939_393969

def f (x : ℝ) := 2 * x^2 + 3 * x + 1

theorem zero_points_of_f :
  ∃ x y : ℝ, x ≠ y ∧ f x = 0 ∧ f y = 0 ∧ 
  ∀ z : ℝ, f z = 0 → z = x ∨ z = y ∧
  x = -1/2 ∧ y = -1 :=
sorry

end zero_points_of_f_l3939_393969


namespace a_formula_S_min_l3939_393963

-- Define the sequence and its sum
def S (n : ℕ) : ℤ := n^2 - 48*n

def a : ℕ → ℤ
  | 0 => 0  -- We define a₀ = 0 to make a total function
  | n + 1 => S (n + 1) - S n

-- Theorem for the general formula of a_n
theorem a_formula (n : ℕ) : a (n + 1) = 2 * (n + 1) - 49 := by sorry

-- Theorem for the minimum value of S_n
theorem S_min : ∃ n : ℕ, S n = -576 ∧ ∀ m : ℕ, S m ≥ -576 := by sorry

end a_formula_S_min_l3939_393963


namespace min_value_2x_plus_y_l3939_393915

theorem min_value_2x_plus_y (x y : ℝ) (hx : x > 0) (hy : y > 0) (heq : x * (x + y) = 5 * x + y) :
  ∃ (m : ℝ), m = 9 ∧ ∀ z, z = 2 * x + y → z ≥ m :=
by sorry

end min_value_2x_plus_y_l3939_393915


namespace sum_first_four_terms_l3939_393903

def a (n : ℕ) : ℤ := (-1)^n * (3*n - 2)

theorem sum_first_four_terms : 
  (a 1) + (a 2) + (a 3) + (a 4) = 6 := by
sorry

end sum_first_four_terms_l3939_393903


namespace brick_surface_area_l3939_393958

/-- The surface area of a rectangular prism. -/
def surface_area (length width height : ℝ) : ℝ :=
  2 * (length * width + length * height + width * height)

/-- Theorem: The surface area of a 10 cm x 4 cm x 3 cm brick is 164 cm². -/
theorem brick_surface_area :
  surface_area 10 4 3 = 164 := by
  sorry

end brick_surface_area_l3939_393958


namespace first_quarter_profit_determination_l3939_393957

/-- Represents the quarterly profits of a store in dollars. -/
structure QuarterlyProfits where
  first : ℕ
  third : ℕ
  fourth : ℕ

/-- Calculates the annual profit given quarterly profits. -/
def annualProfit (q : QuarterlyProfits) : ℕ :=
  q.first + q.third + q.fourth

/-- Theorem stating that given the annual profit and profits from the third and fourth quarters,
    the first quarter profit can be determined. -/
theorem first_quarter_profit_determination
  (annual_profit : ℕ)
  (third_quarter : ℕ)
  (fourth_quarter : ℕ)
  (h1 : third_quarter = 3000)
  (h2 : fourth_quarter = 2000)
  (h3 : annual_profit = 8000)
  (h4 : ∃ q : QuarterlyProfits, q.third = third_quarter ∧ q.fourth = fourth_quarter ∧ annualProfit q = annual_profit) :
  ∃ q : QuarterlyProfits, q.first = 3000 ∧ q.third = third_quarter ∧ q.fourth = fourth_quarter ∧ annualProfit q = annual_profit :=
by sorry

end first_quarter_profit_determination_l3939_393957


namespace jellybean_count_l3939_393967

/-- The number of jellybeans in a bag with black, green, and orange beans -/
def total_jellybeans (black green orange : ℕ) : ℕ := black + green + orange

/-- Theorem: The total number of jellybeans in the bag is 27 -/
theorem jellybean_count :
  ∀ (black green orange : ℕ),
  black = 8 →
  green = black + 2 →
  orange = green - 1 →
  total_jellybeans black green orange = 27 := by
sorry

end jellybean_count_l3939_393967


namespace isabellas_original_hair_length_l3939_393945

/-- The length of Isabella's hair before the haircut -/
def original_length : ℝ := sorry

/-- The length of Isabella's hair after the haircut -/
def after_haircut_length : ℝ := 9

/-- The length of hair that was cut off -/
def cut_length : ℝ := 9

/-- Theorem stating that Isabella's original hair length was 18 inches -/
theorem isabellas_original_hair_length :
  original_length = after_haircut_length + cut_length ∧ original_length = 18 := by
  sorry

end isabellas_original_hair_length_l3939_393945


namespace coin_toss_is_classical_model_l3939_393982

structure Experiment where
  name : String
  is_finite : Bool
  is_equiprobable : Bool

def is_classical_probability_model (e : Experiment) : Prop :=
  e.is_finite ∧ e.is_equiprobable

def seed_germination : Experiment :=
  { name := "Seed germination",
    is_finite := true,
    is_equiprobable := false }

def product_measurement : Experiment :=
  { name := "Product measurement",
    is_finite := false,
    is_equiprobable := false }

def coin_toss : Experiment :=
  { name := "Coin toss",
    is_finite := true,
    is_equiprobable := true }

def target_shooting : Experiment :=
  { name := "Target shooting",
    is_finite := true,
    is_equiprobable := false }

theorem coin_toss_is_classical_model :
  is_classical_probability_model coin_toss ∧
  ¬is_classical_probability_model seed_germination ∧
  ¬is_classical_probability_model product_measurement ∧
  ¬is_classical_probability_model target_shooting :=
by sorry

end coin_toss_is_classical_model_l3939_393982


namespace calculation_proof_l3939_393917

theorem calculation_proof : ((-2)^2 : ℝ) + Real.sqrt 16 - 2 * Real.sin (π / 6) + (2023 - Real.pi)^0 = 8 := by
  sorry

end calculation_proof_l3939_393917


namespace determine_back_iff_conditions_met_l3939_393944

/-- Represents a card with two sides -/
structure Card where
  side1 : Nat
  side2 : Nat

/-- Checks if a number appears on a card -/
def numberOnCard (c : Card) (n : Nat) : Prop :=
  c.side1 = n ∨ c.side2 = n

/-- Represents the deck of n cards -/
def deck (n : Nat) : List Card :=
  List.range n |>.map (λ i => ⟨i, i + 1⟩)

/-- Represents the cards seen so far -/
def SeenCards := List Nat

/-- Determines if the back of the last card can be identified -/
def canDetermineBack (n : Nat) (k : Nat) (seen : SeenCards) : Prop :=
  (k = 0 ∨ k = n) ∨
  (0 < k ∧ k < n ∧
    (seen.count (k + 1) = 2 ∨
     (∃ j, 1 ≤ j ∧ j ≤ n - k - 1 ∧
       (∀ i, k + 1 ≤ i ∧ i ≤ k + j → seen.count i ≥ 1) ∧
       (if k + j + 1 = n then seen.count n ≥ 1 else seen.count (k + j + 1) = 2)) ∨
     seen.count (k - 1) = 2 ∨
     (∃ j, 1 ≤ j ∧ j ≤ k - 1 ∧
       (∀ i, k - j ≤ i ∧ i ≤ k - 1 → seen.count i ≥ 1) ∧
       (if k - j - 1 = 0 then seen.count 0 ≥ 1 else seen.count (k - j - 1) = 2))))

/-- The main theorem to be proved -/
theorem determine_back_iff_conditions_met (n : Nat) (k : Nat) (seen : SeenCards) :
  canDetermineBack n k seen ↔
  (∀ (lastCard : Card),
    numberOnCard lastCard k →
    lastCard ∈ deck n →
    ∃! backNumber, numberOnCard lastCard backNumber ∧ backNumber ≠ k) := by
  sorry

end determine_back_iff_conditions_met_l3939_393944


namespace alice_commission_percentage_l3939_393942

/-- Proves that Alice's commission percentage is 2% given her sales, salary, and savings information --/
theorem alice_commission_percentage (sales : ℝ) (basic_salary : ℝ) (savings : ℝ) 
  (h1 : sales = 2500)
  (h2 : basic_salary = 240)
  (h3 : savings = 29)
  (h4 : savings = 0.1 * (basic_salary + sales * commission_rate)) :
  commission_rate = 0.02 := by
  sorry

#check alice_commission_percentage

end alice_commission_percentage_l3939_393942


namespace bridge_length_l3939_393905

/-- The length of a bridge given train parameters and crossing time -/
theorem bridge_length (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time : ℝ) : 
  train_length = 110 →
  train_speed_kmh = 45 →
  crossing_time = 30 →
  (train_speed_kmh * 1000 / 3600) * crossing_time - train_length = 265 := by
  sorry

end bridge_length_l3939_393905


namespace fraction_equality_l3939_393999

theorem fraction_equality (x : ℝ) (h : x = 5) : (x^4 - 8*x^2 + 16) / (x^2 - 4) = 21 := by
  sorry

end fraction_equality_l3939_393999


namespace min_value_of_expression_l3939_393975

def S : Finset Int := {-8, -6, -4, -1, 3, 5, 7, 14}

theorem min_value_of_expression (p q r s t u v w : Int) 
  (h_distinct : p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ p ≠ t ∧ p ≠ u ∧ p ≠ v ∧ p ≠ w ∧
                q ≠ r ∧ q ≠ s ∧ q ≠ t ∧ q ≠ u ∧ q ≠ v ∧ q ≠ w ∧
                r ≠ s ∧ r ≠ t ∧ r ≠ u ∧ r ≠ v ∧ r ≠ w ∧
                s ≠ t ∧ s ≠ u ∧ s ≠ v ∧ s ≠ w ∧
                t ≠ u ∧ t ≠ v ∧ t ≠ w ∧
                u ≠ v ∧ u ≠ w ∧
                v ≠ w)
  (h_in_S : p ∈ S ∧ q ∈ S ∧ r ∈ S ∧ s ∈ S ∧ t ∈ S ∧ u ∈ S ∧ v ∈ S ∧ w ∈ S) :
  ∃ (x : Int), 3 * (p + q + r + s)^2 + (t + u + v + w)^2 ≥ 300 ∧
               3 * x^2 + (20 - x)^2 = 300 :=
by sorry

end min_value_of_expression_l3939_393975


namespace scientific_notation_of_small_number_l3939_393935

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  significand : ℝ
  exponent : ℤ
  is_valid : 1 ≤ |significand| ∧ |significand| < 10

/-- Converts a real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem scientific_notation_of_small_number :
  toScientificNotation 0.00000002 = ScientificNotation.mk 2 (-8) sorry := by
  sorry

end scientific_notation_of_small_number_l3939_393935


namespace complex_magnitude_equality_l3939_393976

theorem complex_magnitude_equality (t : ℝ) : 
  t > 0 → (Complex.abs (-5 + t * Complex.I) = 3 * Real.sqrt 6 ↔ t = Real.sqrt 29) := by
  sorry

end complex_magnitude_equality_l3939_393976


namespace unique_box_dimensions_l3939_393910

theorem unique_box_dimensions : ∃! (a b c : ℕ+), 
  (a ≥ b) ∧ (b ≥ c) ∧ 
  (a.val * b.val * c.val = 2 * (a.val * b.val + a.val * c.val + b.val * c.val)) := by
  sorry

end unique_box_dimensions_l3939_393910


namespace inscribed_circle_diameter_l3939_393965

theorem inscribed_circle_diameter (DE DF EF : ℝ) (h1 : DE = 10) (h2 : DF = 5) (h3 : EF = 9) :
  let s := (DE + DF + EF) / 2
  let area := Real.sqrt (s * (s - DE) * (s - DF) * (s - EF))
  let diameter := 2 * area / s
  diameter = Real.sqrt 14 := by sorry

end inscribed_circle_diameter_l3939_393965


namespace polynomial_divisibility_l3939_393933

theorem polynomial_divisibility (r : ℝ) :
  (∃ s : ℝ, 10 * X^3 - 5 * X^2 - 52 * X + 60 = 10 * (X - r)^2 * (X - s)) →
  r = -3/2 := by
sorry

end polynomial_divisibility_l3939_393933


namespace particle_movement_probability_reach_origin_l3939_393978

/-- Probability of reaching (0,0) from (x,y) before hitting any other point on the axes -/
def P (x y : ℕ) : ℚ :=
  if x = 0 ∧ y = 0 then 1
  else if x = 0 ∨ y = 0 then 0
  else (P (x-1) y + P x (y-1) + P (x-1) (y-1)) / 3

/-- The particle's movement rules and starting position -/
theorem particle_movement (x y : ℕ) (h : x > 0 ∧ y > 0) :
  P x y = (P (x-1) y + P x (y-1) + P (x-1) (y-1)) / 3 :=
sorry

/-- The probability of reaching (0,0) from (5,5) -/
theorem probability_reach_origin : P 5 5 = 381 / 2187 :=
sorry

end particle_movement_probability_reach_origin_l3939_393978


namespace ball_count_in_bag_l3939_393947

/-- Given a bag with red, black, and white balls, prove that the total number of balls is 7
    when the probability of drawing a red ball equals the probability of drawing a white ball. -/
theorem ball_count_in_bag (x : ℕ) : 
  (3 : ℚ) / (4 + x) = (x : ℚ) / (4 + x) → 3 + 1 + x = 7 := by
  sorry

end ball_count_in_bag_l3939_393947


namespace largest_whole_number_nine_times_less_than_200_l3939_393946

theorem largest_whole_number_nine_times_less_than_200 :
  ∀ x : ℕ, x ≤ 22 ↔ 9 * x < 200 :=
by sorry

end largest_whole_number_nine_times_less_than_200_l3939_393946


namespace prob_xavier_yvonne_not_zelda_l3939_393988

-- Define the difficulty factors and probabilities
variable (a b c : ℝ)
variable (p_xavier : ℝ := (1/3)^a)
variable (p_yvonne : ℝ := (1/2)^b)
variable (p_zelda : ℝ := (5/8)^c)

-- Define the theorem
theorem prob_xavier_yvonne_not_zelda :
  p_xavier * p_yvonne * (1 - p_zelda) = (1/16) * a * b * c := by
  sorry

end prob_xavier_yvonne_not_zelda_l3939_393988


namespace symmetric_point_wrt_origin_l3939_393914

/-- Given a point P(-2, 3) in a Cartesian coordinate system, 
    its symmetric point with respect to the origin has coordinates (2, -3). -/
theorem symmetric_point_wrt_origin : 
  let P : ℝ × ℝ := (-2, 3)
  let symmetric_point (p : ℝ × ℝ) : ℝ × ℝ := (-p.1, -p.2)
  symmetric_point P = (2, -3) := by sorry

end symmetric_point_wrt_origin_l3939_393914


namespace max_value_of_expression_l3939_393987

theorem max_value_of_expression (x : ℝ) (h : -1 ≤ x ∧ x ≤ 2) :
  ∃ (max : ℝ), max = 5 ∧ ∀ y, -1 ≤ y ∧ y ≤ 2 → 2 + |y - 2| ≤ max :=
by sorry

end max_value_of_expression_l3939_393987


namespace percentage_of_non_science_majors_l3939_393954

theorem percentage_of_non_science_majors
  (women_science_percentage : Real)
  (men_class_percentage : Real)
  (men_science_percentage : Real)
  (h1 : women_science_percentage = 0.1)
  (h2 : men_class_percentage = 0.4)
  (h3 : men_science_percentage = 0.8500000000000001) :
  1 - (women_science_percentage * (1 - men_class_percentage) +
       men_science_percentage * men_class_percentage) = 0.6 := by
  sorry

end percentage_of_non_science_majors_l3939_393954


namespace sum_of_digits_of_n_is_nine_l3939_393904

/-- Two distinct digits -/
def distinct_digits (d e : Nat) : Prop :=
  d ≠ e ∧ d < 10 ∧ e < 10

/-- Sum of digits is prime -/
def sum_is_prime (d e : Nat) : Prop :=
  Nat.Prime (d + e)

/-- k is prime and greater than both d and e -/
def k_is_valid_prime (d e k : Nat) : Prop :=
  Nat.Prime k ∧ k > d ∧ k > e

/-- n is the product of d, e, and k -/
def n_is_product (n d e k : Nat) : Prop :=
  n = d * e * k

/-- k is related to d and e -/
def k_relation (d e k : Nat) : Prop :=
  k = 10 * d + e

/-- n is the largest such product -/
def n_is_largest (n : Nat) : Prop :=
  ∀ m d e k, distinct_digits d e → sum_is_prime d e → k_is_valid_prime d e k →
    k_relation d e k → n_is_product m d e k → m ≤ n

/-- n is the smallest multiple of k -/
def n_is_smallest_multiple (n k : Nat) : Prop :=
  k ∣ n ∧ ∀ m, m < n → ¬(k ∣ m)

/-- Sum of digits of a number -/
def sum_of_digits (n : Nat) : Nat :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

theorem sum_of_digits_of_n_is_nine :
  ∃ n d e k, distinct_digits d e ∧ sum_is_prime d e ∧ k_is_valid_prime d e k ∧
    k_relation d e k ∧ n_is_product n d e k ∧ n_is_largest n ∧
    n_is_smallest_multiple n k ∧ sum_of_digits n = 9 :=
sorry

end sum_of_digits_of_n_is_nine_l3939_393904


namespace train_crossing_time_l3939_393906

/-- Proves that a train 40 meters long, traveling at 144 km/hr, will take 1 second to cross an electric pole. -/
theorem train_crossing_time (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time : ℝ) :
  train_length = 40 →
  train_speed_kmh = 144 →
  crossing_time = train_length / (train_speed_kmh * 1000 / 3600) →
  crossing_time = 1 := by sorry

end train_crossing_time_l3939_393906


namespace inequality_system_solution_l3939_393974

theorem inequality_system_solution :
  ∀ x : ℝ, (2 * x + 1 < 3 * x - 2 ∧ 3 * (x - 2) - x ≤ 4) ↔ (3 < x ∧ x ≤ 5) := by
  sorry

end inequality_system_solution_l3939_393974


namespace f_of_one_eq_six_l3939_393900

def f (x : ℝ) : ℝ := x^2 + 2*x + 3

theorem f_of_one_eq_six : f 1 = 6 := by
  sorry

end f_of_one_eq_six_l3939_393900


namespace quadratic_inequality_solution_set_l3939_393991

theorem quadratic_inequality_solution_set :
  {x : ℝ | 3 * x^2 - 2 * x - 8 < 0} = {x : ℝ | -4/3 < x ∧ x < 2} := by
  sorry

end quadratic_inequality_solution_set_l3939_393991


namespace outfit_combinations_l3939_393932

/-- The number of shirts available -/
def num_shirts : ℕ := 8

/-- The number of pairs of pants available -/
def num_pants : ℕ := 5

/-- The number of ties available -/
def num_ties : ℕ := 6

/-- The number of jackets available -/
def num_jackets : ℕ := 2

/-- The number of different outfits that can be created -/
def num_outfits : ℕ := num_shirts * num_pants * (num_ties + 1) * (num_jackets + 1)

/-- Theorem stating that the number of different outfits is 840 -/
theorem outfit_combinations : num_outfits = 840 := by
  sorry

end outfit_combinations_l3939_393932


namespace original_denominator_problem_l3939_393998

theorem original_denominator_problem (d : ℚ) : 
  (3 : ℚ) / d ≠ 0 →  -- Ensure the original fraction is well-defined
  (3 + 4 : ℚ) / (d + 4) = 1 / 3 → 
  d = 17 := by
sorry

end original_denominator_problem_l3939_393998


namespace student_teacher_ratio_l3939_393924

/-- Proves that the current ratio of students to teachers is 50:1 given the problem conditions -/
theorem student_teacher_ratio 
  (current_teachers : ℕ) 
  (current_students : ℕ) 
  (h1 : current_teachers = 3)
  (h2 : (current_students + 50) / (current_teachers + 5) = 25) :
  current_students / current_teachers = 50 := by
sorry

end student_teacher_ratio_l3939_393924


namespace max_tiles_on_floor_l3939_393962

/-- Represents the dimensions of a rectangle -/
structure Dimensions where
  length : ℕ
  width : ℕ

/-- Calculates the maximum number of tiles that can be placed in a checkerboard pattern -/
def maxTiles (floor : Dimensions) (tile : Dimensions) : ℕ :=
  let lengthTiles := floor.length / tile.length
  let widthTiles := floor.width / tile.width
  (lengthTiles / 2) * (widthTiles / 2)

/-- Theorem stating the maximum number of tiles that can be placed on the given floor -/
theorem max_tiles_on_floor :
  let floor := Dimensions.mk 280 240
  let tile := Dimensions.mk 40 28
  maxTiles floor tile = 12 := by
  sorry

end max_tiles_on_floor_l3939_393962


namespace quadratic_equation_solution_l3939_393993

theorem quadratic_equation_solution (a : ℝ) : 
  (∀ x : ℝ, x = 1 → a * x^2 - 6 * x + 3 = 0) → a = 3 := by
  sorry

end quadratic_equation_solution_l3939_393993


namespace degree_not_determined_by_A_P_l3939_393939

/-- A characteristic associated with a polynomial -/
def A_P (P : Polynomial ℝ) : Set ℝ := sorry

/-- Theorem stating that the degree of a polynomial cannot be uniquely determined from A_P -/
theorem degree_not_determined_by_A_P :
  ∃ (P1 P2 : Polynomial ℝ), A_P P1 = A_P P2 ∧ P1.degree ≠ P2.degree := by
  sorry

end degree_not_determined_by_A_P_l3939_393939


namespace two_digit_number_problem_l3939_393930

theorem two_digit_number_problem :
  ∀ (x y : ℕ),
    x < 10 ∧ y < 10 ∧  -- Ensures x and y are single digits
    y = x + 2 ∧        -- Unit's digit exceeds 10's digit by 2
    (10 * x + y) * (x + y) = 144  -- Product condition
    → 10 * x + y = 24 :=
by
  sorry

end two_digit_number_problem_l3939_393930


namespace constant_term_expansion_l3939_393952

-- Define the binomial coefficient
def binomial (n k : ℕ) : ℕ := sorry

-- Define the function for the expansion
def expansion_term (x : ℝ) (r : ℕ) : ℝ :=
  (-1)^r * binomial 16 r * x^(16 - 4*r/3)

-- State the theorem
theorem constant_term_expansion :
  expansion_term 1 12 = 1820 :=
sorry

end constant_term_expansion_l3939_393952


namespace moving_circle_trajectory_l3939_393949

/-- The equation of the fixed circle -/
def fixed_circle (x y : ℝ) : Prop := x^2 + y^2 - 4*x = 0

/-- The equation of the y-axis -/
def y_axis (x : ℝ) : Prop := x = 0

/-- The trajectory of the center of the moving circle -/
def trajectory (x y : ℝ) : Prop := (x > 0 ∧ y^2 = 8*x) ∨ (x ≤ 0 ∧ y = 0)

/-- Theorem stating the trajectory of the center of the moving circle -/
theorem moving_circle_trajectory :
  ∀ (x y : ℝ), 
  (∃ (r : ℝ), r > 0 ∧ 
    (∃ (x₀ y₀ : ℝ), fixed_circle x₀ y₀ ∧ (x - x₀)^2 + (y - y₀)^2 = r^2) ∧
    (∃ (x₁ : ℝ), y_axis x₁ ∧ (x - x₁)^2 + y^2 = r^2)) →
  trajectory x y :=
sorry

end moving_circle_trajectory_l3939_393949


namespace volume_weight_proportion_l3939_393980

/-- Given a substance where volume is directly proportional to weight,
    if 48 cubic inches of the substance weigh 112 ounces,
    then 56 ounces of the substance will have a volume of 24 cubic inches. -/
theorem volume_weight_proportion (volume weight : ℝ → ℝ) :
  (∀ w₁ w₂, volume w₁ / volume w₂ = w₁ / w₂) →  -- volume is directly proportional to weight
  volume 112 = 48 →                            -- 48 cubic inches weigh 112 ounces
  volume 56 = 24                               -- 56 ounces have a volume of 24 cubic inches
:= by sorry

end volume_weight_proportion_l3939_393980


namespace disaster_relief_team_selection_l3939_393985

/-- The number of internal medicine doctors -/
def internal_doctors : ℕ := 12

/-- The number of surgeons -/
def surgeons : ℕ := 8

/-- The size of the disaster relief medical team -/
def team_size : ℕ := 5

/-- Doctor A is an internal medicine doctor -/
def doctor_A : Fin internal_doctors := sorry

/-- Doctor B is a surgeon -/
def doctor_B : Fin surgeons := sorry

/-- The number of ways to select 5 doctors including A and B -/
def selection_with_A_and_B : ℕ := sorry

/-- The number of ways to select 5 doctors excluding both A and B -/
def selection_without_A_and_B : ℕ := sorry

/-- The number of ways to select 5 doctors including at least one of A or B -/
def selection_with_A_or_B : ℕ := sorry

/-- The number of ways to select 5 doctors with at least one internal medicine doctor and one surgeon -/
def selection_with_both_specialties : ℕ := sorry

theorem disaster_relief_team_selection :
  selection_with_A_and_B = 816 ∧
  selection_without_A_and_B = 8568 ∧
  selection_with_A_or_B = 6936 ∧
  selection_with_both_specialties = 14656 := by sorry

end disaster_relief_team_selection_l3939_393985


namespace f_range_l3939_393968

-- Define the function f
def f (x : ℝ) : ℝ := 3 * (x - 4)

-- State the theorem
theorem f_range :
  Set.range f = {y : ℝ | y < -27 ∨ y > -27} :=
by
  sorry

end f_range_l3939_393968


namespace ones_digit_of_large_power_l3939_393990

theorem ones_digit_of_large_power : ∃ n : ℕ, 34^(11^34) ≡ 4 [ZMOD 10] :=
  sorry

end ones_digit_of_large_power_l3939_393990


namespace hannah_kids_stockings_l3939_393979

theorem hannah_kids_stockings (total_stuffers : ℕ) 
  (candy_canes_per_kid : ℕ) (beanie_babies_per_kid : ℕ) (books_per_kid : ℕ) :
  total_stuffers = 21 ∧ 
  candy_canes_per_kid = 4 ∧ 
  beanie_babies_per_kid = 2 ∧ 
  books_per_kid = 1 →
  ∃ (num_kids : ℕ), 
    num_kids * (candy_canes_per_kid + beanie_babies_per_kid + books_per_kid) = total_stuffers ∧
    num_kids = 3 := by
  sorry

end hannah_kids_stockings_l3939_393979


namespace prop_2_prop_4_prop_1_counter_prop_3_counter_l3939_393964

-- Define basic geometric concepts
def Line : Type := sorry
def Plane : Type := sorry
def Point : Type := sorry

-- Define geometric relations
def parallel (a b : Plane) : Prop := sorry
def perpendicular (a b : Plane) : Prop := sorry
def perpendicular_line_plane (l : Line) (p : Plane) : Prop := sorry
def line_in_plane (l : Line) (p : Plane) : Prop := sorry
def line_of_intersection (p1 p2 : Plane) : Line := sorry

-- Proposition 2
theorem prop_2 (p1 p2 : Plane) (l : Line) : 
  perpendicular_line_plane l p1 → line_in_plane l p2 → perpendicular p1 p2 := by sorry

-- Proposition 4
theorem prop_4 (p1 p2 : Plane) (l : Line) :
  perpendicular p1 p2 → 
  line_in_plane l p1 → 
  ¬perpendicular_line_plane l (line_of_intersection p1 p2) → 
  ¬perpendicular_line_plane l p2 := by sorry

-- Proposition 1 (counterexample)
theorem prop_1_counter : ∃ (p1 p2 p3 : Plane) (l1 l2 : Line),
  line_in_plane l1 p1 ∧ line_in_plane l2 p1 ∧
  parallel p2 p1 ∧ parallel p3 p1 ∧
  ¬parallel p2 p3 := by sorry

-- Proposition 3 (counterexample)
theorem prop_3_counter : ∃ (l1 l2 l3 : Line),
  perpendicular_line_plane l1 l3 ∧ 
  perpendicular_line_plane l2 l3 ∧
  ¬parallel l1 l2 := by sorry

end prop_2_prop_4_prop_1_counter_prop_3_counter_l3939_393964


namespace zoo_trip_bus_capacity_l3939_393956

/-- Given a school trip to the zoo with the following conditions:
  * total_students: The total number of students on the trip
  * num_buses: The number of buses used for transportation
  * students_in_cars: The number of students who traveled in cars
  * students_per_bus: The number of students in each bus

  This theorem proves that when total_students = 396, num_buses = 7, and students_in_cars = 4,
  the number of students in each bus (students_per_bus) is equal to 56. -/
theorem zoo_trip_bus_capacity 
  (total_students : ℕ) 
  (num_buses : ℕ) 
  (students_in_cars : ℕ) 
  (students_per_bus : ℕ) 
  (h1 : total_students = 396) 
  (h2 : num_buses = 7) 
  (h3 : students_in_cars = 4) 
  (h4 : students_per_bus * num_buses + students_in_cars = total_students) :
  students_per_bus = 56 := by
  sorry

end zoo_trip_bus_capacity_l3939_393956


namespace band_member_earnings_l3939_393951

theorem band_member_earnings 
  (attendees : ℕ) 
  (band_share : ℚ) 
  (ticket_price : ℕ) 
  (band_members : ℕ) 
  (h1 : attendees = 500) 
  (h2 : band_share = 70 / 100) 
  (h3 : ticket_price = 30) 
  (h4 : band_members = 4) : 
  (attendees * ticket_price * band_share) / band_members = 2625 := by
sorry

end band_member_earnings_l3939_393951


namespace emily_max_servings_l3939_393940

/-- Represents the recipe and available ingredients for a fruit smoothie --/
structure SmoothieIngredients where
  recipe_bananas : ℕ
  recipe_strawberries : ℕ
  recipe_yogurt : ℕ
  available_bananas : ℕ
  available_strawberries : ℕ
  available_yogurt : ℕ

/-- Calculates the maximum number of servings that can be made --/
def max_servings (ingredients : SmoothieIngredients) : ℕ :=
  min
    (ingredients.available_bananas * 3 / ingredients.recipe_bananas)
    (min
      (ingredients.available_strawberries * 3 / ingredients.recipe_strawberries)
      (ingredients.available_yogurt * 3 / ingredients.recipe_yogurt))

/-- Theorem stating that Emily can make at most 6 servings --/
theorem emily_max_servings :
  let emily_ingredients : SmoothieIngredients := {
    recipe_bananas := 2,
    recipe_strawberries := 1,
    recipe_yogurt := 2,
    available_bananas := 4,
    available_strawberries := 3,
    available_yogurt := 6
  }
  max_servings emily_ingredients = 6 := by
  sorry

end emily_max_servings_l3939_393940


namespace dividend_percentage_calculation_l3939_393941

/-- Calculates the dividend percentage given investment details --/
theorem dividend_percentage_calculation
  (investment : ℝ)
  (share_face_value : ℝ)
  (premium_rate : ℝ)
  (total_dividend : ℝ)
  (h1 : investment = 14400)
  (h2 : share_face_value = 100)
  (h3 : premium_rate = 0.2)
  (h4 : total_dividend = 600) :
  let share_cost := share_face_value * (1 + premium_rate)
  let num_shares := investment / share_cost
  let dividend_per_share := total_dividend / num_shares
  let dividend_percentage := (dividend_per_share / share_face_value) * 100
  dividend_percentage = 5 := by
sorry


end dividend_percentage_calculation_l3939_393941


namespace circle_radius_square_tangents_l3939_393948

theorem circle_radius_square_tangents (side_length : ℝ) (angle : ℝ) (sin_half_angle : ℝ) :
  side_length = Real.sqrt (2 + Real.sqrt 2) →
  angle = π / 4 →
  sin_half_angle = (Real.sqrt (2 - Real.sqrt 2)) / 2 →
  ∃ (radius : ℝ), radius = Real.sqrt 2 + Real.sqrt (2 - Real.sqrt 2) :=
by sorry

end circle_radius_square_tangents_l3939_393948


namespace strawberry_charity_donation_is_correct_l3939_393938

/-- The amount of money donated to charity from strawberry jam sales -/
def strawberry_charity_donation : ℚ :=
let betty_strawberries : ℕ := 25
let matthew_strawberries : ℕ := betty_strawberries + 30
let natalie_strawberries : ℕ := matthew_strawberries / 3
let emily_strawberries : ℕ := natalie_strawberries / 2
let ethan_strawberries : ℕ := natalie_strawberries * 2
let total_strawberries : ℕ := betty_strawberries + matthew_strawberries + natalie_strawberries + emily_strawberries + ethan_strawberries
let strawberries_per_jar : ℕ := 12
let jars_made : ℕ := total_strawberries / strawberries_per_jar
let price_per_jar : ℚ := 6
let total_revenue : ℚ := (jars_made : ℚ) * price_per_jar
let donation_percentage : ℚ := 40 / 100
donation_percentage * total_revenue

theorem strawberry_charity_donation_is_correct :
  strawberry_charity_donation = 26.4 := by
  sorry

end strawberry_charity_donation_is_correct_l3939_393938


namespace roots_sum_l3939_393929

theorem roots_sum (m₁ m₂ : ℝ) : 
  (∃ a b : ℝ, (m₁ * a^2 - (3 * m₁ - 2) * a + 7 = 0) ∧ 
              (m₁ * b^2 - (3 * m₁ - 2) * b + 7 = 0) ∧ 
              (a / b + b / a = 3 / 2)) ∧
  (∃ a b : ℝ, (m₂ * a^2 - (3 * m₂ - 2) * a + 7 = 0) ∧ 
              (m₂ * b^2 - (3 * m₂ - 2) * b + 7 = 0) ∧ 
              (a / b + b / a = 3 / 2)) →
  m₁ + m₂ = 73 / 18 := by
sorry

end roots_sum_l3939_393929


namespace inequality_range_l3939_393920

theorem inequality_range (a : ℝ) (h : 0 ≤ a ∧ a ≤ 4) :
  ∀ x : ℝ, x^2 + a*x > 4*x + a - 3 ↔ x < -1 ∨ x > 3 := by sorry

end inequality_range_l3939_393920


namespace sum_of_four_numbers_l3939_393921

theorem sum_of_four_numbers : 2468 + 8642 + 6824 + 4286 = 22220 := by
  sorry

end sum_of_four_numbers_l3939_393921


namespace pc_length_l3939_393937

/-- A convex quadrilateral with specific properties -/
structure SpecialQuadrilateral where
  -- Points of the quadrilateral
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ
  -- Point P on diagonal AC
  P : ℝ × ℝ
  -- Convexity condition (simplified)
  convex : True
  -- CD perpendicular to AC
  cd_perp_ac : (C.1 - D.1) * (C.1 - A.1) + (C.2 - D.2) * (C.2 - A.2) = 0
  -- AB perpendicular to BD
  ab_perp_bd : (A.1 - B.1) * (B.1 - D.1) + (A.2 - B.2) * (B.2 - D.2) = 0
  -- CD length
  cd_length : Real.sqrt ((C.1 - D.1)^2 + (C.2 - D.2)^2) = 72
  -- AB length
  ab_length : Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 35
  -- P on AC
  p_on_ac : ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ P = (A.1 + t * (C.1 - A.1), A.2 + t * (C.2 - A.2))
  -- BP perpendicular to AD
  bp_perp_ad : (B.1 - P.1) * (A.1 - D.1) + (B.2 - P.2) * (A.2 - D.2) = 0
  -- AP length
  ap_length : Real.sqrt ((A.1 - P.1)^2 + (A.2 - P.2)^2) = 15

/-- The main theorem stating that PC = 72.5 in the special quadrilateral -/
theorem pc_length (q : SpecialQuadrilateral) : 
  Real.sqrt ((q.P.1 - q.C.1)^2 + (q.P.2 - q.C.2)^2) = 72.5 := by
  sorry

end pc_length_l3939_393937


namespace sqrt_equation_solution_l3939_393901

theorem sqrt_equation_solution :
  ∃! z : ℝ, Real.sqrt (10 + 3 * z) = 8 :=
by sorry

end sqrt_equation_solution_l3939_393901


namespace probability_same_team_l3939_393926

/-- The probability of two volunteers joining the same team out of three teams -/
theorem probability_same_team (num_teams : ℕ) (num_volunteers : ℕ) : 
  num_teams = 3 → num_volunteers = 2 → 
  (num_teams.choose num_volunteers : ℚ) / (num_teams ^ num_volunteers : ℚ) = 1/3 :=
by sorry

end probability_same_team_l3939_393926


namespace ticket_sales_difference_l3939_393959

/-- Proves the difference in ticket sales given ticket prices and total sales -/
theorem ticket_sales_difference (student_price non_student_price : ℕ) 
  (total_sales total_tickets : ℕ) : 
  student_price = 6 →
  non_student_price = 9 →
  total_sales = 10500 →
  total_tickets = 1700 →
  ∃ (student_tickets non_student_tickets : ℕ),
    student_tickets + non_student_tickets = total_tickets ∧
    student_price * student_tickets + non_student_price * non_student_tickets = total_sales ∧
    student_tickets - non_student_tickets = 1500 :=
by sorry

end ticket_sales_difference_l3939_393959


namespace parallel_vectors_angle_l3939_393961

theorem parallel_vectors_angle (α : ℝ) 
  (h_acute : 0 < α ∧ α < π / 2)
  (h_parallel : (3/2, Real.sin α) = (Real.cos α * k, 1/3 * k) → k ≠ 0) :
  α = π / 4 := by
  sorry

end parallel_vectors_angle_l3939_393961


namespace equation_solutions_l3939_393996

theorem equation_solutions : 
  ∀ m n : ℕ, m^2 + 2 * 3^n = m * (2^(n+1) - 1) ↔ 
  (m = 6 ∧ n = 3) ∨ (m = 9 ∧ n = 3) ∨ (m = 9 ∧ n = 5) ∨ (m = 54 ∧ n = 5) := by
  sorry

end equation_solutions_l3939_393996


namespace survey_selection_count_l3939_393925

/-- Represents the total number of households selected in a stratified sampling survey. -/
def total_selected (total_households : ℕ) (middle_income : ℕ) (low_income : ℕ) (high_income_selected : ℕ) : ℕ :=
  (high_income_selected * total_households) / (total_households - middle_income - low_income)

/-- Theorem stating that the total number of households selected in the survey is 24. -/
theorem survey_selection_count :
  total_selected 480 200 160 6 = 24 := by
  sorry

end survey_selection_count_l3939_393925


namespace max_value_of_3a_plus_2_l3939_393995

theorem max_value_of_3a_plus_2 (a : ℝ) (h : 10 * a^2 + 3 * a + 2 = 5) :
  3 * a + 2 ≤ (31 + 3 * Real.sqrt 129) / 20 :=
by sorry

end max_value_of_3a_plus_2_l3939_393995


namespace d_share_is_750_l3939_393909

/-- Represents the share of money for each person -/
structure Share :=
  (amount : ℝ)

/-- Represents the distribution of money among 5 people -/
structure Distribution :=
  (a b c d e : Share)

/-- The total amount of money to be distributed -/
def total_amount (dist : Distribution) : ℝ :=
  dist.a.amount + dist.b.amount + dist.c.amount + dist.d.amount + dist.e.amount

/-- The condition that the distribution follows the proportion 5 : 2 : 4 : 3 : 1 -/
def proportional_distribution (dist : Distribution) : Prop :=
  5 * dist.b.amount = 2 * dist.a.amount ∧
  5 * dist.c.amount = 4 * dist.a.amount ∧
  5 * dist.d.amount = 3 * dist.a.amount ∧
  5 * dist.e.amount = 1 * dist.a.amount

/-- The condition that the combined share of A and C is 3/5 of the total amount -/
def combined_share_condition (dist : Distribution) : Prop :=
  dist.a.amount + dist.c.amount = 3/5 * total_amount dist

/-- The condition that E gets $250 less than B -/
def e_less_than_b_condition (dist : Distribution) : Prop :=
  dist.b.amount - dist.e.amount = 250

theorem d_share_is_750 (dist : Distribution) 
  (h1 : proportional_distribution dist)
  (h2 : combined_share_condition dist)
  (h3 : e_less_than_b_condition dist) :
  dist.d.amount = 750 := by
  sorry

end d_share_is_750_l3939_393909


namespace red_candy_count_l3939_393907

theorem red_candy_count (total : ℕ) (blue : ℕ) (h1 : total = 3409) (h2 : blue = 3264) :
  total - blue = 145 := by
  sorry

end red_candy_count_l3939_393907


namespace parabola_shift_l3939_393902

/-- The original parabola function -/
def original_parabola (x : ℝ) : ℝ := x^2 - 6*x + 5

/-- The shifted parabola function -/
def shifted_parabola (x : ℝ) : ℝ := (x-4)^2 - 2

/-- Theorem stating that the shifted parabola is equivalent to 
    shifting the original parabola 1 unit right and 2 units up -/
theorem parabola_shift :
  ∀ x : ℝ, shifted_parabola x = original_parabola (x - 1) + 2 :=
by sorry

end parabola_shift_l3939_393902
