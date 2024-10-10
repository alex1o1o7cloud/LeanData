import Mathlib

namespace first_week_rate_correct_l578_57829

/-- The daily rate for the first week in a student youth hostel. -/
def first_week_rate : ℝ := 18

/-- The daily rate for days after the first week. -/
def additional_week_rate : ℝ := 12

/-- The total number of days stayed. -/
def total_days : ℕ := 23

/-- The total cost for the stay. -/
def total_cost : ℝ := 318

/-- Theorem stating that the first week rate is correct given the conditions. -/
theorem first_week_rate_correct :
  first_week_rate * 7 + additional_week_rate * (total_days - 7) = total_cost :=
by sorry

end first_week_rate_correct_l578_57829


namespace max_sin_C_in_triangle_l578_57804

theorem max_sin_C_in_triangle (A B C : Real) (h : ∀ A B C, (1 / Real.tan A) + (1 / Real.tan B) = 6 / Real.tan C) :
  ∃ (max_sin_C : Real), max_sin_C = Real.sqrt 15 / 4 ∧ ∀ (sin_C : Real), sin_C ≤ max_sin_C := by
  sorry

end max_sin_C_in_triangle_l578_57804


namespace complex_equation_solution_l578_57824

theorem complex_equation_solution (z : ℂ) : (2 - Complex.I) * z = 4 + 3 * Complex.I → z = 1 + 2 * Complex.I := by
  sorry

end complex_equation_solution_l578_57824


namespace locus_characterization_l578_57835

def has_solution (u v : ℝ) (n : ℕ) : Prop :=
  ∃ x y : ℝ, (Real.sin x)^(2*n) + (Real.cos y)^(2*n) = u ∧ (Real.sin x)^n + (Real.cos y)^n = v

theorem locus_characterization (u v : ℝ) (n : ℕ) :
  has_solution u v n ↔ 
    (v^2 ≤ 2*u ∧ (v - 1)^2 ≥ (u - 1)) ∧
    ((n % 2 = 0 → (0 ≤ v ∧ v ≤ 2 ∧ v^2 ≥ u)) ∧
     (n % 2 = 1 → (-2 ≤ v ∧ v ≤ 2 ∧ (v + 1)^2 ≥ (u - 1)))) :=
by sorry

end locus_characterization_l578_57835


namespace age_ratio_problem_l578_57839

theorem age_ratio_problem (sam sue kendra : ℕ) : 
  kendra = 3 * sam →
  kendra = 18 →
  (sam + 3) + (sue + 3) + (kendra + 3) = 36 →
  sam / sue = 2 := by
sorry

end age_ratio_problem_l578_57839


namespace parabola_intersection_ratio_l578_57813

/-- Given a parabola y = 2p(x - a) where a > 0, and a line y = kx passing through the origin
    (k ≠ 0) intersecting the parabola at two points, the ratio of the sum of x-coordinates
    to the product of x-coordinates of these intersection points is equal to 1/a. -/
theorem parabola_intersection_ratio (p a k : ℝ) (ha : a > 0) (hk : k ≠ 0) :
  let f : ℝ → ℝ := λ x ↦ 2 * p * (x - a)
  let g : ℝ → ℝ := λ x ↦ k * x
  let roots := {x : ℝ | f x = g x}
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁ ∈ roots ∧ x₂ ∈ roots ∧ (x₁ + x₂) / (x₁ * x₂) = 1 / a :=
by
  sorry


end parabola_intersection_ratio_l578_57813


namespace cone_volume_from_cylinder_l578_57885

/-- Given a cylinder with volume 72π cm³, a cone with the same height and twice the radius
    of the cylinder has a volume of 96π cm³. -/
theorem cone_volume_from_cylinder (r h : ℝ) : 
  r > 0 → h > 0 → π * r^2 * h = 72 * π → 
  (1/3 : ℝ) * π * (2*r)^2 * h = 96 * π := by
sorry

end cone_volume_from_cylinder_l578_57885


namespace multiply_three_point_six_by_zero_point_twenty_five_l578_57826

theorem multiply_three_point_six_by_zero_point_twenty_five :
  3.6 * 0.25 = 0.9 := by
  sorry

end multiply_three_point_six_by_zero_point_twenty_five_l578_57826


namespace driveway_snow_volume_l578_57866

/-- The volume of snow on a driveway -/
def snow_volume (length width depth : ℝ) : ℝ := length * width * depth

/-- Theorem: The volume of snow on a driveway with length 30 feet, width 3 feet, 
    and snow depth 0.75 feet is equal to 67.5 cubic feet -/
theorem driveway_snow_volume :
  snow_volume 30 3 0.75 = 67.5 := by
  sorry

end driveway_snow_volume_l578_57866


namespace factorial_300_trailing_zeros_l578_57859

/-- The number of trailing zeros in n! -/
def trailingZeros (n : ℕ) : ℕ := sorry

/-- Theorem: The number of trailing zeros in 300! is 74 -/
theorem factorial_300_trailing_zeros :
  trailingZeros 300 = 74 := by sorry

end factorial_300_trailing_zeros_l578_57859


namespace largest_multiple_of_8_under_100_l578_57842

theorem largest_multiple_of_8_under_100 : 
  ∃ n : ℕ, n * 8 = 96 ∧ 
    ∀ m : ℕ, m * 8 < 100 → m * 8 ≤ 96 := by
  sorry

end largest_multiple_of_8_under_100_l578_57842


namespace parabola_points_order_l578_57849

theorem parabola_points_order : 
  let y₁ : ℝ := -1/2 * (-2)^2 + 2 * (-2)
  let y₂ : ℝ := -1/2 * (-1)^2 + 2 * (-1)
  let y₃ : ℝ := -1/2 * 8^2 + 2 * 8
  y₃ < y₁ ∧ y₁ < y₂ := by sorry

end parabola_points_order_l578_57849


namespace absolute_value_sum_inequality_l578_57860

theorem absolute_value_sum_inequality (b : ℝ) :
  (∃ x : ℝ, |x - 5| + |x - 10| > b) ↔ (0 < b ∧ b < 5) := by
  sorry

end absolute_value_sum_inequality_l578_57860


namespace negative_squared_times_squared_l578_57817

theorem negative_squared_times_squared (a : ℝ) : -a^2 * a^2 = -a^4 := by
  sorry

end negative_squared_times_squared_l578_57817


namespace alonzo_unsold_tomatoes_l578_57868

/-- Calculates the amount of unsold tomatoes given the total harvest and amounts sold to two buyers. -/
def unsold_tomatoes (total_harvest : ℝ) (sold_to_maxwell : ℝ) (sold_to_wilson : ℝ) : ℝ :=
  total_harvest - (sold_to_maxwell + sold_to_wilson)

/-- Proves that given the specific amounts in Mr. Alonzo's tomato sales, the unsold amount is 42 kg. -/
theorem alonzo_unsold_tomatoes :
  unsold_tomatoes 245.5 125.5 78 = 42 := by
  sorry

end alonzo_unsold_tomatoes_l578_57868


namespace tea_mixture_price_l578_57840

/-- Given three varieties of tea mixed in a 1:1:2 ratio, with the first two varieties
    costing 126 and 135 rupees per kg respectively, and the mixture worth 152 rupees per kg,
    prove that the third variety costs 173.5 rupees per kg. -/
theorem tea_mixture_price (price1 price2 mixture_price : ℚ) 
    (h1 : price1 = 126)
    (h2 : price2 = 135)
    (h3 : mixture_price = 152) : ∃ price3 : ℚ,
  price3 = 173.5 ∧ 
  (price1 + price2 + 2 * price3) / 4 = mixture_price :=
by
  sorry

end tea_mixture_price_l578_57840


namespace speed_conversion_l578_57827

-- Define the conversion factor from m/s to km/h
def meters_per_second_to_kmph : ℝ := 3.6

-- Define the given speed in meters per second
def speed_ms : ℝ := 200.016

-- Define the speed in km/h that we want to prove
def speed_kmph : ℝ := 720.0576

-- Theorem statement
theorem speed_conversion :
  speed_ms * meters_per_second_to_kmph = speed_kmph :=
by
  sorry

end speed_conversion_l578_57827


namespace cube_face_sum_l578_57881

/-- Represents the numbers on the faces of a cube -/
structure CubeNumbers where
  a : ℕ+
  b : ℕ+
  c : ℕ+
  e : ℕ+
  f : ℕ+
  g : ℕ+
  h : ℕ+
  vertex_sum_eq : (a + e) * (b + f) * (c + g) * h = 2002

/-- The sum of numbers on the faces of the cube is 39 -/
theorem cube_face_sum (cube : CubeNumbers) : 
  cube.a + cube.b + cube.c + cube.e + cube.f + cube.g + cube.h = 39 := by
  sorry


end cube_face_sum_l578_57881


namespace square_difference_l578_57898

theorem square_difference (x y : ℝ) 
  (h1 : (x + y) / 2 = 5)
  (h2 : (x - y) / 2 = 2) : 
  x^2 - y^2 = 40 := by
sorry

end square_difference_l578_57898


namespace equation_solution_l578_57845

theorem equation_solution : 
  ∀ x : ℝ, (x - 5)^2 = (1/16)⁻¹ ↔ x = 1 ∨ x = 9 := by
sorry

end equation_solution_l578_57845


namespace triangle_shape_l578_57883

theorem triangle_shape (a b c : ℝ) (A B C : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c)
  (h4 : 0 < A) (h5 : 0 < B) (h6 : 0 < C) 
  (h7 : A + B + C = Real.pi)
  (h8 : 2 * a * Real.cos B = c)
  (h9 : a * Real.sin B = b * Real.sin A)
  (h10 : b * Real.sin C = c * Real.sin B)
  (h11 : c * Real.sin A = a * Real.sin C) :
  A = B ∨ B = C ∨ A = C :=
sorry

end triangle_shape_l578_57883


namespace min_value_of_a_min_value_is_one_l578_57896

theorem min_value_of_a (a : ℝ) : 
  (∀ x : ℝ, (2 * x) / (x^2 + 1) ≤ a) → a ≥ 1 := by
  sorry

theorem min_value_is_one :
  ∃ a : ℝ, (∀ x : ℝ, (2 * x) / (x^2 + 1) ≤ a) ∧ 
    (∀ b : ℝ, (∀ x : ℝ, (2 * x) / (x^2 + 1) ≤ b) → a ≤ b) := by
  sorry

end min_value_of_a_min_value_is_one_l578_57896


namespace quadratic_equation_roots_l578_57812

theorem quadratic_equation_roots (a b c : ℝ) (h : a ≠ 0) :
  (b^2 - 4*a*c > 0 → ∃ x : ℝ, a*x^2 + b*x + c = 0) ∧
  (∃ b c : ℝ, (∃ x : ℝ, a*x^2 + b*x + c = 0) ∧ ¬(b^2 - 4*a*c > 0)) :=
sorry

end quadratic_equation_roots_l578_57812


namespace function_extrema_and_inequality_l578_57856

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * Real.log x - 0.5 * x^2 + x

def g (x : ℝ) : ℝ := 0.5 * x^2 - 2 * x + 1

theorem function_extrema_and_inequality (e : ℝ) (h_e : e = Real.exp 1) :
  (∃ (max min : ℝ), 
    (∀ x ∈ Set.Icc 1 (e^2), f 2 x ≤ max) ∧ 
    (∃ x₀ ∈ Set.Icc 1 (e^2), f 2 x₀ = max) ∧
    (∀ x ∈ Set.Icc 1 (e^2), min ≤ f 2 x) ∧ 
    (∃ x₁ ∈ Set.Icc 1 (e^2), f 2 x₁ = min) ∧
    max = 2 * Real.log 2 ∧
    min = 4 + e^2 - 0.5 * e^4) ∧
  (∀ a : ℝ, (∀ x > 0, f a x + g x ≤ 0) ↔ a = 1) :=
sorry

end function_extrema_and_inequality_l578_57856


namespace inequality_proof_l578_57846

theorem inequality_proof (a b : ℝ) (n : ℕ) (x₁ y₁ x₂ y₂ A : ℝ) :
  a > 0 →
  b > 0 →
  n > 1 →
  x₁ > 0 →
  y₁ > 0 →
  x₂ > 0 →
  y₂ > 0 →
  x₁^n - a*y₁^n = b →
  x₂^n - a*y₂^n = b →
  y₁ < y₂ →
  A = (1/2) * |x₁*y₂ - x₂*y₁| →
  b*y₂ > 2*n*y₁^(n-1)*a^(1-1/n)*A :=
by sorry

end inequality_proof_l578_57846


namespace sally_picked_42_peaches_l578_57864

/-- The number of peaches Sally picked -/
def peaches_picked (initial : ℕ) (final : ℕ) : ℕ := final - initial

/-- Theorem: Sally picked 42 peaches -/
theorem sally_picked_42_peaches : peaches_picked 13 55 = 42 := by
  sorry

end sally_picked_42_peaches_l578_57864


namespace cylinder_height_in_hemisphere_l578_57872

theorem cylinder_height_in_hemisphere (r c h : ℝ) : 
  r > 0 → c > 0 → h > 0 →
  r = 7 → c = 3 →
  h^2 + c^2 = r^2 →
  h = 2 * Real.sqrt 10 :=
by sorry

end cylinder_height_in_hemisphere_l578_57872


namespace kelly_baking_powder_l578_57889

def yesterday_amount : ℝ := 0.4
def difference : ℝ := 0.1

theorem kelly_baking_powder :
  let current_amount := yesterday_amount - difference
  current_amount = 0.3 := by
sorry

end kelly_baking_powder_l578_57889


namespace total_spending_l578_57850

/-- The amount Ben spends -/
def ben_spent : ℝ := 50

/-- The amount David spends -/
def david_spent : ℝ := 37.5

/-- The difference in spending between Ben and David -/
def spending_difference : ℝ := 12.5

/-- The difference in cost per item between Ben and David -/
def cost_difference_per_item : ℝ := 0.25

theorem total_spending :
  ben_spent + david_spent = 87.5 ∧
  ben_spent - david_spent = spending_difference ∧
  ben_spent / david_spent = 4 / 3 :=
by sorry

end total_spending_l578_57850


namespace trig_identity_l578_57894

theorem trig_identity (α : Real) (h : Real.sin (π / 8 + α) = 3 / 4) :
  Real.cos (3 * π / 8 - α) = 3 / 4 := by
  sorry

end trig_identity_l578_57894


namespace modular_arithmetic_proof_l578_57830

theorem modular_arithmetic_proof : (305 * 20 - 20 * 9 + 5) % 19 = 16 := by
  sorry

end modular_arithmetic_proof_l578_57830


namespace problem_2019_1981_l578_57814

theorem problem_2019_1981 : (2019 + 1981)^2 / 121 = 132231 := by
  sorry

end problem_2019_1981_l578_57814


namespace decimal_comparisons_l578_57893

theorem decimal_comparisons :
  (9.38 > 3.98) ∧
  (0.62 > 0.23) ∧
  (2.5 > 2.05) ∧
  (53.6 > 5.36) ∧
  (9.42 > 9.377) := by
  sorry

end decimal_comparisons_l578_57893


namespace c_investment_is_81000_l578_57836

/-- Calculates the investment of partner C in a partnership business -/
def calculate_c_investment (a_investment b_investment : ℕ) (total_profit c_profit : ℕ) : ℕ :=
  let total_investment_ab := a_investment + b_investment
  let c_investment := (c_profit * (total_investment_ab + c_profit * total_investment_ab / (total_profit - c_profit))) / total_profit
  c_investment

/-- Theorem: Given the specific investments and profits, C's investment is 81000 -/
theorem c_investment_is_81000 :
  calculate_c_investment 27000 72000 80000 36000 = 81000 := by
  sorry

end c_investment_is_81000_l578_57836


namespace perpendicular_vectors_second_component_l578_57862

/-- Given two 2D vectors a and b, if they are perpendicular, then the second component of b is 2. -/
theorem perpendicular_vectors_second_component (a b : ℝ × ℝ) :
  a = (1, 2) →
  b.1 = -4 →
  a.1 * b.1 + a.2 * b.2 = 0 →
  b.2 = 2 := by
sorry

end perpendicular_vectors_second_component_l578_57862


namespace monochromatic_triangle_exists_l578_57844

-- Define a type for colors
inductive Color
| Red
| Blue

-- Define a function type for edge coloring
def EdgeColoring := Fin 6 → Fin 6 → Color

-- Main theorem
theorem monochromatic_triangle_exists (coloring : EdgeColoring) : 
  ∃ (a b c : Fin 6), a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ 
  ((coloring a b = coloring b c ∧ coloring b c = coloring a c) ∨
   (coloring a b = Color.Red ∧ coloring b c = Color.Red ∧ coloring a c = Color.Red) ∨
   (coloring a b = Color.Blue ∧ coloring b c = Color.Blue ∧ coloring a c = Color.Blue)) :=
sorry

end monochromatic_triangle_exists_l578_57844


namespace lending_time_combined_l578_57875

-- Define the lending time for chocolate bars and bonbons
def lending_time_chocolate (bars : ℚ) : ℚ := (3 / 2) * bars

def lending_time_bonbons (bonbons : ℚ) : ℚ := (1 / 6) * bonbons

-- Theorem to prove
theorem lending_time_combined : 
  lending_time_chocolate 1 + lending_time_bonbons 3 = 2 := by
  sorry

end lending_time_combined_l578_57875


namespace correct_factorization_l578_57882

theorem correct_factorization (x : ℝ) : 1 - 2*x + x^2 = (1 - x)^2 := by
  sorry

end correct_factorization_l578_57882


namespace arithmetic_sequence_sum_l578_57890

theorem arithmetic_sequence_sum (a : ℕ → ℤ) :
  (∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d) →  -- arithmetic sequence condition
  a 1 + a 4 + a 7 = 45 →                     -- first given condition
  a 2 + a 5 + a 8 = 29 →                     -- second given condition
  a 3 + a 6 + a 9 = 13 :=                    -- conclusion to prove
by
  sorry

end arithmetic_sequence_sum_l578_57890


namespace ruffy_is_nine_l578_57886

/-- Ruffy's current age -/
def ruffy_age : ℕ := 9

/-- Orlie's current age -/
def orlie_age : ℕ := 12

/-- Relation between Ruffy's and Orlie's current ages -/
axiom current_age_relation : ruffy_age = (3 * orlie_age) / 4

/-- Relation between Ruffy's and Orlie's ages four years ago -/
axiom past_age_relation : ruffy_age - 4 = (orlie_age - 4) / 2 + 1

/-- Theorem: Ruffy's current age is 9 years -/
theorem ruffy_is_nine : ruffy_age = 9 := by sorry

end ruffy_is_nine_l578_57886


namespace equation_solution_l578_57841

theorem equation_solution : ∃! x : ℝ, (x^2 + 2*x + 3) / (x + 2) = x + 3 := by
  use -1
  constructor
  · -- Prove that x = -1 satisfies the equation
    sorry
  · -- Prove uniqueness
    sorry

#check equation_solution

end equation_solution_l578_57841


namespace sqrt_190_44_sqrt_176_9_and_18769_integer_between_sqrt_l578_57897

-- Define the square root function
noncomputable def sqrt (x : ℝ) : ℝ := Real.sqrt x

-- Theorem 1
theorem sqrt_190_44 : sqrt 190.44 = 13.8 ∨ sqrt 190.44 = -13.8 := by sorry

-- Theorem 2
theorem sqrt_176_9_and_18769 :
  (13.3 < sqrt 176.9 ∧ sqrt 176.9 < 13.4) ∧ sqrt 18769 = 137 := by sorry

-- Theorem 3
theorem integer_between_sqrt :
  ∀ n : ℤ, (13.5 < sqrt (n : ℝ) ∧ sqrt (n : ℝ) < 13.6) → (n = 183 ∨ n = 184) := by sorry

end sqrt_190_44_sqrt_176_9_and_18769_integer_between_sqrt_l578_57897


namespace sum_reciprocal_F_powers_of_two_converges_to_one_l578_57819

/-- Definition of the sequence F -/
def F : ℕ → ℚ
  | 0 => 1
  | 1 => 2
  | (n+2) => (3/2) * F (n+1) - (1/2) * F n

/-- The sum of the reciprocals of F(2^n) converges to 1 -/
theorem sum_reciprocal_F_powers_of_two_converges_to_one :
  ∑' n, (1 : ℝ) / F (2^n) = 1 := by sorry

end sum_reciprocal_F_powers_of_two_converges_to_one_l578_57819


namespace siblings_height_l578_57861

/-- The total height of 5 siblings -/
def total_height (h1 h2 h3 h4 h5 : ℕ) : ℕ := h1 + h2 + h3 + h4 + h5

/-- Theorem stating the total height of the 5 siblings is 330 inches -/
theorem siblings_height :
  ∃ (h5 : ℕ), 
    total_height 66 66 60 68 h5 = 330 ∧ h5 = 68 + 2 := by
  sorry

end siblings_height_l578_57861


namespace quadratic_inequality_solution_set_l578_57891

theorem quadratic_inequality_solution_set (x : ℝ) : 
  x^2 - 2*x ≤ 0 ↔ 0 ≤ x ∧ x ≤ 2 :=
by sorry

end quadratic_inequality_solution_set_l578_57891


namespace min_abc_value_l578_57810

-- Define the set M
def M : Set ℝ := {x | 2/3 < x ∧ x < 2}

-- Define t as the largest positive integer in M
def t : ℕ := 1

-- Theorem statement
theorem min_abc_value (a b c : ℝ) 
  (ha : a > 1) (hb : b > 1) (hc : c > 1)
  (h_abc : (a - 1) * (b - 1) * (c - 1) = t) :
  a * b * c ≥ 8 :=
sorry

end min_abc_value_l578_57810


namespace vector_equality_l578_57863

def A : ℝ × ℝ := (1, 3)
def B : ℝ × ℝ := (2, 4)

def a (x : ℝ) : ℝ × ℝ := (2*x - 1, x^2 + 3*x - 3)

theorem vector_equality (x : ℝ) : a x = (B.1 - A.1, B.2 - A.2) → x = 1 := by
  sorry

end vector_equality_l578_57863


namespace mimi_picked_24_shells_l578_57815

/-- The number of seashells Mimi picked up -/
def mimi_shells : ℕ := 24

/-- The number of seashells Kyle found -/
def kyle_shells : ℕ := 2 * mimi_shells

/-- The number of seashells Leigh grabbed -/
def leigh_shells : ℕ := 16

/-- Theorem stating that Mimi picked up 24 seashells -/
theorem mimi_picked_24_shells : mimi_shells = 24 :=
by
  have h1 : kyle_shells = 2 * mimi_shells := by rfl
  have h2 : leigh_shells = kyle_shells / 3 := by sorry
  have h3 : leigh_shells = 16 := by rfl
  sorry


end mimi_picked_24_shells_l578_57815


namespace only_happiness_symmetrical_l578_57816

-- Define a type for Chinese characters
inductive ChineseCharacter : Type
| happiness : ChineseCharacter  -- 喜
| longevity : ChineseCharacter  -- 寿
| blessing : ChineseCharacter   -- 福
| prosperity : ChineseCharacter -- 禄

-- Define symmetry for Chinese characters
def isSymmetrical (c : ChineseCharacter) : Prop :=
  match c with
  | ChineseCharacter.happiness => true
  | _ => false

-- Theorem statement
theorem only_happiness_symmetrical :
  ∀ c : ChineseCharacter, isSymmetrical c ↔ c = ChineseCharacter.happiness :=
by sorry

end only_happiness_symmetrical_l578_57816


namespace at_least_one_third_l578_57857

theorem at_least_one_third (a b c : ℝ) (h : a + b + c = 1) :
  (a ≥ 1/3) ∨ (b ≥ 1/3) ∨ (c ≥ 1/3) := by
  sorry

end at_least_one_third_l578_57857


namespace xyz_sum_sqrt_l578_57884

theorem xyz_sum_sqrt (x y z : ℝ) 
  (eq1 : y + z = 15)
  (eq2 : z + x = 17)
  (eq3 : x + y = 16) :
  Real.sqrt (x * y * z * (x + y + z)) = 72 * Real.sqrt 7 := by
  sorry

end xyz_sum_sqrt_l578_57884


namespace union_condition_intersection_condition_l578_57800

-- Define sets A and B
def A : Set ℝ := {x : ℝ | -2 < x ∧ x < 5}
def B (m : ℝ) : Set ℝ := {x : ℝ | m < x ∧ x < m + 3}

-- Theorem for part (I)
theorem union_condition (m : ℝ) :
  A ∪ B m = A ↔ m ∈ Set.Icc (-2) 2 := by sorry

-- Theorem for part (II)
theorem intersection_condition (m : ℝ) :
  (A ∩ B m).Nonempty ↔ m ∈ Set.Ioo (-5) 2 := by sorry

end union_condition_intersection_condition_l578_57800


namespace sum_remainders_divisible_by_500_l578_57834

/-- The set of all possible remainders when 3^n (n is a nonnegative integer) is divided by 500 -/
def R : Finset ℕ :=
  sorry

/-- The sum of all elements in R -/
def S : ℕ := sorry

/-- Theorem: The sum of all distinct remainders when 3^n (n is a nonnegative integer) 
    is divided by 500 is divisible by 500 -/
theorem sum_remainders_divisible_by_500 : 500 ∣ S := by
  sorry

end sum_remainders_divisible_by_500_l578_57834


namespace age_ratio_in_two_years_l578_57888

/-- Pete's current age -/
def p : ℕ := sorry

/-- Mandy's current age -/
def m : ℕ := sorry

/-- The number of years until the ratio of their ages is 3:2 -/
def x : ℕ := sorry

/-- Pete's age two years ago was twice Mandy's age two years ago -/
axiom past_condition_1 : p - 2 = 2 * (m - 2)

/-- Pete's age four years ago was three times Mandy's age four years ago -/
axiom past_condition_2 : p - 4 = 3 * (m - 4)

/-- The ratio of their ages will be 3:2 after x years -/
axiom future_ratio : (p + x) / (m + x) = 3 / 2

theorem age_ratio_in_two_years :
  x = 2 :=
sorry

end age_ratio_in_two_years_l578_57888


namespace sqrt_450_simplification_l578_57808

theorem sqrt_450_simplification : Real.sqrt 450 = 15 * Real.sqrt 2 := by
  sorry

end sqrt_450_simplification_l578_57808


namespace race_length_proof_l578_57801

/-- The length of a race where runner A beats runner B by 35 meters in 7 seconds,
    and A's time over the course is 33 seconds. -/
def race_length : ℝ := 910

theorem race_length_proof :
  let time_A : ℝ := 33
  let lead_distance : ℝ := 35
  let lead_time : ℝ := 7
  race_length = (lead_distance * time_A) / (lead_time / time_A) := by
  sorry

#check race_length_proof

end race_length_proof_l578_57801


namespace unique_solution_for_n_equals_one_l578_57876

theorem unique_solution_for_n_equals_one (n : ℕ+) :
  (∃ x : ℤ, x^(n : ℕ) + (2 + x)^(n : ℕ) + (2 - x)^(n : ℕ) = 0) ↔ n = 1 := by
  sorry

end unique_solution_for_n_equals_one_l578_57876


namespace digit_721_of_3_over_11_l578_57865

theorem digit_721_of_3_over_11 : ∃ (d : ℕ), d = 2 ∧ 
  (∃ (seq : ℕ → ℕ), 
    (∀ n, seq n < 10) ∧ 
    (∀ n, (3 * 10^(n+1)) % 11 = seq n) ∧
    seq 720 = d) := by
  sorry

end digit_721_of_3_over_11_l578_57865


namespace domain_shift_l578_57874

-- Define the function f
def f : ℝ → ℝ := sorry

-- Define the domain of f
def domain_f : Set ℝ := Set.Icc 1 4

-- State the theorem
theorem domain_shift :
  (∀ x, f x ≠ 0 → x ∈ domain_f) →
  (∀ x, f (x - 1) ≠ 0 → x ∈ Set.Icc 2 5) :=
sorry

end domain_shift_l578_57874


namespace magnitude_of_complex_square_root_l578_57871

theorem magnitude_of_complex_square_root (w : ℂ) (h : w^2 = 48 - 14*I) : 
  Complex.abs w = 5 * Real.sqrt 2 := by
sorry

end magnitude_of_complex_square_root_l578_57871


namespace fraction_sum_squared_l578_57838

theorem fraction_sum_squared (a b c x y z : ℝ) 
  (h1 : x / a + y / b + z / c = 4)
  (h2 : a / x + b / y + c / z = 0)
  : x^2 / a^2 + y^2 / b^2 + z^2 / c^2 = 16 := by
  sorry

end fraction_sum_squared_l578_57838


namespace sequence_not_contains_square_l578_57873

def a : ℕ → ℚ
  | 0 => 2016
  | n + 1 => a n + 2 / (a n)

theorem sequence_not_contains_square : ∀ n : ℕ, ¬ ∃ q : ℚ, a n = q^2 := by
  sorry

end sequence_not_contains_square_l578_57873


namespace perimeter_of_right_triangle_with_circles_l578_57809

/-- A right triangle with inscribed circles -/
structure RightTriangleWithCircles where
  -- The side lengths of the triangle
  a : ℝ
  b : ℝ
  c : ℝ
  -- The radius of the inscribed circles
  r : ℝ
  -- Conditions
  right_triangle : a^2 + b^2 = c^2
  isosceles : a = b
  circle_radius : r = 2
  -- Relationship between side lengths and circle radius
  side_circle_relation : a = 4 * r

/-- The perimeter of a right triangle with inscribed circles -/
def perimeter (t : RightTriangleWithCircles) : ℝ :=
  t.a + t.b + t.c

/-- Theorem: The perimeter of the specified right triangle with inscribed circles is 16 + 8√2 -/
theorem perimeter_of_right_triangle_with_circles (t : RightTriangleWithCircles) :
  perimeter t = 16 + 8 * Real.sqrt 2 := by
  sorry


end perimeter_of_right_triangle_with_circles_l578_57809


namespace rectangle_perimeter_l578_57899

/-- Given a square with perimeter 160 units divided into 4 congruent rectangles,
    prove that the perimeter of one rectangle is 120 units. -/
theorem rectangle_perimeter (square_perimeter : ℝ) (h1 : square_perimeter = 160) :
  let square_side : ℝ := square_perimeter / 4
  let rect_width : ℝ := square_side / 2
  let rect_height : ℝ := square_side
  let rect_perimeter : ℝ := 2 * (rect_width + rect_height)
  rect_perimeter = 120 := by sorry

end rectangle_perimeter_l578_57899


namespace quadratic_roots_sum_reciprocals_l578_57821

theorem quadratic_roots_sum_reciprocals (a b : ℝ) 
  (ha : a^2 + a - 1 = 0) (hb : b^2 + b - 1 = 0) : 
  a/b + b/a = 2 ∨ a/b + b/a = -3 := by
sorry

end quadratic_roots_sum_reciprocals_l578_57821


namespace water_consumption_l578_57820

theorem water_consumption (yesterday_amount : ℝ) (percentage_decrease : ℝ) 
  (h1 : yesterday_amount = 48)
  (h2 : percentage_decrease = 4)
  (h3 : yesterday_amount = (100 - percentage_decrease) / 100 * two_days_ago_amount) :
  two_days_ago_amount = 50 :=
by
  sorry

end water_consumption_l578_57820


namespace range_of_a_l578_57843

-- Define the propositions p and q as functions of a
def p (a : ℝ) : Prop := ∀ x y : ℝ, x < y → (2*a - 1)^x < (2*a - 1)^y

def q (a : ℝ) : Prop := ∀ x : ℝ, 2*a*x^2 - 2*a*x + 1 > 0

-- Define the range of a
def range_a (a : ℝ) : Prop := (0 ≤ a ∧ a ≤ 1) ∨ (a ≥ 2)

-- State the theorem
theorem range_of_a (a : ℝ) : 
  (p a ∨ q a) ∧ ¬(p a ∧ q a) → range_a a :=
by sorry

end range_of_a_l578_57843


namespace calculate_expression_l578_57822

theorem calculate_expression : 5 * 405 + 4 * 405 - 3 * 405 + 404 = 2834 := by
  sorry

end calculate_expression_l578_57822


namespace inequality_proof_l578_57825

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (h_sum : a * b + b * c + c * a = 1) : 
  (a / Real.sqrt (a^2 + 1)) + (b / Real.sqrt (b^2 + 1)) + (c / Real.sqrt (c^2 + 1)) ≤ 3/2 := by
  sorry

end inequality_proof_l578_57825


namespace B_largest_at_45_l578_57895

/-- B_k is defined as the binomial coefficient (500 choose k) multiplied by 0.1^k -/
def B (k : ℕ) : ℝ := (Nat.choose 500 k : ℝ) * (0.1 ^ k)

/-- Theorem stating that B_k is largest when k = 45 -/
theorem B_largest_at_45 : ∀ k : ℕ, k ≤ 500 → B 45 ≥ B k := by
  sorry

end B_largest_at_45_l578_57895


namespace third_term_of_geometric_series_l578_57848

/-- Given an infinite geometric series with common ratio 1/4 and sum 16,
    the third term of the series is 3/4. -/
theorem third_term_of_geometric_series :
  ∀ (a : ℝ),
  (a / (1 - (1/4 : ℝ)) = 16) →  -- Sum of infinite geometric series
  (a * (1/4 : ℝ)^2 = 3/4) :=    -- Third term of the series
by sorry

end third_term_of_geometric_series_l578_57848


namespace find_other_number_l578_57833

theorem find_other_number (A B : ℕ+) 
  (h_lcm : Nat.lcm A B = 2310)
  (h_gcd : Nat.gcd A B = 30)
  (h_A : A = 770) : 
  B = 90 := by
sorry

end find_other_number_l578_57833


namespace division_remainder_l578_57802

theorem division_remainder : ∃ q : ℤ, 3021 = 97 * q + 14 ∧ 0 ≤ 14 ∧ 14 < 97 := by
  sorry

end division_remainder_l578_57802


namespace sqrt_eight_minus_sqrt_two_equals_sqrt_two_l578_57823

theorem sqrt_eight_minus_sqrt_two_equals_sqrt_two : 
  Real.sqrt 8 - Real.sqrt 2 = Real.sqrt 2 := by
  sorry

end sqrt_eight_minus_sqrt_two_equals_sqrt_two_l578_57823


namespace max_value_theorem_max_value_achievable_l578_57854

theorem max_value_theorem (x y : ℝ) :
  (2 * x + 3 * y + 4) / Real.sqrt (x^2 + y^2 + 1) ≤ Real.sqrt 29 :=
by sorry

theorem max_value_achievable :
  ∃ x y : ℝ, (2 * x + 3 * y + 4) / Real.sqrt (x^2 + y^2 + 1) = Real.sqrt 29 :=
by sorry

end max_value_theorem_max_value_achievable_l578_57854


namespace hexagon_minus_sectors_area_l578_57818

/-- The area of the region inside a regular hexagon but outside circular sectors --/
theorem hexagon_minus_sectors_area (s : ℝ) (r : ℝ) (θ : ℝ) : 
  s = 10 → r = 5 → θ = 120 → 
  (6 * (s^2 * Real.sqrt 3 / 4)) - (6 * (θ / 360) * Real.pi * r^2) = 150 * Real.sqrt 3 - 50 * Real.pi :=
by sorry

end hexagon_minus_sectors_area_l578_57818


namespace right_triangle_hypotenuse_l578_57867

theorem right_triangle_hypotenuse (a b c : ℝ) : 
  a = 3 ∧ b = 4 ∧ c^2 = a^2 + b^2 ∨ (a = 3 ∧ b = 4 ∧ c = b) → c = 5 ∨ c = 4 := by
  sorry

end right_triangle_hypotenuse_l578_57867


namespace complex_absolute_value_product_l578_57870

theorem complex_absolute_value_product : 
  ∃ (z w : ℂ), z = 3 * Real.sqrt 5 - 5 * I ∧ w = 2 * Real.sqrt 2 + 4 * I ∧ 
  Complex.abs (z * w) = 8 * Real.sqrt 105 := by
  sorry

end complex_absolute_value_product_l578_57870


namespace f_of_two_eq_two_l578_57869

/-- A function satisfying the given functional equation -/
def FunctionalEquation (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f x + 3 * f (8 - x) = x

/-- Theorem stating that for any function satisfying the functional equation, f(2) = 2 -/
theorem f_of_two_eq_two (f : ℝ → ℝ) (h : FunctionalEquation f) : f 2 = 2 := by
  sorry

#check f_of_two_eq_two

end f_of_two_eq_two_l578_57869


namespace ninth_day_practice_correct_l578_57880

/-- The number of minutes Jenna practices piano on the 9th day to achieve
    an average of 100 minutes per day over a 9-day period, given her
    practice times for the first 8 days. -/
def ninth_day_practice (days_type1 days_type2 : ℕ) 
                       (minutes_type1 minutes_type2 : ℕ) : ℕ :=
  let total_days := days_type1 + days_type2 + 1
  let target_total := total_days * 100
  let current_total := days_type1 * minutes_type1 + days_type2 * minutes_type2
  target_total - current_total

theorem ninth_day_practice_correct :
  ninth_day_practice 6 2 80 105 = 210 :=
by sorry

end ninth_day_practice_correct_l578_57880


namespace eliot_account_balance_l578_57828

theorem eliot_account_balance 
  (al_balance : ℝ) 
  (eliot_balance : ℝ) 
  (al_more : al_balance > eliot_balance)
  (difference_sum : al_balance - eliot_balance = (1 / 12) * (al_balance + eliot_balance))
  (increased_difference : 1.1 * al_balance = 1.2 * eliot_balance + 20) :
  eliot_balance = 200 := by
sorry

end eliot_account_balance_l578_57828


namespace tangent_line_at_one_l578_57878

noncomputable def f (x : ℝ) : ℝ := 2 * Real.log x + x^2

theorem tangent_line_at_one :
  ∃ (m b : ℝ), ∀ (x y : ℝ),
    (y = m * (x - 1) + f 1) ↔ (y = m * x + b ∧ 4 * x - y - 3 = 0) :=
by sorry

end tangent_line_at_one_l578_57878


namespace oliver_workout_ratio_l578_57832

/-- Oliver's workout schedule problem -/
theorem oliver_workout_ratio :
  let monday : ℕ := 4  -- Monday's workout hours
  let tuesday : ℕ := monday - 2  -- Tuesday's workout hours
  let thursday : ℕ := 2 * tuesday  -- Thursday's workout hours
  let total : ℕ := 18  -- Total workout hours over four days
  let wednesday : ℕ := total - (monday + tuesday + thursday)  -- Wednesday's workout hours
  (wednesday : ℚ) / monday = 2 := by
  sorry

end oliver_workout_ratio_l578_57832


namespace combined_tower_height_l578_57847

/-- The combined height of four towers given specific conditions -/
theorem combined_tower_height :
  ∀ (clyde grace sarah linda : ℝ),
  grace = 8 * clyde →
  grace = 40.5 →
  sarah = 2 * clyde →
  linda = (clyde + grace + sarah) / 3 →
  clyde + grace + sarah + linda = 74.25 := by
  sorry

end combined_tower_height_l578_57847


namespace bisecting_line_theorem_l578_57851

/-- The pentagon vertices -/
def A : ℝ × ℝ := (0, 0)
def B : ℝ × ℝ := (11, 0)
def C : ℝ × ℝ := (11, 2)
def D : ℝ × ℝ := (6, 2)
def E : ℝ × ℝ := (0, 8)

/-- The area of the pentagon -/
noncomputable def pentagonArea : ℝ := sorry

/-- The x-coordinate of the bisecting line -/
noncomputable def bisectingLineX : ℝ := 8 - 2 * Real.sqrt 6

/-- The area of the left part of the pentagon when divided by the line x = bisectingLineX -/
noncomputable def leftArea : ℝ := sorry

/-- The area of the right part of the pentagon when divided by the line x = bisectingLineX -/
noncomputable def rightArea : ℝ := sorry

/-- Theorem stating that the line x = 8 - 2√6 bisects the area of the pentagon -/
theorem bisecting_line_theorem : leftArea = rightArea ∧ leftArea + rightArea = pentagonArea := by sorry

end bisecting_line_theorem_l578_57851


namespace time_for_type_A_is_60_l578_57811

/-- Represents the time allocation for an examination with different problem types. -/
structure ExamTime where
  totalQuestions : ℕ
  typeA : ℕ
  typeB : ℕ
  typeC : ℕ
  totalTime : ℕ
  lastHour : ℕ

/-- Calculates the time spent on Type A problems in the examination. -/
def timeForTypeA (e : ExamTime) : ℕ :=
  let typeB_time := (e.lastHour * 2) / e.typeC
  (e.typeA * typeB_time * 2)

/-- Theorem stating that the time spent on Type A problems is 60 minutes. -/
theorem time_for_type_A_is_60 (e : ExamTime) 
  (h1 : e.totalQuestions = 200)
  (h2 : e.typeA = 20)
  (h3 : e.typeB = 100)
  (h4 : e.typeC = 80)
  (h5 : e.totalTime = 180)
  (h6 : e.lastHour = 60) :
  timeForTypeA e = 60 := by
  sorry

end time_for_type_A_is_60_l578_57811


namespace correct_calculation_l578_57806

-- Define the variables
variable (AB : ℝ) (C : ℝ) (D : ℝ) (E : ℝ)

-- Define the conditions
def xiao_hu_error := AB * C + D * E * 10 = 39.6
def da_hu_error := AB * C * D * E = 36.9

-- State the theorem
theorem correct_calculation (h1 : xiao_hu_error AB C D E) (h2 : da_hu_error AB C D E) :
  AB * C + D * E = 26.1 := by
  sorry

end correct_calculation_l578_57806


namespace circle_equation_l578_57858

/-- Given a circle C with center (a, 0) tangent to the line y = (√3/3)x at point N(3, √3),
    prove that the equation of circle C is (x-4)² + y² = 4 -/
theorem circle_equation (a : ℝ) :
  let C : Set (ℝ × ℝ) := {p | (p.1 - a)^2 + p.2^2 = ((3 - a)^2 + 3)}
  let l : Set (ℝ × ℝ) := {p | p.2 = (Real.sqrt 3 / 3) * p.1}
  let N : ℝ × ℝ := (3, Real.sqrt 3)
  (N ∈ C) ∧ (N ∈ l) ∧ (∀ p ∈ C, p ≠ N → p ∉ l) →
  C = {p : ℝ × ℝ | (p.1 - 4)^2 + p.2^2 = 4} :=
by sorry

end circle_equation_l578_57858


namespace cylinder_height_after_forging_l578_57887

theorem cylinder_height_after_forging (initial_diameter initial_height new_diameter : ℝ) 
  (h_initial_diameter : initial_diameter = 6)
  (h_initial_height : initial_height = 24)
  (h_new_diameter : new_diameter = 16) :
  let new_height := (initial_diameter^2 * initial_height) / new_diameter^2
  new_height = 27 / 8 := by sorry

end cylinder_height_after_forging_l578_57887


namespace no_solution_to_equation_l578_57855

theorem no_solution_to_equation :
  ¬∃ s : ℝ, (s^2 - 6*s + 8) / (s^2 - 9*s + 20) = (s^2 - 3*s - 18) / (s^2 - 2*s - 15) :=
by sorry

end no_solution_to_equation_l578_57855


namespace triangle_angle_calculation_l578_57831

theorem triangle_angle_calculation (a b c : ℝ) (A B C : ℝ) :
  a = Real.sqrt 2 →
  b = Real.sqrt 3 →
  B = π / 3 →  -- 60° in radians
  (a / Real.sin A = b / Real.sin B) →  -- Law of Sines
  A = π / 4  -- 45° in radians
:= by sorry

end triangle_angle_calculation_l578_57831


namespace right_triangle_hypotenuse_l578_57877

theorem right_triangle_hypotenuse (a b c : ℝ) : 
  a = 12 →                             -- One side is 12 cm
  (1/2) * a * b = 54 →                 -- Area of the triangle is 54 square centimeters
  a^2 + b^2 = c^2 →                    -- Pythagorean theorem (right-angled triangle)
  c = 15 :=                            -- Hypotenuse length is 15 cm
by sorry

end right_triangle_hypotenuse_l578_57877


namespace local_road_speed_l578_57807

theorem local_road_speed (local_distance : ℝ) (highway_distance : ℝ) 
  (highway_speed : ℝ) (average_speed : ℝ) (local_speed : ℝ) : 
  local_distance = 60 ∧ 
  highway_distance = 65 ∧ 
  highway_speed = 65 ∧ 
  average_speed = 41.67 ∧
  (local_distance + highway_distance) / ((local_distance / local_speed) + (highway_distance / highway_speed)) = average_speed →
  local_speed = 30 := by
sorry

end local_road_speed_l578_57807


namespace spherical_coordinate_conversion_l578_57852

/-- Proves that the given spherical coordinates are equivalent to the standard representation -/
theorem spherical_coordinate_conversion (ρ θ φ : Real) :
  ρ > 0 →
  0 ≤ θ ∧ θ < 2 * π →
  0 ≤ φ ∧ φ ≤ π →
  (ρ, θ, φ) = (4, 4 * π / 3, π / 5) ↔ (ρ, θ, φ) = (4, π / 3, 9 * π / 5) :=
by sorry

end spherical_coordinate_conversion_l578_57852


namespace fencing_cost_approx_l578_57879

-- Define the diameter of the circular field
def diameter : ℝ := 40

-- Define the cost per meter of fencing
def cost_per_meter : ℝ := 3

-- Define pi as a constant (approximation)
def π : ℝ := 3.14159

-- Define the function to calculate the circumference of a circle
def circumference (d : ℝ) : ℝ := π * d

-- Define the function to calculate the total cost of fencing
def total_cost (c : ℝ) (rate : ℝ) : ℝ := c * rate

-- Theorem stating that the total cost is approximately 377
theorem fencing_cost_approx :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 1 ∧ 
  abs (total_cost (circumference diameter) cost_per_meter - 377) < ε :=
sorry

end fencing_cost_approx_l578_57879


namespace fraction_equality_l578_57837

theorem fraction_equality (a b : ℝ) (h : a / b = 5 / 2) : (a + 2*b) / (a - b) = 3 := by
  sorry

end fraction_equality_l578_57837


namespace proportional_function_quadrants_l578_57853

/-- A proportional function passing through the second and fourth quadrants has a negative coefficient. -/
theorem proportional_function_quadrants (k : ℝ) :
  (∀ x y : ℝ, y = k * x →
    ((x < 0 ∧ y > 0) ∨ (x > 0 ∧ y < 0))) →
  k < 0 := by
  sorry

end proportional_function_quadrants_l578_57853


namespace f_properties_l578_57892

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x + Real.log (x + 1) - a * x - Real.cos x

theorem f_properties (a : ℝ) :
  (∀ x > -1, a ≤ 1 → Monotone (f a)) ∧
  (∃ a, deriv (f a) 0 = 0) := by sorry

end f_properties_l578_57892


namespace line_tangent_to_parabola_l578_57803

/-- A line is tangent to a parabola if and only if the discriminant of the resulting quadratic equation is zero -/
theorem line_tangent_to_parabola (k : ℝ) :
  (∃ x y : ℝ, 3 * x + 5 * y + k = 0 ∧ y^2 = 24 * x ∧
    ∀ x' y' : ℝ, 3 * x' + 5 * y' + k = 0 ∧ y'^2 = 24 * x' → (x', y') = (x, y))
  ↔ k = 50 := by
  sorry

end line_tangent_to_parabola_l578_57803


namespace slab_rate_calculation_l578_57805

/-- Given a room with specified dimensions and total flooring cost, 
    calculate the rate per square meter for the slabs. -/
theorem slab_rate_calculation (length width total_cost : ℝ) 
    (h_length : length = 5.5)
    (h_width : width = 3.75)
    (h_total_cost : total_cost = 24750) : 
  total_cost / (length * width) = 1200 := by
  sorry


end slab_rate_calculation_l578_57805
