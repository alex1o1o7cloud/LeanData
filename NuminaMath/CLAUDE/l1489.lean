import Mathlib

namespace train_passing_time_l1489_148995

/-- Two trains passing problem -/
theorem train_passing_time (length1 length2 : ℝ) (speed1 speed2 : ℝ) (h1 : length1 = 280)
    (h2 : length2 = 350) (h3 : speed1 = 72) (h4 : speed2 = 54) :
    (length1 + length2) / ((speed1 - speed2) * (1000 / 3600)) = 126 := by
  sorry

end train_passing_time_l1489_148995


namespace average_of_four_numbers_l1489_148904

theorem average_of_four_numbers (p q r s : ℝ) 
  (h : (5 : ℝ) / 4 * (p + q + r + s) = 15) : 
  (p + q + r + s) / 4 = 3 := by
  sorry

end average_of_four_numbers_l1489_148904


namespace regular_polygon_perimeter_l1489_148912

/-- A regular polygon with side length 7 units and exterior angle 45 degrees has a perimeter of 56 units. -/
theorem regular_polygon_perimeter (s : ℝ) (θ : ℝ) (h1 : s = 7) (h2 : θ = 45) :
  let n : ℝ := 360 / θ
  let perimeter : ℝ := n * s
  perimeter = 56 := by sorry

end regular_polygon_perimeter_l1489_148912


namespace machine_C_time_l1489_148945

/-- Time for Machine A to finish the job -/
def timeA : ℝ := 4

/-- Time for Machine B to finish the job -/
def timeB : ℝ := 12

/-- Time for all machines together to finish the job -/
def timeAll : ℝ := 2

/-- Time for Machine C to finish the job -/
def timeC : ℝ := 6

/-- Theorem stating that given the conditions, Machine C takes 6 hours to finish the job alone -/
theorem machine_C_time : 
  1 / timeA + 1 / timeB + 1 / timeC = 1 / timeAll := by sorry

end machine_C_time_l1489_148945


namespace sum_of_squares_bounds_l1489_148958

/-- A quadrilateral inscribed in a unit square -/
structure InscribedQuadrilateral where
  w : Real
  x : Real
  y : Real
  z : Real
  w_in_range : 0 ≤ w ∧ w ≤ 1
  x_in_range : 0 ≤ x ∧ x ≤ 1
  y_in_range : 0 ≤ y ∧ y ≤ 1
  z_in_range : 0 ≤ z ∧ z ≤ 1

/-- The sum of squares of the sides of an inscribed quadrilateral -/
def sumOfSquares (q : InscribedQuadrilateral) : Real :=
  (q.w^2 + q.x^2) + ((1-q.x)^2 + q.y^2) + ((1-q.y)^2 + q.z^2) + ((1-q.z)^2 + (1-q.w)^2)

/-- Theorem: The sum of squares of the sides of a quadrilateral inscribed in a unit square is between 2 and 4 -/
theorem sum_of_squares_bounds (q : InscribedQuadrilateral) : 
  2 ≤ sumOfSquares q ∧ sumOfSquares q ≤ 4 := by
  sorry

end sum_of_squares_bounds_l1489_148958


namespace log_fifty_equals_one_plus_log_five_l1489_148967

theorem log_fifty_equals_one_plus_log_five : Real.log 50 / Real.log 10 = 1 + Real.log 5 / Real.log 10 := by
  sorry

end log_fifty_equals_one_plus_log_five_l1489_148967


namespace quadratic_sqrt2_closure_l1489_148920

-- Define a structure for numbers of the form a + b√2
structure QuadraticSqrt2 where
  a : ℚ
  b : ℚ

-- Define addition for QuadraticSqrt2
def add (x y : QuadraticSqrt2) : QuadraticSqrt2 :=
  ⟨x.a + y.a, x.b + y.b⟩

-- Define subtraction for QuadraticSqrt2
def sub (x y : QuadraticSqrt2) : QuadraticSqrt2 :=
  ⟨x.a - y.a, x.b - y.b⟩

-- Define multiplication for QuadraticSqrt2
def mul (x y : QuadraticSqrt2) : QuadraticSqrt2 :=
  ⟨x.a * y.a + 2 * x.b * y.b, x.a * y.b + x.b * y.a⟩

-- Define division for QuadraticSqrt2
def div (x y : QuadraticSqrt2) : QuadraticSqrt2 :=
  let denom := y.a * y.a - 2 * y.b * y.b
  ⟨(x.a * y.a - 2 * x.b * y.b) / denom, (x.b * y.a - x.a * y.b) / denom⟩

theorem quadratic_sqrt2_closure (x y : QuadraticSqrt2) (h : y.a * y.a ≠ 2 * y.b * y.b) :
  (∃ (z : QuadraticSqrt2), add x y = z) ∧
  (∃ (z : QuadraticSqrt2), sub x y = z) ∧
  (∃ (z : QuadraticSqrt2), mul x y = z) ∧
  (∃ (z : QuadraticSqrt2), div x y = z) :=
sorry

end quadratic_sqrt2_closure_l1489_148920


namespace min_value_of_S_l1489_148971

theorem min_value_of_S (x y : ℝ) : 2 * x^2 - x*y + y^2 + 2*x + 3*y ≥ -4 := by
  sorry

end min_value_of_S_l1489_148971


namespace factorial_ratio_equals_24_l1489_148922

theorem factorial_ratio_equals_24 :
  ∃! (n : ℕ), n > 3 ∧ n.factorial / (n - 3).factorial = 24 := by
  sorry

end factorial_ratio_equals_24_l1489_148922


namespace smallest_M_inequality_l1489_148925

theorem smallest_M_inequality (a b c : ℝ) :
  let M : ℝ := (9 * Real.sqrt 2) / 32
  ∀ N : ℝ, (∀ x y z : ℝ, |x*y*(x^2 - y^2) + y*z*(y^2 - z^2) + z*x*(z^2 - x^2)| ≤ N*(x^2 + y^2 + z^2)^2) →
    M ≤ N ∧ |a*b*(a^2 - b^2) + b*c*(b^2 - c^2) + c*a*(c^2 - a^2)| ≤ M*(a^2 + b^2 + c^2)^2 :=
by sorry

end smallest_M_inequality_l1489_148925


namespace rooster_count_l1489_148942

theorem rooster_count (total_birds : ℕ) (rooster_ratio hen_ratio chick_ratio duck_ratio goose_ratio : ℕ) 
  (h1 : total_birds = 9000)
  (h2 : rooster_ratio = 4)
  (h3 : hen_ratio = 2)
  (h4 : chick_ratio = 6)
  (h5 : duck_ratio = 3)
  (h6 : goose_ratio = 1) :
  (total_birds * rooster_ratio) / (rooster_ratio + hen_ratio + chick_ratio + duck_ratio + goose_ratio) = 2250 := by
  sorry

end rooster_count_l1489_148942


namespace total_crayons_l1489_148930

/-- Represents the number of crayons in a box of type 1 -/
def box_type1 : ℕ := 8 + 4 + 5

/-- Represents the number of crayons in a box of type 2 -/
def box_type2 : ℕ := 7 + 6 + 3

/-- Represents the number of crayons in a box of type 3 -/
def box_type3 : ℕ := 11 + 5 + 2

/-- Represents the number of crayons in the unique box -/
def unique_box : ℕ := 9 + 2 + 7

/-- Represents the total number of boxes -/
def total_boxes : ℕ := 3 + 4 + 2 + 1

theorem total_crayons : 
  3 * box_type1 + 4 * box_type2 + 2 * box_type3 + unique_box = 169 :=
sorry

end total_crayons_l1489_148930


namespace root_sum_theorem_l1489_148989

theorem root_sum_theorem (α β γ : ℂ) : 
  (α^3 - α - 1 = 0) → 
  (β^3 - β - 1 = 0) → 
  (γ^3 - γ - 1 = 0) → 
  ((1 - α) / (1 + α) + (1 - β) / (1 + β) + (1 - γ) / (1 + γ) = 1) := by
sorry

end root_sum_theorem_l1489_148989


namespace not_p_sufficient_not_necessary_for_not_q_range_of_a_for_not_r_necessary_not_sufficient_for_not_p_l1489_148933

-- Define the predicates p, q, and r
def p (x : ℝ) : Prop := |3*x - 4| > 2
def q (x : ℝ) : Prop := 1 / (x^2 - x - 2) > 0
def r (x a : ℝ) : Prop := (x - a) * (x - a - 1) < 0

-- Define the negations of p, q, and r
def not_p (x : ℝ) : Prop := ¬(p x)
def not_q (x : ℝ) : Prop := ¬(q x)
def not_r (x a : ℝ) : Prop := ¬(r x a)

-- Theorem 1: ¬p is a sufficient but not necessary condition for ¬q
theorem not_p_sufficient_not_necessary_for_not_q :
  (∀ x, not_p x → not_q x) ∧ ¬(∀ x, not_q x → not_p x) :=
sorry

-- Theorem 2: Range of a for which ¬r is a necessary but not sufficient condition for ¬p
theorem range_of_a_for_not_r_necessary_not_sufficient_for_not_p :
  ∀ a, (∀ x, not_p x → not_r x a) ∧ ¬(∀ x, not_r x a → not_p x) ↔ (a ≥ 2 ∨ a ≤ -1/3) :=
sorry

end not_p_sufficient_not_necessary_for_not_q_range_of_a_for_not_r_necessary_not_sufficient_for_not_p_l1489_148933


namespace range_of_a_for_two_zeros_l1489_148928

noncomputable def f (a x : ℝ) : ℝ := 
  if x ≥ a then x else x^3 - 3*x

noncomputable def g (a x : ℝ) : ℝ := 2 * f a x - a * x

theorem range_of_a_for_two_zeros (a : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ g a x = 0 ∧ g a y = 0) ∧
  (∀ x y z : ℝ, x ≠ y ∧ y ≠ z ∧ x ≠ z → ¬(g a x = 0 ∧ g a y = 0 ∧ g a z = 0)) →
  a > -3/2 ∧ a < 2 :=
sorry

end range_of_a_for_two_zeros_l1489_148928


namespace angle_457_properties_l1489_148949

-- Define the set of angles with the same terminal side as -457°
def same_terminal_side (β : ℝ) : Prop :=
  ∃ k : ℤ, β = k * 360 - 457

-- Define the third quadrant
def third_quadrant (θ : ℝ) : Prop :=
  180 < θ % 360 ∧ θ % 360 < 270

-- Theorem statement
theorem angle_457_properties :
  (∀ β, same_terminal_side β ↔ ∃ k : ℤ, β = k * 360 - 457) ∧
  third_quadrant (-457) := by
  sorry

end angle_457_properties_l1489_148949


namespace birds_in_tree_l1489_148910

theorem birds_in_tree (initial_birds final_birds : ℕ) 
  (h1 : initial_birds = 29)
  (h2 : final_birds = 42) :
  final_birds - initial_birds = 13 := by
  sorry

end birds_in_tree_l1489_148910


namespace triangle_median_equality_l1489_148998

-- Define the triangle PQR
structure Triangle :=
  (P Q R : ℝ × ℝ)

-- Define the length function
def length (a b : ℝ × ℝ) : ℝ := sorry

-- Define the median function
def median (t : Triangle) (v : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem triangle_median_equality (t : Triangle) :
  length t.P t.Q = 2 →
  length t.P t.R = 3 →
  length t.Q t.R = median t t.P →
  length t.Q t.R = Real.sqrt (26 * 0.2) :=
by sorry

end triangle_median_equality_l1489_148998


namespace cubic_minus_xy_squared_factorization_l1489_148929

theorem cubic_minus_xy_squared_factorization (x y : ℝ) :
  x^3 - x*y^2 = x*(x+y)*(x-y) := by
  sorry

end cubic_minus_xy_squared_factorization_l1489_148929


namespace fraction_of_single_men_l1489_148909

theorem fraction_of_single_men
  (total : ℕ)
  (h_total_pos : total > 0)
  (women_ratio : ℚ)
  (h_women_ratio : women_ratio = 70 / 100)
  (married_ratio : ℚ)
  (h_married_ratio : married_ratio = 40 / 100)
  (married_men_ratio : ℚ)
  (h_married_men_ratio : married_men_ratio = 2 / 3)
  : (total - women_ratio * total - married_men_ratio * (total - women_ratio * total)) / (total - women_ratio * total) = 1 / 3 := by
  sorry

end fraction_of_single_men_l1489_148909


namespace sum_of_three_numbers_l1489_148952

theorem sum_of_three_numbers (a b c : ℝ) 
  (h1 : a^2 + b^2 + c^2 = 252) 
  (h2 : a*b + b*c + c*a = 116) : 
  a + b + c = 22 := by
sorry

end sum_of_three_numbers_l1489_148952


namespace stratified_sample_size_l1489_148957

/-- Represents the composition of a school population -/
structure SchoolPopulation where
  teachers : ℕ
  male_students : ℕ
  female_students : ℕ

/-- Represents a stratified sample from the school population -/
structure StratifiedSample where
  total_size : ℕ
  female_sample : ℕ

/-- Theorem: Given a school population and a stratified sample where 80 people are drawn
    from the female students, the total sample size is 192 -/
theorem stratified_sample_size 
  (pop : SchoolPopulation) 
  (sample : StratifiedSample) :
  pop.teachers = 200 →
  pop.male_students = 1200 →
  pop.female_students = 1000 →
  sample.female_sample = 80 →
  sample.total_size = 192 := by
  sorry

#check stratified_sample_size

end stratified_sample_size_l1489_148957


namespace store_profit_optimization_l1489_148906

/-- Represents the store's sales and profit model -/
structure StoreSalesModel where
  purchase_price : ℕ
  initial_selling_price : ℕ
  initial_monthly_sales : ℕ
  additional_sales_per_yuan : ℕ

/-- Calculates the monthly profit given a price reduction -/
def monthly_profit (model : StoreSalesModel) (price_reduction : ℕ) : ℕ :=
  let new_price := model.initial_selling_price - price_reduction
  let new_sales := model.initial_monthly_sales + model.additional_sales_per_yuan * price_reduction
  (new_price - model.purchase_price) * new_sales

/-- Theorem stating the initial monthly profit and the optimal price reduction -/
theorem store_profit_optimization (model : StoreSalesModel) 
  (h1 : model.purchase_price = 280)
  (h2 : model.initial_selling_price = 360)
  (h3 : model.initial_monthly_sales = 60)
  (h4 : model.additional_sales_per_yuan = 5) :
  (monthly_profit model 0 = 4800) ∧ 
  (monthly_profit model 60 = 7200) ∧ 
  (∀ x, x ≠ 60 → monthly_profit model x ≤ 7200) :=
sorry

end store_profit_optimization_l1489_148906


namespace coin_value_difference_l1489_148960

/-- Represents the number of coins of each type -/
structure CoinCount where
  pennies : ℕ
  nickels : ℕ
  dimes : ℕ

/-- Calculates the total value in cents for a given coin count -/
def totalValue (coins : CoinCount) : ℕ :=
  coins.pennies + 5 * coins.nickels + 10 * coins.dimes

/-- Represents the constraint that there are 2500 coins in total -/
def totalCoins (coins : CoinCount) : Prop :=
  coins.pennies + coins.nickels + coins.dimes = 2500

/-- Represents the constraint that there is at least one of each type of coin -/
def atLeastOne (coins : CoinCount) : Prop :=
  coins.pennies ≥ 1 ∧ coins.nickels ≥ 1 ∧ coins.dimes ≥ 1

theorem coin_value_difference :
  ∃ (maxCoins minCoins : CoinCount),
    totalCoins maxCoins ∧
    totalCoins minCoins ∧
    atLeastOne maxCoins ∧
    atLeastOne minCoins ∧
    (∀ (coins : CoinCount), totalCoins coins → atLeastOne coins →
      totalValue coins ≤ totalValue maxCoins) ∧
    (∀ (coins : CoinCount), totalCoins coins → atLeastOne coins →
      totalValue coins ≥ totalValue minCoins) ∧
    totalValue maxCoins - totalValue minCoins = 22473 :=
by sorry

end coin_value_difference_l1489_148960


namespace result_calculation_l1489_148915

theorem result_calculation (h1 : 7125 / 1.25 = 5700) (h2 : x = 3) : 
  (712.5 / 12.5) ^ x = 185193 := by
sorry

end result_calculation_l1489_148915


namespace pen_price_proof_l1489_148979

theorem pen_price_proof (total_cost : ℝ) (notebook_ratio : ℝ) :
  total_cost = 36.45 →
  notebook_ratio = 15 / 4 →
  ∃ (pen_price : ℝ),
    pen_price + 3 * (notebook_ratio * pen_price) = total_cost ∧
    pen_price = 5.4 :=
by sorry

end pen_price_proof_l1489_148979


namespace set_in_proportion_l1489_148935

/-- A set of four numbers (a, b, c, d) is in proportion if a:b = c:d -/
def IsInProportion (a b c d : ℚ) : Prop :=
  a * d = b * c

/-- Prove that the set (1, 2, 2, 4) is in proportion -/
theorem set_in_proportion :
  IsInProportion 1 2 2 4 := by
  sorry

end set_in_proportion_l1489_148935


namespace second_number_solution_l1489_148964

theorem second_number_solution (x : ℝ) :
  12.1212 + x - 9.1103 = 20.011399999999995 →
  x = 18.000499999999995 := by
sorry

end second_number_solution_l1489_148964


namespace abs_d_equals_three_l1489_148938

/-- A polynomial with integer coefficients that has 3+i as a root -/
def f (a b c d : ℤ) : ℂ → ℂ := λ x => a*x^5 + b*x^4 + c*x^3 + d*x^2 + b*x + a

/-- The theorem stating that under given conditions, |d| = 3 -/
theorem abs_d_equals_three (a b c d : ℤ) : 
  f a b c d (3 + I) = 0 → 
  Nat.gcd (Nat.gcd (Nat.gcd a.natAbs b.natAbs) c.natAbs) d.natAbs = 1 → 
  d.natAbs = 3 := by sorry

end abs_d_equals_three_l1489_148938


namespace tree_boy_growth_rate_ratio_l1489_148914

/-- Given the initial and final heights of a tree and a boy, calculate the ratio of their growth rates. -/
theorem tree_boy_growth_rate_ratio
  (tree_initial : ℝ) (tree_final : ℝ)
  (boy_initial : ℝ) (boy_final : ℝ)
  (h_tree_initial : tree_initial = 16)
  (h_tree_final : tree_final = 40)
  (h_boy_initial : boy_initial = 24)
  (h_boy_final : boy_final = 36) :
  (tree_final - tree_initial) / (boy_final - boy_initial) = 2 := by
sorry

end tree_boy_growth_rate_ratio_l1489_148914


namespace function_inequality_implies_positive_a_l1489_148950

open Real

theorem function_inequality_implies_positive_a (a : ℝ) :
  (∃ x₀ : ℝ, x₀ ∈ Set.Icc 1 (Real.exp 1) ∧ 
    a * (x₀ - 1 / x₀) - 2 * log x₀ > -a / x₀) →
  a > 0 := by
sorry

end function_inequality_implies_positive_a_l1489_148950


namespace sqrt_problem_l1489_148963

theorem sqrt_problem (m n a : ℝ) : 
  (∃ (x : ℝ), x^2 = m ∧ x = 3) → 
  (∃ (y z : ℝ), y^2 = n ∧ z^2 = n ∧ y = a + 4 ∧ z = 2*a - 16) →
  m = 9 ∧ n = 64 ∧ (7*m - n)^(1/3) = -1 := by sorry

end sqrt_problem_l1489_148963


namespace chord_equation_l1489_148993

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := 3 * x^2 + 4 * y^2 = 24

-- Define point P
def P : ℝ × ℝ := (1, -2)

-- Define a chord AB that passes through P and is bisected by P
structure Chord :=
  (A B : ℝ × ℝ)
  (passes_through_P : (A.1 + B.1) / 2 = P.1 ∧ (A.2 + B.2) / 2 = P.2)
  (on_ellipse : ellipse A.1 A.2 ∧ ellipse B.1 B.2)

-- Theorem statement
theorem chord_equation (AB : Chord) : 
  ∃ (a b c : ℝ), a ≠ 0 ∧ b ≠ 0 ∧ (∀ (x y : ℝ), 
    ((x, y) = AB.A ∨ (x, y) = AB.B) → a * x + b * y + c = 0) ∧
    a = 3 ∧ b = -2 ∧ c = -7 := by
  sorry

end chord_equation_l1489_148993


namespace all_can_be_top_l1489_148991

/-- Represents a person with height and weight -/
structure Person where
  height : ℝ
  weight : ℝ

/-- Defines the "not inferior" relation between two people -/
def notInferior (a b : Person) : Prop :=
  a.height ≥ b.height ∨ a.weight ≥ b.weight

/-- Defines a top person as someone who is not inferior to all others -/
def isTop (p : Person) (group : Finset Person) : Prop :=
  ∀ q ∈ group, p ≠ q → notInferior p q

/-- Theorem: It's possible to have 100 top people in a group of 100 -/
theorem all_can_be_top :
  ∃ (group : Finset Person), Finset.card group = 100 ∧
    ∀ p ∈ group, isTop p group := by
  sorry


end all_can_be_top_l1489_148991


namespace complex_equation_solution_l1489_148978

theorem complex_equation_solution (z : ℂ) : z * (1 + Complex.I) = 2 * Complex.I → z = 1 + Complex.I := by
  sorry

end complex_equation_solution_l1489_148978


namespace product_digits_sum_l1489_148980

/-- Converts a base-9 number to base-10 --/
def toBase10 (digits : List Nat) : Nat :=
  digits.foldr (fun d acc => acc * 9 + d) 0

/-- Converts a base-10 number to base-9 --/
def toBase9 (n : Nat) : List Nat :=
  if n = 0 then [0] else
  let rec aux (m : Nat) (acc : List Nat) : List Nat :=
    if m = 0 then acc else aux (m / 9) ((m % 9) :: acc)
  aux n []

/-- Sums the digits of a number represented as a list of digits --/
def sumDigits (digits : List Nat) : Nat :=
  digits.sum

theorem product_digits_sum :
  let a := [1, 2, 5]  -- 125 in base 9
  let b := [3, 3]     -- 33 in base 9
  let product := toBase10 a * toBase10 b
  sumDigits (toBase9 product) = 16 := by sorry

end product_digits_sum_l1489_148980


namespace points_form_circle_l1489_148970

theorem points_form_circle :
  ∀ (t : ℝ), (∃ (x y : ℝ), x = Real.cos t ∧ y = Real.sin t) →
  ∃ (r : ℝ), x^2 + y^2 = r^2 :=
sorry

end points_form_circle_l1489_148970


namespace BA_is_2I_l1489_148956

theorem BA_is_2I (A : Matrix (Fin 4) (Fin 2) ℝ) (B : Matrix (Fin 2) (Fin 4) ℝ) 
  (h : A * B = !![1, 0, -1, 0; 0, 1, 0, -1; -1, 0, 1, 0; 0, -1, 0, 1]) :
  B * A = !![2, 0; 0, 2] := by sorry

end BA_is_2I_l1489_148956


namespace perfect_square_sum_l1489_148939

theorem perfect_square_sum (n : ℕ) : 
  n > 0 ∧ n < 200 ∧ (∃ k : ℕ, n^2 + (n+1)^2 = k^2) ↔ n = 3 ∨ n = 20 ∨ n = 119 := by
  sorry

end perfect_square_sum_l1489_148939


namespace fraction_sum_cubes_l1489_148985

theorem fraction_sum_cubes : (5 / 6 : ℚ)^3 + (3 / 5 : ℚ)^3 = 21457 / 27000 := by
  sorry

end fraction_sum_cubes_l1489_148985


namespace range_of_a_l1489_148962

-- Define the sets N and M
def N (a : ℝ) : Set ℝ := {x | (x - a) * (x + a - 2) < 0}
def M : Set ℝ := {x | -1/2 ≤ x ∧ x < 2}

-- State the theorem
theorem range_of_a :
  (∀ x, x ∈ M → x ∈ N a) → (a ≤ -1/2 ∨ a ≥ 5/2) :=
by sorry

end range_of_a_l1489_148962


namespace complex_equation_solution_l1489_148982

theorem complex_equation_solution (z : ℂ) : 
  Complex.abs z - 2 * z = -1 + 8 * Complex.I → 
  z = 3 - 4 * Complex.I ∨ z = -5/3 - 4 * Complex.I := by
sorry

end complex_equation_solution_l1489_148982


namespace sqrt_sum_squares_equals_twice_sum_iff_zero_l1489_148908

theorem sqrt_sum_squares_equals_twice_sum_iff_zero (a b : ℝ) : 
  a ≥ 0 → b ≥ 0 → (Real.sqrt (a^2 + b^2) = 2 * (a + b) ↔ a = 0 ∧ b = 0) := by
sorry

end sqrt_sum_squares_equals_twice_sum_iff_zero_l1489_148908


namespace equation_holds_iff_conditions_l1489_148917

theorem equation_holds_iff_conditions (a b c : ℤ) :
  a * (a - b) + b * (b - c) + c * (c - a) = 2 ↔ 
  ((a = b ∧ b = c + 1) ∨ (a = c ∧ b - 1 = c)) := by
sorry

end equation_holds_iff_conditions_l1489_148917


namespace train_length_problem_l1489_148976

/-- The length of a train given its speed and time to cross a fixed point. -/
def trainLength (speed : ℝ) (time : ℝ) : ℝ := speed * time

/-- Theorem: A train traveling at 30 m/s that takes 12 seconds to cross a fixed point has a length of 360 meters. -/
theorem train_length_problem : trainLength 30 12 = 360 := by
  sorry

end train_length_problem_l1489_148976


namespace difference_of_squares_65_35_l1489_148969

theorem difference_of_squares_65_35 : 65^2 - 35^2 = 3000 := by
  sorry

end difference_of_squares_65_35_l1489_148969


namespace local_max_at_one_f_one_eq_zero_l1489_148992

/-- The function f(x) = x³ - 3x² + 2 has a local maximum value of 0 at x = 1 -/
theorem local_max_at_one (f : ℝ → ℝ) (hf : ∀ x, f x = x^3 - 3*x^2 + 2) :
  ∃ δ > 0, ∀ x, |x - 1| < δ → f x ≤ f 1 := by
  sorry

/-- The value of f(1) is 0 -/
theorem f_one_eq_zero (f : ℝ → ℝ) (hf : ∀ x, f x = x^3 - 3*x^2 + 2) :
  f 1 = 0 := by
  sorry

end local_max_at_one_f_one_eq_zero_l1489_148992


namespace parallelogram_height_l1489_148941

theorem parallelogram_height (area base height : ℝ) : 
  area = 480 ∧ base = 32 ∧ area = base * height → height = 15 := by
  sorry

end parallelogram_height_l1489_148941


namespace stable_painted_area_l1489_148986

/-- Calculates the total area to be painted for a rectangular stable with a chimney -/
def total_painted_area (width length height chim_width chim_length chim_height : ℝ) : ℝ :=
  let wall_area_1 := 2 * 2 * (width * height)
  let wall_area_2 := 2 * 2 * (length * height)
  let roof_area := width * length
  let ceiling_area := width * length
  let chimney_area := 4 * (chim_width * chim_height) + (chim_width * chim_length)
  wall_area_1 + wall_area_2 + roof_area + ceiling_area + chimney_area

/-- Theorem stating that the total area to be painted for the given stable is 1060 sq yd -/
theorem stable_painted_area :
  total_painted_area 12 15 6 2 2 2 = 1060 := by
  sorry

end stable_painted_area_l1489_148986


namespace solve_for_P_l1489_148996

theorem solve_for_P : ∃ P : ℝ, (P^3).sqrt = 81 * Real.rpow 81 (1/3) → P = Real.rpow 3 (32/9) := by
  sorry

end solve_for_P_l1489_148996


namespace largest_angle_of_obtuse_isosceles_triangle_l1489_148932

-- Define the triangle PQR
structure Triangle (P Q R : Point) where
  -- Add any necessary fields

-- Define the properties of the triangle
def isObtuse (t : Triangle P Q R) : Prop := sorry
def isIsosceles (t : Triangle P Q R) : Prop := sorry
def angleMeasure (p : Point) (t : Triangle P Q R) : ℝ := sorry
def largestAngle (t : Triangle P Q R) : ℝ := sorry

-- Theorem statement
theorem largest_angle_of_obtuse_isosceles_triangle 
  (P Q R : Point) (t : Triangle P Q R)
  (h_obtuse : isObtuse t)
  (h_isosceles : isIsosceles t)
  (h_angle_P : angleMeasure P t = 30) :
  largestAngle t = 120 := by sorry

end largest_angle_of_obtuse_isosceles_triangle_l1489_148932


namespace largest_coin_distribution_l1489_148983

theorem largest_coin_distribution (n : ℕ) : 
  (∃ k : ℕ, n = 12 * k + 3) ∧ 
  n < 100 ∧ 
  (∀ m : ℕ, (∃ j : ℕ, m = 12 * j + 3) → m < 100 → m ≤ n) → 
  n = 99 := by
sorry

end largest_coin_distribution_l1489_148983


namespace perpendicular_similarity_l1489_148918

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- A triangle defined by three points -/
structure Triangle where
  A : Point
  B : Point
  C : Point

/-- Checks if a triangle is acute -/
def isAcute (t : Triangle) : Prop :=
  sorry -- Definition of acute triangle

/-- Checks if a point is inside a triangle -/
def isInside (p : Point) (t : Triangle) : Prop :=
  sorry -- Definition of point being inside a triangle

/-- Constructs a new triangle by dropping perpendiculars from a point to the sides of another triangle -/
def dropPerpendiculars (p : Point) (t : Triangle) : Triangle :=
  sorry -- Definition of dropping perpendiculars

/-- Checks if two triangles are similar -/
def areSimilar (t1 t2 : Triangle) : Prop :=
  sorry -- Definition of triangle similarity

theorem perpendicular_similarity 
  (ABC : Triangle) 
  (P : Point) 
  (h_acute : isAcute ABC) 
  (h_inside : isInside P ABC) : 
  let A₁B₁C₁ := dropPerpendiculars P ABC
  let A₂B₂C₂ := dropPerpendiculars P A₁B₁C₁
  let A₃B₃C₃ := dropPerpendiculars P A₂B₂C₂
  areSimilar A₃B₃C₃ ABC :=
by
  sorry

end perpendicular_similarity_l1489_148918


namespace distinct_roots_find_m_l1489_148966

-- Define the quadratic equation
def quadratic (x m : ℝ) : ℝ := x^2 - 2*m*x + m^2 - 9

-- Define the discriminant of the quadratic equation
def discriminant (m : ℝ) : ℝ := (-2*m)^2 - 4*(m^2 - 9)

-- Theorem 1: The quadratic equation always has two distinct real roots
theorem distinct_roots (m : ℝ) : discriminant m > 0 := by sorry

-- Define the roots of the quadratic equation
noncomputable def x₁ (m : ℝ) : ℝ := sorry
noncomputable def x₂ (m : ℝ) : ℝ := sorry

-- Theorem 2: When x₂ = 3x₁, m = ±6
theorem find_m : 
  ∃ m : ℝ, (x₂ m = 3 * x₁ m) ∧ (m = 6 ∨ m = -6) := by sorry

end distinct_roots_find_m_l1489_148966


namespace cobbler_weekly_shoes_l1489_148965

/-- The number of pairs of shoes a cobbler can mend per hour -/
def shoes_per_hour : ℕ := 3

/-- The number of hours the cobbler works from Monday to Thursday each day -/
def hours_per_day : ℕ := 8

/-- The number of days the cobbler works full hours (Monday to Thursday) -/
def full_days : ℕ := 4

/-- The number of hours the cobbler works on Friday -/
def friday_hours : ℕ := 3

/-- The total number of pairs of shoes the cobbler can mend in a week -/
def total_shoes : ℕ := shoes_per_hour * (hours_per_day * full_days + friday_hours)

theorem cobbler_weekly_shoes : total_shoes = 105 := by
  sorry

end cobbler_weekly_shoes_l1489_148965


namespace statement_1_statement_2_false_statement_3_statement_4_l1489_148975

-- Statement ①
theorem statement_1 (a b : ℝ) (f : ℝ → ℝ) (h : ∀ x ∈ Set.Icc (2*a - 1) (a + 4), f x = a*x^2 + (2*a + b)*x + 2) :
  (∀ x ∈ Set.Icc (2*a - 1) (a + 4), f x = f (-x)) → b = 2 := by sorry

-- Statement ②
theorem statement_2_false : ∃ f : ℝ → ℝ, 
  (∀ x, f x = min (-2*x + 2) (-2*x^2 + 4*x + 2)) ∧ 
  (∃ x, f x > 1) := by sorry

-- Statement ③
theorem statement_3 (a : ℝ) (f : ℝ → ℝ) (h : ∀ x, f x = |2*x + a|) :
  (∀ x y, x ≥ 3 ∧ y > x → f x < f y) → a = -6 := by sorry

-- Statement ④
theorem statement_4 (f : ℝ → ℝ) 
  (h1 : ∃ x, f x ≠ 0) 
  (h2 : ∀ x y, f (x * y) = x * f y + y * f x) :
  ∀ x, f (-x) = -f x := by sorry

end statement_1_statement_2_false_statement_3_statement_4_l1489_148975


namespace discount_calculation_l1489_148934

-- Define the initial discount
def initial_discount : ℝ := 0.40

-- Define the additional discount
def additional_discount : ℝ := 0.10

-- Define the claimed total discount
def claimed_discount : ℝ := 0.55

-- Theorem to prove the actual discount and the difference
theorem discount_calculation :
  let remaining_after_initial := 1 - initial_discount
  let remaining_after_additional := remaining_after_initial * (1 - additional_discount)
  let actual_discount := 1 - remaining_after_additional
  let discount_difference := claimed_discount - actual_discount
  actual_discount = 0.46 ∧ discount_difference = 0.09 := by sorry

end discount_calculation_l1489_148934


namespace lines_perp_to_plane_are_parallel_l1489_148903

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular relation between a line and a plane
variable (perp : Line → Plane → Prop)

-- Define the parallel relation between two lines
variable (parallel : Line → Line → Prop)

-- Theorem statement
theorem lines_perp_to_plane_are_parallel 
  (a b : Line) (α : Plane) 
  (h1 : perp a α) (h2 : perp b α) : 
  parallel a b :=
sorry

end lines_perp_to_plane_are_parallel_l1489_148903


namespace probability_of_divisor_of_12_l1489_148981

/-- An 8-sided die numbered from 1 to 8 -/
def Die := Finset.range 8

/-- The set of divisors of 12 that are less than or equal to 8 -/
def DivisorsOf12 : Finset ℕ := {1, 2, 3, 4, 6}

/-- The probability of rolling a divisor of 12 on an 8-sided die -/
def probability : ℚ := (DivisorsOf12.card : ℚ) / (Die.card : ℚ)

theorem probability_of_divisor_of_12 : probability = 5/8 := by
  sorry

end probability_of_divisor_of_12_l1489_148981


namespace distribute_five_balls_three_boxes_l1489_148951

/-- The number of ways to distribute n indistinguishable balls into k distinguishable boxes -/
def distribute_balls (n : ℕ) (k : ℕ) : ℕ := sorry

/-- There are 5 indistinguishable balls -/
def num_balls : ℕ := 5

/-- There are 3 distinguishable boxes -/
def num_boxes : ℕ := 3

/-- The theorem states that there are 21 ways to distribute 5 indistinguishable balls into 3 distinguishable boxes -/
theorem distribute_five_balls_three_boxes : 
  distribute_balls num_balls num_boxes = 21 := by sorry

end distribute_five_balls_three_boxes_l1489_148951


namespace shop_weekly_earnings_value_l1489_148973

/-- Represents the shop's weekly earnings calculation -/
def shop_weekly_earnings : ℝ :=
  let open_minutes : ℕ := 12 * 60
  let womens_tshirts_sold : ℕ := open_minutes / 30
  let mens_tshirts_sold : ℕ := open_minutes / 40
  let womens_jeans_sold : ℕ := open_minutes / 45
  let mens_jeans_sold : ℕ := open_minutes / 60
  let unisex_hoodies_sold : ℕ := open_minutes / 70

  let daily_earnings : ℝ :=
    womens_tshirts_sold * 18 +
    mens_tshirts_sold * 15 +
    womens_jeans_sold * 40 +
    mens_jeans_sold * 45 +
    unisex_hoodies_sold * 35

  let wednesday_earnings : ℝ := daily_earnings * 0.9
  let saturday_earnings : ℝ := daily_earnings * 1.05
  let other_days_earnings : ℝ := daily_earnings * 5

  wednesday_earnings + saturday_earnings + other_days_earnings

theorem shop_weekly_earnings_value :
  shop_weekly_earnings = 15512.40 := by sorry

end shop_weekly_earnings_value_l1489_148973


namespace problem_solution_l1489_148900

theorem problem_solution : 
  3.2 * 2.25 - (5 * 0.85) / 2.5 = 5.5 := by
  sorry

end problem_solution_l1489_148900


namespace absolute_value_inequality_solution_set_l1489_148968

theorem absolute_value_inequality_solution_set :
  {x : ℝ | |5 - 2*x| < 3} = {x : ℝ | 1 < x ∧ x < 4} := by
  sorry

end absolute_value_inequality_solution_set_l1489_148968


namespace train_passenger_ratio_l1489_148988

theorem train_passenger_ratio :
  let initial_passengers : ℕ := 288
  let first_drop : ℕ := initial_passengers / 3
  let first_take : ℕ := 280
  let second_take : ℕ := 12
  let third_station_passengers : ℕ := 248
  
  let after_first_station : ℕ := initial_passengers - first_drop + first_take
  let dropped_second_station : ℕ := after_first_station - (third_station_passengers - second_take)
  let ratio : ℚ := dropped_second_station / after_first_station
  
  ratio = 1 / 2 := by sorry

end train_passenger_ratio_l1489_148988


namespace masons_father_age_l1489_148947

/-- Given the ages of Mason and Sydney, and their relationship to Mason's father's age,
    prove that Mason's father is 66 years old. -/
theorem masons_father_age (mason_age sydney_age father_age : ℕ) : 
  mason_age = 20 →
  sydney_age = 3 * mason_age →
  father_age = sydney_age + 6 →
  father_age = 66 := by
  sorry

end masons_father_age_l1489_148947


namespace total_seashells_l1489_148913

theorem total_seashells (joan_shells jessica_shells : ℕ) : 
  joan_shells = 6 → jessica_shells = 8 → joan_shells + jessica_shells = 14 := by
  sorry

end total_seashells_l1489_148913


namespace segment_ratio_l1489_148954

/-- Represents a point on a line segment --/
structure Point :=
  (x : ℝ)

/-- Represents a line segment between two points --/
structure Segment (A B : Point) :=
  (length : ℝ)

/-- The main theorem --/
theorem segment_ratio 
  (A B C D : Point)
  (h1 : Segment A D)
  (h2 : Segment A B)
  (h3 : Segment B D)
  (h4 : Segment A C)
  (h5 : Segment C D)
  (h6 : Segment B C)
  (cond1 : B.x < D.x ∧ D.x < C.x)
  (cond2 : h2.length = 3 * h3.length)
  (cond3 : h4.length = 4 * h5.length)
  (cond4 : h1.length = h2.length + h3.length + h5.length) :
  h6.length / h1.length = 5 / 6 :=
sorry

end segment_ratio_l1489_148954


namespace tan_20_plus_4sin_20_equals_sqrt_3_l1489_148959

theorem tan_20_plus_4sin_20_equals_sqrt_3 :
  Real.tan (20 * π / 180) + 4 * Real.sin (20 * π / 180) = Real.sqrt 3 := by
  sorry

end tan_20_plus_4sin_20_equals_sqrt_3_l1489_148959


namespace right_triangle_ratio_l1489_148953

theorem right_triangle_ratio (a b c x y : ℝ) : 
  a > 0 → b > 0 → c > 0 → x > 0 → y > 0 →
  c^2 = a^2 + b^2 →  -- Pythagorean theorem
  x * y = a^2 →     -- Geometric mean theorem for x
  x * y = b^2 →     -- Geometric mean theorem for y
  x + y = c →       -- x and y form the hypotenuse
  a / b = 2 / 5 →   -- Given ratio
  x / y = 4 / 25 := by sorry

end right_triangle_ratio_l1489_148953


namespace factor_implies_absolute_value_l1489_148977

theorem factor_implies_absolute_value (m n : ℤ) :
  (∀ x : ℝ, (x - 3) * (x + 4) ∣ (3 * x^3 - m * x + n)) →
  |3 * m - 2 * n| = 45 := by
  sorry

end factor_implies_absolute_value_l1489_148977


namespace injective_function_equality_l1489_148902

theorem injective_function_equality (f : ℕ → ℕ) (h_inj : Function.Injective f) 
  (h_cond : ∀ n : ℕ, f (f n) ≤ (f n + n) / 2) : 
  ∀ n : ℕ, f n = n := by
  sorry

end injective_function_equality_l1489_148902


namespace sin_seven_pi_sixths_l1489_148901

theorem sin_seven_pi_sixths : Real.sin (7 * π / 6) = -1 / 2 := by
  sorry

end sin_seven_pi_sixths_l1489_148901


namespace a_share_of_profit_l1489_148927

/-- Calculates the share of profit for a partner in a business partnership --/
def calculateShareOfProfit (investmentA investmentB investmentC totalProfit : ℕ) : ℕ :=
  let totalInvestment := investmentA + investmentB + investmentC
  (investmentA * totalProfit) / totalInvestment

/-- Theorem stating that A's share of the profit is 4260 --/
theorem a_share_of_profit :
  calculateShareOfProfit 6300 4200 10500 14200 = 4260 := by
  sorry

#eval calculateShareOfProfit 6300 4200 10500 14200

end a_share_of_profit_l1489_148927


namespace center_value_is_35_l1489_148923

/-- Represents a 4x4 array where each row and column forms an arithmetic sequence -/
def ArithmeticArray := Matrix (Fin 4) (Fin 4) ℝ

/-- Checks if a row or column is an arithmetic sequence -/
def is_arithmetic_sequence (seq : Fin 4 → ℝ) : Prop :=
  ∃ d : ℝ, ∀ i j : Fin 4, i.val < j.val → seq j - seq i = d * (j.val - i.val)

/-- Definition of our specific arithmetic array -/
def special_array (A : ArithmeticArray) : Prop :=
  (∀ i : Fin 4, is_arithmetic_sequence (λ j => A i j)) ∧ 
  (∀ j : Fin 4, is_arithmetic_sequence (λ i => A i j)) ∧
  A 0 0 = 3 ∧ A 0 3 = 27 ∧ A 3 0 = 6 ∧ A 3 3 = 66

/-- The center value of the array -/
def center_value (A : ArithmeticArray) : ℝ := A 1 1

theorem center_value_is_35 (A : ArithmeticArray) (h : special_array A) : 
  center_value A = 35 := by
  sorry

end center_value_is_35_l1489_148923


namespace cubic_polynomial_root_l1489_148999

theorem cubic_polynomial_root (x : ℝ) : x = Real.rpow 3 (1/3) + 2 →
  x^3 - 6*x^2 + 12*x - 11 = 0 := by sorry

#check cubic_polynomial_root

end cubic_polynomial_root_l1489_148999


namespace min_value_expression_l1489_148921

theorem min_value_expression (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h_abc : a * b * c = 3) :
  a^2 + 8*a*b + 32*b^2 + 24*b*c + 8*c^2 ≥ 72 ∧
  ∃ (a₀ b₀ c₀ : ℝ), 0 < a₀ ∧ 0 < b₀ ∧ 0 < c₀ ∧ a₀ * b₀ * c₀ = 3 ∧
    a₀^2 + 8*a₀*b₀ + 32*b₀^2 + 24*b₀*c₀ + 8*c₀^2 = 72 :=
by sorry

end min_value_expression_l1489_148921


namespace angle_bisector_product_not_unique_l1489_148946

/-- A triangle represented by its three side lengths -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  positive_a : 0 < a
  positive_b : 0 < b
  positive_c : 0 < c
  triangle_inequality : a < b + c ∧ b < a + c ∧ c < a + b

/-- The product of the lengths of the three angle bisectors of a triangle -/
def angle_bisector_product (t : Triangle) : ℝ := sorry

/-- Statement: The product of the three angle bisectors does not uniquely determine a triangle -/
theorem angle_bisector_product_not_unique :
  ∃ (t1 t2 : Triangle), t1 ≠ t2 ∧ angle_bisector_product t1 = angle_bisector_product t2 :=
sorry

end angle_bisector_product_not_unique_l1489_148946


namespace trigonometric_sum_equality_l1489_148931

theorem trigonometric_sum_equality (θ φ : Real) 
  (h : (Real.cos θ)^6 / (Real.cos φ)^2 + (Real.sin θ)^6 / (Real.sin φ)^2 = 1) :
  ∃ (x : Real), x = (Real.sin φ)^6 / (Real.sin θ)^2 + (Real.cos φ)^6 / (Real.cos θ)^2 ∧ 
  (∀ (y : Real), y = (Real.sin φ)^6 / (Real.sin θ)^2 + (Real.cos φ)^6 / (Real.cos θ)^2 → y ≤ x) ∧
  x = 1 :=
by sorry

end trigonometric_sum_equality_l1489_148931


namespace double_magic_result_l1489_148994

/-- Magic box function that takes two rational numbers and produces a new rational number -/
def magic_box (a b : ℚ) : ℚ := a^2 + b + 1

/-- The result of applying the magic box function twice -/
def double_magic (a b c : ℚ) : ℚ :=
  let m := magic_box a b
  magic_box m c

/-- Theorem stating that the double application of the magic box function
    with inputs (-2, 3) and then (m, 1) results in 66 -/
theorem double_magic_result : double_magic (-2) 3 1 = 66 := by
  sorry

end double_magic_result_l1489_148994


namespace arccos_one_equals_zero_l1489_148987

theorem arccos_one_equals_zero : Real.arccos 1 = 0 := by
  sorry

end arccos_one_equals_zero_l1489_148987


namespace max_lambda_inequality_l1489_148972

theorem max_lambda_inequality (a b x y : ℝ) 
  (h_nonneg_a : a ≥ 0) (h_nonneg_b : b ≥ 0) (h_nonneg_x : x ≥ 0) (h_nonneg_y : y ≥ 0)
  (h_sum : a + b = 27) :
  (a * x^2 + b * y^2 + 4 * x * y)^3 ≥ 2916 * (a * x^2 * y + b * x * y^2)^2 := by
  sorry

end max_lambda_inequality_l1489_148972


namespace evaluate_expression_l1489_148940

theorem evaluate_expression : (3200 - 3131)^2 / 121 = 36 := by
  sorry

end evaluate_expression_l1489_148940


namespace parabola_intersection_l1489_148936

-- Define the two parabola functions
def f (x : ℝ) : ℝ := 3 * x^2 - 15 * x - 4
def g (x : ℝ) : ℝ := 2 * x^2 - 8 * x + 8

-- Define the intersection points
def intersection_points : Set (ℝ × ℝ) := {(3, -22), (4, -16)}

-- Theorem statement
theorem parabola_intersection :
  ∀ x y : ℝ, f x = g x ∧ y = f x ↔ (x, y) ∈ intersection_points :=
sorry

end parabola_intersection_l1489_148936


namespace polynomial_factorization_l1489_148984

theorem polynomial_factorization (x : ℝ) : x^3 + 2*x^2 - 3*x = x*(x+3)*(x-1) := by
  sorry

end polynomial_factorization_l1489_148984


namespace forgetful_scientist_rain_probability_l1489_148961

/-- The probability of taking an umbrella -/
def umbrella_probability : ℝ := 0.2

/-- The Forgetful Scientist scenario -/
structure ForgetfulScientist where
  /-- The probability of rain -/
  rain_prob : ℝ
  /-- The probability of having no umbrella at the destination -/
  no_umbrella_prob : ℝ
  /-- The condition that the Scientist takes an umbrella if it's raining or there's no umbrella -/
  umbrella_condition : umbrella_probability = rain_prob + no_umbrella_prob - rain_prob * no_umbrella_prob
  /-- The condition that the probabilities are between 0 and 1 -/
  prob_bounds : 0 ≤ rain_prob ∧ rain_prob ≤ 1 ∧ 0 ≤ no_umbrella_prob ∧ no_umbrella_prob ≤ 1

/-- The theorem stating that the probability of rain is 1/9 -/
theorem forgetful_scientist_rain_probability (fs : ForgetfulScientist) : fs.rain_prob = 1/9 := by
  sorry

end forgetful_scientist_rain_probability_l1489_148961


namespace inequalities_proof_l1489_148937

theorem inequalities_proof (a b : ℝ) 
  (ha : a > 0) (hb : b > 0) (hab : a + b = 4) : 
  (1 / a + 1 / (b + 1) ≥ 4 / 5) ∧ 
  (4 / (a * b) + a / b ≥ (Real.sqrt 5 + 1) / 2) := by
  sorry

end inequalities_proof_l1489_148937


namespace inequality_solution_l1489_148911

theorem inequality_solution (x : ℝ) : (x + 10) / (x^2 + 2*x + 5) ≥ 0 ↔ x ≥ -10 := by
  sorry

end inequality_solution_l1489_148911


namespace smallest_value_complex_expression_l1489_148926

theorem smallest_value_complex_expression (a b c : ℤ) (ω : ℂ) 
  (h_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c)
  (h_omega_fourth : ω^4 = 1)
  (h_omega_not_one : ω ≠ 1) :
  ∃ (min_val : ℝ), 
    (∀ (x y z : ℤ) (h_xyz_distinct : x ≠ y ∧ y ≠ z ∧ x ≠ z), 
      Complex.abs (x + y * ω + z * ω^3) ≥ min_val) ∧
    (∃ (p q r : ℤ) (h_pqr_distinct : p ≠ q ∧ q ≠ r ∧ p ≠ r), 
      Complex.abs (p + q * ω + r * ω^3) = min_val) ∧
    min_val = Real.sqrt 3 :=
by sorry

end smallest_value_complex_expression_l1489_148926


namespace problem_solution_l1489_148948

theorem problem_solution (a b c x y z : ℝ) 
  (pos_a : 0 < a) (pos_b : 0 < b) (pos_c : 0 < c) 
  (pos_x : 0 < x) (pos_y : 0 < y) (pos_z : 0 < z)
  (sum_abc : a^2 + b^2 + c^2 = 49)
  (sum_xyz : x^2 + y^2 + z^2 = 64)
  (sum_prod : a*x + b*y + c*z = 56) :
  (a + b + c) / (x + y + z) = 7/8 := by
sorry

end problem_solution_l1489_148948


namespace mikes_apples_l1489_148943

theorem mikes_apples (nancy_apples keith_ate_apples apples_left : ℝ) 
  (h1 : nancy_apples = 3.0)
  (h2 : keith_ate_apples = 6.0)
  (h3 : apples_left = 4.0) :
  ∃ mike_apples : ℝ, mike_apples = 7.0 ∧ mike_apples + nancy_apples - keith_ate_apples = apples_left :=
by sorry

end mikes_apples_l1489_148943


namespace problem_statement_l1489_148955

theorem problem_statement (a b : ℝ) (h1 : 0 < a) (h2 : a < b) (h3 : a + b = 2) :
  (1 < b ∧ b < 2) ∧ a * b < 1 := by sorry

end problem_statement_l1489_148955


namespace limit_f_difference_quotient_l1489_148990

def f (x : ℝ) : ℝ := x^2 - 3*x

theorem limit_f_difference_quotient :
  ∀ ε > 0, ∃ δ > 0, ∀ t : ℝ, 0 < |t| ∧ |t| < δ →
    |(f 2 - f (2 - 3*t)) / t + 2| < ε :=
sorry

end limit_f_difference_quotient_l1489_148990


namespace conference_attendance_l1489_148924

/-- The number of writers at the conference -/
def writers : ℕ := 45

/-- The number of editors at the conference -/
def editors : ℕ := 37

/-- The number of people who are both writers and editors -/
def both : ℕ := 18

/-- The number of people who are neither writers nor editors -/
def neither : ℕ := 2 * both

/-- The total number of people attending the conference -/
def total : ℕ := writers + editors - both + neither

theorem conference_attendance :
  editors > 36 ∧ both ≤ 18 → total = 100 := by sorry

end conference_attendance_l1489_148924


namespace four_thirds_of_twelve_fifths_l1489_148907

theorem four_thirds_of_twelve_fifths :
  (4 : ℚ) / 3 * (12 : ℚ) / 5 = (16 : ℚ) / 5 := by
  sorry

end four_thirds_of_twelve_fifths_l1489_148907


namespace initial_average_runs_l1489_148905

theorem initial_average_runs (initial_matches : ℕ) (additional_runs : ℕ) (average_increase : ℕ) : 
  initial_matches = 10 →
  additional_runs = 89 →
  average_increase = 5 →
  ∃ (initial_average : ℕ),
    (initial_matches * initial_average + additional_runs) / (initial_matches + 1) = initial_average + average_increase ∧
    initial_average = 34 :=
by sorry

end initial_average_runs_l1489_148905


namespace sum_of_digits_8_to_1002_l1489_148916

theorem sum_of_digits_8_to_1002 :
  let n := 8^1002
  let tens_digit := (n / 10) % 10
  let units_digit := n % 10
  tens_digit + units_digit = 10 := by
  sorry

end sum_of_digits_8_to_1002_l1489_148916


namespace markers_problem_l1489_148997

/-- Given the initial number of markers, the number of markers in each new box,
    and the final number of markers, prove that the number of new boxes bought is 6. -/
theorem markers_problem (initial_markers final_markers markers_per_box : ℕ)
  (h1 : initial_markers = 32)
  (h2 : final_markers = 86)
  (h3 : markers_per_box = 9) :
  (final_markers - initial_markers) / markers_per_box = 6 := by
  sorry

end markers_problem_l1489_148997


namespace right_triangle_acute_angles_l1489_148944

theorem right_triangle_acute_angles (α β : ℝ) : 
  α > 0 ∧ β > 0 ∧  -- Acute angles are positive
  α + β = 90 ∧     -- Sum of acute angles in a right triangle is 90°
  α = 4 * β →      -- Ratio of acute angles is 4:1
  (min α β = 18 ∧ max α β = 72) := by
sorry

end right_triangle_acute_angles_l1489_148944


namespace max_inscribed_sphere_volume_l1489_148919

theorem max_inscribed_sphere_volume (cone_base_diameter : ℝ) (cone_volume : ℝ) 
  (h_diameter : cone_base_diameter = 12)
  (h_volume : cone_volume = 96 * Real.pi) : 
  let cone_radius : ℝ := cone_base_diameter / 2
  let cone_height : ℝ := 3 * cone_volume / (Real.pi * cone_radius^2)
  let cone_slant_height : ℝ := Real.sqrt (cone_radius^2 + cone_height^2)
  let sphere_radius : ℝ := cone_radius * cone_height / (cone_radius + cone_height + cone_slant_height)
  let sphere_volume : ℝ := 4 / 3 * Real.pi * sphere_radius^3
  sphere_volume = 36 * Real.pi := by
sorry

end max_inscribed_sphere_volume_l1489_148919


namespace coffee_shop_sales_l1489_148974

theorem coffee_shop_sales (coffee_customers : ℕ) (coffee_price : ℕ) 
  (tea_customers : ℕ) (tea_price : ℕ) : 
  coffee_customers = 7 → 
  coffee_price = 5 → 
  tea_customers = 8 → 
  tea_price = 4 → 
  coffee_customers * coffee_price + tea_customers * tea_price = 67 := by
sorry

end coffee_shop_sales_l1489_148974
