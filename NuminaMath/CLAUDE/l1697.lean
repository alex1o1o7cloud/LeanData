import Mathlib

namespace multiply_31_15_by_4_l1697_169742

-- Define a type for angles in degrees and minutes
structure Angle :=
  (degrees : ℕ)
  (minutes : ℕ)

-- Define multiplication of an angle by a natural number
def multiplyAngle (a : Angle) (n : ℕ) : Angle :=
  let totalMinutes := (a.degrees * 60 + a.minutes) * n
  { degrees := totalMinutes / 60,
    minutes := totalMinutes % 60 }

-- Theorem statement
theorem multiply_31_15_by_4 :
  let initial_angle : Angle := { degrees := 31, minutes := 15 }
  let result := multiplyAngle initial_angle 4
  result.degrees = 125 ∧ result.minutes = 0 :=
by sorry

end multiply_31_15_by_4_l1697_169742


namespace print_shop_charge_l1697_169741

/-- The charge per color copy at print shop X -/
def charge_X : ℝ := 1.20

/-- The number of copies -/
def num_copies : ℕ := 40

/-- The additional charge at print shop Y compared to print shop X for 40 copies -/
def additional_charge : ℝ := 20

/-- The charge per color copy at print shop Y -/
def charge_Y : ℝ := 1.70

theorem print_shop_charge :
  charge_Y * num_copies = charge_X * num_copies + additional_charge :=
by sorry

end print_shop_charge_l1697_169741


namespace count_perfect_squares_l1697_169738

theorem count_perfect_squares (max_value : Nat) (divisor : Nat) : 
  (Finset.filter (fun n : Nat => 
    n^2 % divisor = 0 ∧ n^2 < max_value) 
    (Finset.range max_value)).card = 175 :=
by sorry

#check count_perfect_squares (4 * 10^7) 36

end count_perfect_squares_l1697_169738


namespace balance_equation_l1697_169793

/-- The balance on a fuel card after refueling -/
def balance (initial_balance : ℝ) (price_per_liter : ℝ) (liters_refueled : ℝ) : ℝ :=
  initial_balance - price_per_liter * liters_refueled

/-- Theorem stating the functional relationship between balance and liters refueled -/
theorem balance_equation (x y : ℝ) :
  let initial_balance : ℝ := 1000
  let price_per_liter : ℝ := 7.92
  y = balance initial_balance price_per_liter x →
  y = 1000 - 7.92 * x :=
by sorry

end balance_equation_l1697_169793


namespace g_inv_composition_l1697_169759

/-- Function g defined on a finite domain -/
def g : Fin 5 → Fin 5
| 0 => 3  -- represents g(1) = 4
| 1 => 4  -- represents g(2) = 5
| 2 => 1  -- represents g(3) = 2
| 3 => 2  -- represents g(4) = 3
| 4 => 0  -- represents g(5) = 1

/-- g is bijective -/
axiom g_bijective : Function.Bijective g

/-- The inverse function of g -/
noncomputable def g_inv : Fin 5 → Fin 5 := Function.invFun g

/-- Theorem stating that g^(-1)(g^(-1)(g^(-1)(4))) = 2 -/
theorem g_inv_composition :
  g_inv (g_inv (g_inv 3)) = 1 := by sorry

end g_inv_composition_l1697_169759


namespace digit_relation_l1697_169777

theorem digit_relation (a b : ℕ) : 
  a ≤ 9 → b ≤ 9 → (a : ℚ) / b = b + (a : ℚ) / 10 → a = 5 ∧ b = 2 := by
sorry

end digit_relation_l1697_169777


namespace root_sum_reciprocal_l1697_169715

theorem root_sum_reciprocal (p q r : ℂ) : 
  (p^3 - 2*p + 2 = 0) → 
  (q^3 - 2*q + 2 = 0) → 
  (r^3 - 2*r + 2 = 0) → 
  (1/(p+2) + 1/(q+2) + 1/(r+2) = 3/5) :=
by sorry

end root_sum_reciprocal_l1697_169715


namespace quadratic_points_relation_l1697_169720

/-- Given that points A(-2,m) and B(-3,n) lie on the graph of y=(x-1)^2, prove that m < n -/
theorem quadratic_points_relation (m n : ℝ) : 
  ((-2 : ℝ) - 1)^2 = m → ((-3 : ℝ) - 1)^2 = n → m < n := by
  sorry

end quadratic_points_relation_l1697_169720


namespace division_value_proof_l1697_169794

theorem division_value_proof (x : ℝ) : (5.5 / x) * 12 = 11 → x = 6 := by
  sorry

end division_value_proof_l1697_169794


namespace binomial_26_6_l1697_169755

theorem binomial_26_6 (h1 : Nat.choose 23 5 = 33649)
                      (h2 : Nat.choose 23 6 = 42504)
                      (h3 : Nat.choose 23 7 = 53130) :
  Nat.choose 26 6 = 290444 := by
  sorry

end binomial_26_6_l1697_169755


namespace cubic_inequality_and_sum_inequality_l1697_169789

theorem cubic_inequality_and_sum_inequality :
  (∀ x : ℝ, x > 0 → x^3 - 3*x ≥ -2) ∧
  (∀ x y z : ℝ, x > 0 → y > 0 → z > 0 →
    x^2*y/z + y^2*z/x + z^2*x/y + 2*(y/(x*z) + z/(x*y) + x/(y*z)) ≥ 9) :=
by sorry

end cubic_inequality_and_sum_inequality_l1697_169789


namespace infinitely_many_N_with_same_digit_sum_l1697_169787

/-- Sum of digits of a natural number -/
def sumOfDigits (n : ℕ) : ℕ := sorry

/-- Check if a natural number is composed of digits 1 to 9 only -/
def isComposedOf1to9 (n : ℕ) : Prop := sorry

/-- The main theorem -/
theorem infinitely_many_N_with_same_digit_sum (A : ℕ) :
  ∃ f : ℕ → ℕ, Monotone f ∧ (∀ m : ℕ, 
    isComposedOf1to9 (f m) ∧ 
    sumOfDigits (f m) = sumOfDigits (A * f m)) :=
sorry

end infinitely_many_N_with_same_digit_sum_l1697_169787


namespace porter_previous_painting_price_l1697_169740

/-- The amount Porter made for his previous painting, in dollars. -/
def previous_painting_price : ℕ := sorry

/-- The amount Porter made for his recent painting, in dollars. -/
def recent_painting_price : ℕ := 44000

/-- The relationship between the prices of the two paintings. -/
axiom price_relation : recent_painting_price = 5 * previous_painting_price - 1000

theorem porter_previous_painting_price :
  previous_painting_price = 9000 := by sorry

end porter_previous_painting_price_l1697_169740


namespace shopkeeper_gain_percentage_l1697_169717

/-- Calculates the gain percentage given the cost price and selling price -/
def gain_percentage (cost_price selling_price : ℚ) : ℚ :=
  (selling_price - cost_price) / cost_price * 100

/-- Theorem stating that for an article with a cost price of 220 and selling price of 264,
    the gain percentage is 20% -/
theorem shopkeeper_gain_percentage :
  let cost_price : ℚ := 220
  let selling_price : ℚ := 264
  gain_percentage cost_price selling_price = 20 := by sorry

end shopkeeper_gain_percentage_l1697_169717


namespace imaginary_part_of_z_l1697_169778

theorem imaginary_part_of_z (i : ℂ) (h : i * i = -1) :
  let z : ℂ := i / (1 - i)
  (z.im : ℝ) = (1 : ℝ) / 2 := by sorry

end imaginary_part_of_z_l1697_169778


namespace die_product_divisible_by_48_l1697_169724

def die_numbers : Finset ℕ := {1, 2, 3, 4, 5, 6, 7, 8}

theorem die_product_divisible_by_48 (S : Finset ℕ) (h : S ⊆ die_numbers) (h_card : S.card = 7) :
  48 ∣ S.prod id :=
sorry

end die_product_divisible_by_48_l1697_169724


namespace average_of_three_numbers_l1697_169752

theorem average_of_three_numbers (y : ℝ) : (15 + 24 + y) / 3 = 23 → y = 30 := by
  sorry

end average_of_three_numbers_l1697_169752


namespace opposite_of_neg_abs_l1697_169739

theorem opposite_of_neg_abs : -(-(|5 - 6|)) = 1 := by
  sorry

end opposite_of_neg_abs_l1697_169739


namespace independence_test_checks_categorical_variables_l1697_169756

/-- An independence test is a statistical method used to check relationships between variables. -/
def independence_test : Type := sorry

/-- Categorical variables are a type of variable in statistics. -/
def categorical_variable : Type := sorry

/-- The relationship between variables that an independence test checks. -/
def relationship_checked_by_independence_test : Type := sorry

/-- Theorem stating that independence tests check relationships between categorical variables. -/
theorem independence_test_checks_categorical_variables :
  relationship_checked_by_independence_test = categorical_variable := by sorry

end independence_test_checks_categorical_variables_l1697_169756


namespace four_tangent_circles_l1697_169713

/-- Represents a circle in a plane --/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Checks if two circles are tangent to each other --/
def are_tangent (c1 c2 : Circle) : Prop :=
  let (x1, y1) := c1.center
  let (x2, y2) := c2.center
  (x2 - x1)^2 + (y2 - y1)^2 = (c1.radius + c2.radius)^2

/-- Counts the number of circles with radius 5 tangent to both given circles --/
def count_tangent_circles (c1 c2 : Circle) : ℕ :=
  sorry

theorem four_tangent_circles (c1 c2 : Circle) :
  c1.radius = 2 →
  c2.radius = 2 →
  are_tangent c1 c2 →
  count_tangent_circles c1 c2 = 4 :=
by sorry

end four_tangent_circles_l1697_169713


namespace second_smallest_hot_dog_packs_l1697_169709

theorem second_smallest_hot_dog_packs : 
  (∃ n : ℕ, n > 0 ∧ 12 * n % 8 = 6 ∧ 
   (∀ m : ℕ, m > 0 ∧ 12 * m % 8 = 6 → m ≥ n) ∧
   (∃ k : ℕ, k > 0 ∧ 12 * k % 8 = 6 ∧ k < n)) → 
  (∃ n : ℕ, n = 4 ∧ 12 * n % 8 = 6 ∧ 
   (∀ m : ℕ, m > 0 ∧ 12 * m % 8 = 6 → m = n ∨ m > n) ∧
   (∃ k : ℕ, k > 0 ∧ 12 * k % 8 = 6 ∧ k < n)) :=
by sorry

end second_smallest_hot_dog_packs_l1697_169709


namespace tournament_games_l1697_169776

/-- The number of games needed in a single-elimination tournament -/
def games_in_tournament (n : ℕ) : ℕ := n - 1

/-- Theorem: A single-elimination tournament with 32 teams requires 31 games -/
theorem tournament_games : games_in_tournament 32 = 31 := by
  sorry

end tournament_games_l1697_169776


namespace quadratic_sum_zero_l1697_169702

def quadratic (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

theorem quadratic_sum_zero
  (a b c : ℝ)
  (h1 : ∃ (x : ℝ), ∀ (y : ℝ), quadratic a b c y ≥ quadratic a b c x)
  (h2 : quadratic a b c 1 = 0)
  (h3 : quadratic a b c (-3) = 0)
  (h4 : ∃ (x : ℝ), quadratic a b c x = 45)
  : a + b + c = 0 := by
  sorry

end quadratic_sum_zero_l1697_169702


namespace base_three_to_base_ten_l1697_169722

/-- Converts a list of digits in base b to a natural number -/
def toBase10 (digits : List Nat) (b : Nat) : Nat :=
  digits.foldr (fun d acc => d + b * acc) 0

/-- The digits of the number in base 3 -/
def baseThreeDigits : List Nat := [1, 0, 2, 0, 1, 2]

theorem base_three_to_base_ten :
  toBase10 baseThreeDigits 3 = 302 := by
  sorry

end base_three_to_base_ten_l1697_169722


namespace book_lending_solution_l1697_169781

/-- Represents the book lending problem with three people. -/
structure BookLending where
  xiaoqiang : ℕ  -- Initial number of books Xiaoqiang has
  feifei : ℕ     -- Initial number of books Feifei has
  xiaojing : ℕ   -- Initial number of books Xiaojing has

/-- The book lending problem satisfies the given conditions. -/
def satisfiesConditions (b : BookLending) : Prop :=
  b.xiaoqiang - 20 + 10 = 35 ∧
  b.feifei + 20 - 15 = 35 ∧
  b.xiaojing + 15 - 10 = 35

/-- The theorem stating the solution to the book lending problem. -/
theorem book_lending_solution :
  ∃ (b : BookLending), satisfiesConditions b ∧ b.xiaoqiang = 45 ∧ b.feifei = 30 ∧ b.xiaojing = 30 := by
  sorry

end book_lending_solution_l1697_169781


namespace pythagorean_triple_divisibility_l1697_169799

theorem pythagorean_triple_divisibility (x y z : ℕ+) (h : x^2 + y^2 = z^2) :
  (3 ∣ x ∨ 3 ∣ y) ∧ (5 ∣ x ∨ 5 ∣ y ∨ 5 ∣ z) := by
  sorry

end pythagorean_triple_divisibility_l1697_169799


namespace seeking_cause_is_sufficient_condition_l1697_169703

/-- The analysis method for proving inequalities -/
structure AnalysisMethod where
  inequality : Prop
  condition : Prop

/-- Definition of a sufficient condition -/
def is_sufficient_condition (am : AnalysisMethod) : Prop :=
  am.condition → am.inequality

/-- "Seeking the cause from the result" in the analysis method -/
def seeking_cause_from_result (am : AnalysisMethod) : Prop :=
  ∃ (condition : Prop), is_sufficient_condition { inequality := am.inequality, condition := condition }

theorem seeking_cause_is_sufficient_condition (am : AnalysisMethod) :
  seeking_cause_from_result am ↔ ∃ (condition : Prop), is_sufficient_condition { inequality := am.inequality, condition := condition } :=
sorry

end seeking_cause_is_sufficient_condition_l1697_169703


namespace line_equations_l1697_169798

/-- Line represented by ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Point in 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Check if a point lies on a line -/
def Line.contains (l : Line) (p : Point) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- Check if two lines are parallel -/
def Line.parallel (l₁ l₂ : Line) : Prop :=
  l₁.a * l₂.b = l₁.b * l₂.a

/-- Check if two lines are perpendicular -/
def Line.perpendicular (l₁ l₂ : Line) : Prop :=
  l₁.a * l₂.a + l₁.b * l₂.b = 0

theorem line_equations (l₁ : Line) (p : Point) :
  l₁.a = 2 ∧ l₁.b = 4 ∧ l₁.c = -1 ∧ p.x = 1 ∧ p.y = -2 →
  (∃ l₂ : Line, l₂.parallel l₁ ∧ l₂.contains p ∧ l₂.a = 1 ∧ l₂.b = 2 ∧ l₂.c = 3) ∧
  (∃ l₂ : Line, l₂.perpendicular l₁ ∧ l₂.contains p ∧ l₂.a = 2 ∧ l₂.b = -1 ∧ l₂.c = -4) := by
  sorry

end line_equations_l1697_169798


namespace coefficient_x3y3_in_x_plus_y_power_6_l1697_169754

theorem coefficient_x3y3_in_x_plus_y_power_6 :
  Nat.choose 6 3 = 20 := by sorry

end coefficient_x3y3_in_x_plus_y_power_6_l1697_169754


namespace total_amount_after_two_years_l1697_169788

/-- Calculates the total amount after compound interest --/
def totalAmount (P : ℝ) (r : ℝ) (t : ℝ) (CI : ℝ) : ℝ :=
  P + CI

/-- Theorem: Given the conditions, the total amount after 2 years is 4326.40 --/
theorem total_amount_after_two_years :
  ∃ (P : ℝ), 
    let r : ℝ := 0.04
    let t : ℝ := 2
    let CI : ℝ := 326.40
    totalAmount P r t CI = 4326.40 :=
by
  sorry


end total_amount_after_two_years_l1697_169788


namespace no_double_factorial_sum_l1697_169770

theorem no_double_factorial_sum : ¬∃ (z : ℤ) (x₁ y₁ x₂ y₂ : ℕ),
  x₁ ≤ y₁ ∧ x₂ ≤ y₂ ∧ 
  (z : ℤ) = x₁.factorial + y₁.factorial ∧
  (z : ℤ) = x₂.factorial + y₂.factorial ∧
  (x₁ ≠ x₂ ∨ y₁ ≠ y₂) :=
sorry

end no_double_factorial_sum_l1697_169770


namespace marble_drawing_probability_l1697_169700

theorem marble_drawing_probability : 
  let total_marbles : ℕ := 10
  let blue_marbles : ℕ := 4
  let green_marbles : ℕ := 6
  let prob_blue : ℚ := blue_marbles / total_marbles
  let prob_green_after_blue : ℚ := green_marbles / (total_marbles - 1)
  let prob_green_after_blue_green : ℚ := (green_marbles - 1) / (total_marbles - 2)
  prob_blue * prob_green_after_blue * prob_green_after_blue_green = 1 / 6 :=
by sorry

end marble_drawing_probability_l1697_169700


namespace fraction_inequality_l1697_169725

theorem fraction_inequality (x : ℝ) : 
  x ∈ Set.Icc (-2 : ℝ) 2 →
  (8 * x - 3 > 2 + 5 * x ↔ 5 / 3 < x ∧ x ≤ 2) := by
  sorry

end fraction_inequality_l1697_169725


namespace intersection_of_A_and_B_l1697_169733

-- Define the sets A and B
def A : Set ℝ := {x | ∃ y, y = Real.sqrt (x + 1)}
def B : Set ℝ := {y | ∃ x, y = x^2 + 1}

-- State the theorem
theorem intersection_of_A_and_B : A ∩ B = Set.Ici 1 := by sorry

end intersection_of_A_and_B_l1697_169733


namespace sally_pokemon_cards_l1697_169749

theorem sally_pokemon_cards (x : ℕ) : 
  x + 41 - 20 = 48 → x = 27 := by
sorry

end sally_pokemon_cards_l1697_169749


namespace bottle_cap_groups_l1697_169786

theorem bottle_cap_groups (total_caps : ℕ) (caps_per_group : ℕ) (num_groups : ℕ) : 
  total_caps = 35 → caps_per_group = 5 → num_groups = total_caps / caps_per_group → num_groups = 7 := by
  sorry

end bottle_cap_groups_l1697_169786


namespace perimeter_increase_first_to_fourth_l1697_169766

/-- Calculates the perimeter of an equilateral triangle given its side length -/
def trianglePerimeter (side : ℝ) : ℝ := 3 * side

/-- Calculates the side length of the nth triangle in the sequence -/
def nthTriangleSide (n : ℕ) : ℝ :=
  3 * (1.6 ^ n)

/-- Theorem stating the percent increase in perimeter from the first to the fourth triangle -/
theorem perimeter_increase_first_to_fourth :
  let first_perimeter := trianglePerimeter 3
  let fourth_perimeter := trianglePerimeter (nthTriangleSide 3)
  (fourth_perimeter - first_perimeter) / first_perimeter * 100 = 309.6 := by
  sorry

end perimeter_increase_first_to_fourth_l1697_169766


namespace cone_height_from_semicircle_l1697_169771

/-- The distance from the highest point of a tipped-over cone to the table,
    where the cone is formed by rolling a semicircular paper. -/
theorem cone_height_from_semicircle (R : ℝ) (h : R = 4) : 
  let r := R / 2
  let h := Real.sqrt (R^2 - r^2)
  2 * (h * r / R) = 2 * Real.sqrt 3 :=
by sorry

end cone_height_from_semicircle_l1697_169771


namespace seed_buckets_l1697_169721

theorem seed_buckets (total : ℕ) (seeds_B : ℕ) (diff_A_B : ℕ) : 
  total = 100 → 
  seeds_B = 30 → 
  diff_A_B = 10 → 
  total - (seeds_B + diff_A_B) - seeds_B = 30 := by sorry

end seed_buckets_l1697_169721


namespace sabrina_profit_is_35_l1697_169797

def sabrina_profit (total_loaves : ℕ) (morning_price : ℚ) (afternoon_price_ratio : ℚ) (evening_price : ℚ) (production_cost : ℚ) : ℚ :=
  let morning_loaves : ℕ := (2 * total_loaves) / 3
  let morning_revenue : ℚ := morning_loaves * morning_price
  let afternoon_loaves : ℕ := (total_loaves - morning_loaves) / 2
  let afternoon_revenue : ℚ := afternoon_loaves * (afternoon_price_ratio * morning_price)
  let evening_loaves : ℕ := total_loaves - morning_loaves - afternoon_loaves
  let evening_revenue : ℚ := evening_loaves * evening_price
  let total_revenue : ℚ := morning_revenue + afternoon_revenue + evening_revenue
  let total_cost : ℚ := total_loaves * production_cost
  total_revenue - total_cost

theorem sabrina_profit_is_35 :
  sabrina_profit 60 2 (1/4) 1 1 = 35 := by
  sorry

end sabrina_profit_is_35_l1697_169797


namespace total_pay_is_880_l1697_169701

/-- The total amount paid to two employees, where one is paid 120% of the other's wage -/
def total_pay (y_pay : ℝ) : ℝ :=
  y_pay + 1.2 * y_pay

/-- Theorem stating that the total pay for two employees is 880 when one is paid 400 and the other 120% of that -/
theorem total_pay_is_880 :
  total_pay 400 = 880 := by
  sorry

end total_pay_is_880_l1697_169701


namespace coefficient_x_squared_l1697_169784

theorem coefficient_x_squared (p q : Polynomial ℤ) (hp : p = 3 * X^2 + 4 * X + 5) (hq : q = 6 * X^2 + 7 * X + 8) :
  (p * q).coeff 2 = 82 := by
  sorry

end coefficient_x_squared_l1697_169784


namespace tea_cost_theorem_l1697_169732

/-- Represents the cost calculation for tea sets and cups under different options -/
def tea_cost (x : ℕ) : Prop :=
  let tea_set_price : ℕ := 200
  let tea_cup_price : ℕ := 20
  let option1_cost : ℕ := 20 * x + 5400
  let option2_cost : ℕ := 19 * x + 5700
  (x > 30) →
  (option1_cost = 30 * tea_set_price + tea_cup_price * (x - 30)) ∧
  (option2_cost = (30 * tea_set_price + x * tea_cup_price) * 95 / 100) ∧
  (x = 50 → option1_cost < option2_cost)

theorem tea_cost_theorem :
  ∀ x : ℕ, tea_cost x :=
sorry

end tea_cost_theorem_l1697_169732


namespace chichikov_dead_souls_l1697_169710

theorem chichikov_dead_souls (x y z : ℕ) (h : x + y + z = 1001) :
  ∃ N : ℕ, N ≤ 1001 ∧
  (∀ w : ℕ, w + min x N + min y N + min z N + min (x + y) N + min (y + z) N + min (x + z) N < N →
   w ≥ 71) :=
sorry

end chichikov_dead_souls_l1697_169710


namespace parabola_properties_l1697_169743

/-- Represents a parabola with focus at distance 2 from directrix -/
structure Parabola where
  p : ℝ
  h_p_pos : p > 0
  h_focus_dist : p = 2

/-- Point on the parabola -/
structure ParabolaPoint (c : Parabola) where
  x : ℝ
  y : ℝ
  h_on_parabola : y^2 = 2 * c.p * x

/-- Point Q satisfying PQ = 9QF -/
structure PointQ (c : Parabola) (p : ParabolaPoint c) where
  x : ℝ
  y : ℝ
  h_relation : (x - p.x)^2 + (y - p.y)^2 = 81 * ((x - c.p)^2 + y^2)

/-- The theorem to be proved -/
theorem parabola_properties (c : Parabola) :
  (∀ (x y : ℝ), y^2 = 2 * c.p * x ↔ y^2 = 4 * x) ∧
  (∀ (p : ParabolaPoint c) (q : PointQ c p),
    ∀ (slope : ℝ), slope = q.y / q.x → slope ≤ 1/3) :=
sorry

end parabola_properties_l1697_169743


namespace ball_max_height_l1697_169791

/-- The height function of the ball -/
def h (t : ℝ) : ℝ := -5 * t^2 + 20 * t + 10

/-- Theorem stating that the maximum height reached by the ball is 30 meters -/
theorem ball_max_height :
  ∃ (t_max : ℝ), ∀ (t : ℝ), h t ≤ h t_max ∧ h t_max = 30 :=
sorry

end ball_max_height_l1697_169791


namespace total_snakes_in_neighborhood_l1697_169753

theorem total_snakes_in_neighborhood (total_people : ℕ) 
  (only_dogs only_cats only_snakes only_rabbits only_birds : ℕ)
  (dogs_and_cats dogs_and_snakes dogs_and_rabbits dogs_and_birds : ℕ)
  (cats_and_snakes cats_and_rabbits cats_and_birds : ℕ)
  (snakes_and_rabbits snakes_and_birds rabbits_and_birds : ℕ)
  (dogs_cats_snakes dogs_cats_rabbits dogs_cats_birds : ℕ)
  (dogs_snakes_rabbits cats_snakes_rabbits : ℕ)
  (all_five : ℕ) :
  total_people = 125 →
  only_dogs = 20 →
  only_cats = 15 →
  only_snakes = 8 →
  only_rabbits = 10 →
  only_birds = 5 →
  dogs_and_cats = 12 →
  dogs_and_snakes = 7 →
  dogs_and_rabbits = 4 →
  dogs_and_birds = 3 →
  cats_and_snakes = 9 →
  cats_and_rabbits = 6 →
  cats_and_birds = 2 →
  snakes_and_rabbits = 5 →
  snakes_and_birds = 3 →
  rabbits_and_birds = 1 →
  dogs_cats_snakes = 4 →
  dogs_cats_rabbits = 2 →
  dogs_cats_birds = 1 →
  dogs_snakes_rabbits = 3 →
  cats_snakes_rabbits = 2 →
  all_five = 1 →
  only_snakes + dogs_and_snakes + cats_and_snakes + snakes_and_rabbits + 
  snakes_and_birds + dogs_cats_snakes + dogs_snakes_rabbits + 
  cats_snakes_rabbits + all_five = 42 :=
by sorry


end total_snakes_in_neighborhood_l1697_169753


namespace two_xy_value_l1697_169744

theorem two_xy_value (x y : ℝ) : 
  y = Real.sqrt (2 * x - 5) + Real.sqrt (5 - 2 * x) - 3 → 2 * x * y = -15 := by
  sorry

end two_xy_value_l1697_169744


namespace ninth_grade_basketball_tournament_l1697_169772

theorem ninth_grade_basketball_tournament (n : ℕ) : 
  (n * (n - 1)) / 2 = 45 → n = 10 := by
  sorry

end ninth_grade_basketball_tournament_l1697_169772


namespace complex_multiplication_l1697_169736

theorem complex_multiplication :
  let i : ℂ := Complex.I
  (3 - 4 * i) * (-7 + 6 * i) = 3 + 46 * i :=
by sorry

end complex_multiplication_l1697_169736


namespace jimmy_crackers_needed_l1697_169728

/-- Calculates the number of crackers needed to reach a target calorie count -/
def crackers_needed (cracker_calories cookie_calories cookies_eaten target_calories : ℕ) : ℕ :=
  (target_calories - cookie_calories * cookies_eaten) / cracker_calories

/-- Proves that Jimmy needs 10 crackers to reach 500 total calories -/
theorem jimmy_crackers_needed :
  crackers_needed 15 50 7 500 = 10 := by
  sorry

end jimmy_crackers_needed_l1697_169728


namespace hotel_revenue_maximization_l1697_169765

/-- Represents the hotel revenue optimization problem -/
def HotelRevenueProblem (totalRooms : ℕ) (initialPrice : ℕ) (initialOccupancy : ℕ) 
  (priceReduction : ℕ) (occupancyIncrease : ℕ) : Prop :=
  ∃ (maxRevenue : ℕ),
    maxRevenue = 22500 ∧
    (∀ (x : ℕ),
      let newPrice := initialPrice - x * priceReduction
      let newOccupancy := initialOccupancy + x * occupancyIncrease
      newPrice * newOccupancy ≤ maxRevenue)

/-- Theorem stating that the hotel revenue problem has a solution -/
theorem hotel_revenue_maximization :
  HotelRevenueProblem 100 400 50 20 5 := by
  sorry

#check hotel_revenue_maximization

end hotel_revenue_maximization_l1697_169765


namespace f_properties_l1697_169758

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * Real.log x + (a + 1) * (1 / x - 2)

theorem f_properties :
  ∀ a : ℝ,
  (∀ x : ℝ, x > 0 → f a x = a * Real.log x + (a + 1) * (1 / x - 2)) →
  (a < -1 → 
    (∃ x₀ : ℝ, x₀ > 0 ∧ x₀ = (a + 1) / a ∧ 
      (∀ x : ℝ, x > 0 → f a x ≤ f a x₀) ∧
      (∀ ε : ℝ, ε > 0 → ∃ x : ℝ, x > 0 ∧ f a x > f a x₀ - ε))) ∧
  (-1 ≤ a ∧ a ≤ 0 → 
    (∀ x₁ x₂ : ℝ, x₁ > 0 ∧ x₂ > 0 → f a x₁ ≠ f a x₂ ∨ x₁ = x₂)) ∧
  (a > 0 → 
    (∃ x₀ : ℝ, x₀ > 0 ∧ x₀ = (a + 1) / a ∧ 
      (∀ x : ℝ, x > 0 → f a x ≥ f a x₀) ∧
      (∀ ε : ℝ, ε > 0 → ∃ x : ℝ, x > 0 ∧ f a x < f a x₀ + ε))) ∧
  (a > 0 → ∀ x : ℝ, x > 0 → f a x > -a^2 / (a + 1) - 2) :=
by sorry

end f_properties_l1697_169758


namespace april_earnings_correct_l1697_169707

/-- Calculate April's earnings from flower sales with tax --/
def april_earnings (rose_price tulip_price daisy_price : ℚ)
  (initial_roses initial_tulips initial_daisies : ℕ)
  (final_roses final_tulips final_daisies : ℕ)
  (tax_rate : ℚ) : ℚ :=
  let roses_sold := initial_roses - final_roses
  let tulips_sold := initial_tulips - final_tulips
  let daisies_sold := initial_daisies - final_daisies
  let revenue := rose_price * roses_sold + tulip_price * tulips_sold + daisy_price * daisies_sold
  let tax := revenue * tax_rate
  revenue + tax

theorem april_earnings_correct :
  april_earnings 4 3 2 13 10 8 4 3 1 (1/10) = 781/10 := by
  sorry

end april_earnings_correct_l1697_169707


namespace original_apple_price_l1697_169782

/-- The original price of apples per pound -/
def original_price : ℝ := sorry

/-- The price increase percentage -/
def price_increase : ℝ := 0.25

/-- The new price of apples per pound after the increase -/
def new_price : ℝ := original_price * (1 + price_increase)

/-- The total weight of apples bought -/
def total_weight : ℝ := 8

/-- The total cost of apples after the price increase -/
def total_cost : ℝ := 64

theorem original_apple_price :
  new_price * total_weight = total_cost →
  original_price = 6.40 := by sorry

end original_apple_price_l1697_169782


namespace linear_equation_sum_l1697_169729

theorem linear_equation_sum (m n : ℤ) : 
  (∃ a b c : ℝ, ∀ x y : ℝ, (n - 1) * x^(n^2) - 3 * y^(m - 2023) = a * x + b * y + c) → 
  m + n = 2023 := by
  sorry

end linear_equation_sum_l1697_169729


namespace quadratic_inequality_solution_set_l1697_169764

theorem quadratic_inequality_solution_set (m : ℝ) (hm : m > 1) :
  {x : ℝ | x^2 + (m - 1) * x - m ≥ 0} = {x : ℝ | x ≤ -m ∨ x ≥ 1} := by
  sorry

end quadratic_inequality_solution_set_l1697_169764


namespace tickets_at_door_correct_l1697_169718

/-- Represents the number of tickets sold at the door -/
def tickets_at_door : ℕ := 672

/-- Represents the number of advanced tickets sold -/
def advanced_tickets : ℕ := 800 - tickets_at_door

/-- The total number of tickets sold -/
def total_tickets : ℕ := 800

/-- The price of an advanced ticket in cents -/
def advanced_price : ℕ := 1450

/-- The price of a ticket at the door in cents -/
def door_price : ℕ := 2200

/-- The total amount of money taken in cents -/
def total_revenue : ℕ := 1664000

theorem tickets_at_door_correct :
  (advanced_tickets * advanced_price + tickets_at_door * door_price = total_revenue) ∧
  (advanced_tickets + tickets_at_door = total_tickets) := by
  sorry

end tickets_at_door_correct_l1697_169718


namespace contrapositive_equivalence_l1697_169769

theorem contrapositive_equivalence : 
  (∀ x : ℝ, x^2 < 4 → -2 < x ∧ x < 2) ↔ 
  (∀ x : ℝ, x ≤ -2 ∨ x ≥ 2 → x^2 ≥ 4) := by sorry

end contrapositive_equivalence_l1697_169769


namespace evaluate_expression_l1697_169783

theorem evaluate_expression : 6 - 8 * (9 - 4^2) / 2 = 34 := by
  sorry

end evaluate_expression_l1697_169783


namespace fair_die_probability_l1697_169747

/-- Probability of rolling at least a four on a fair die -/
def p : ℚ := 1/2

/-- Number of rolls -/
def n : ℕ := 8

/-- Minimum number of successful rolls required -/
def k : ℕ := 6

/-- The probability of rolling at least a four, at least six times in eight rolls of a fair die -/
theorem fair_die_probability : (Finset.range (n + 1 - k)).sum (λ i => n.choose (n - i) * p^(n - i) * (1 - p)^i) = 129/256 := by
  sorry

end fair_die_probability_l1697_169747


namespace ellipse_standard_equation_l1697_169731

/-- The standard equation of an ellipse passing through two specific points -/
theorem ellipse_standard_equation :
  ∃ (m n : ℝ), m > 0 ∧ n > 0 ∧ m ≠ n ∧
  m * (1/3)^2 + n * (1/3)^2 = 1 ∧
  n * (-1/2)^2 = 1 →
  ∀ (x y : ℝ), x^2 / (1/5) + y^2 / (1/4) = 1 ↔ m * x^2 + n * y^2 = 1 :=
by sorry

end ellipse_standard_equation_l1697_169731


namespace simplify_and_rationalize_l1697_169714

theorem simplify_and_rationalize :
  (Real.sqrt 5 / Real.sqrt 6) * (Real.sqrt 7 / Real.sqrt 8) * (Real.sqrt 9 / Real.sqrt 10) = 
  (3 * Real.sqrt 1050) / 120 := by
  sorry

end simplify_and_rationalize_l1697_169714


namespace inequality_solution_set_l1697_169779

theorem inequality_solution_set (x : ℝ) : 
  (4 ≤ x / (3 * x - 7) ∧ x / (3 * x - 7) < 9) ↔ 
  (63 / 26 < x ∧ x ≤ 28 / 11) := by sorry

end inequality_solution_set_l1697_169779


namespace susan_took_35_oranges_l1697_169706

/-- The number of oranges Susan took from the box -/
def oranges_taken (initial final : ℕ) : ℕ := initial - final

/-- Proof that Susan took 35 oranges from the box -/
theorem susan_took_35_oranges (initial final taken : ℕ) 
  (h_initial : initial = 55)
  (h_final : final = 20)
  (h_taken : taken = oranges_taken initial final) : 
  taken = 35 := by
  sorry

end susan_took_35_oranges_l1697_169706


namespace trading_card_boxes_l1697_169757

/-- Calculates the number of fully filled boxes for a card type -/
def fullBoxes (cards : ℕ) (capacity : ℕ) : ℕ := cards / capacity

/-- Represents the trading card sorting problem -/
theorem trading_card_boxes 
  (total_cards : ℕ) 
  (magic_cards : ℕ) 
  (rare_cards : ℕ) 
  (common_cards : ℕ) 
  (magic_capacity : ℕ) 
  (rare_capacity : ℕ) 
  (common_capacity : ℕ) 
  (h1 : total_cards = 94)
  (h2 : magic_cards = 33)
  (h3 : rare_cards = 28)
  (h4 : common_cards = 33)
  (h5 : magic_capacity = 8)
  (h6 : rare_capacity = 10)
  (h7 : common_capacity = 12)
  (h8 : total_cards = magic_cards + rare_cards + common_cards) :
  fullBoxes magic_cards magic_capacity + 
  fullBoxes rare_cards rare_capacity + 
  fullBoxes common_cards common_capacity = 8 := by
sorry

end trading_card_boxes_l1697_169757


namespace cricket_bat_profit_l1697_169734

theorem cricket_bat_profit (selling_price : ℝ) (profit_percentage : ℝ) : 
  selling_price = 900 ∧ profit_percentage = 50 → 
  ∃ (cost_price : ℝ) (profit : ℝ),
    profit = selling_price - cost_price ∧
    profit_percentage = (profit / cost_price) * 100 ∧
    profit = 300 :=
by sorry

end cricket_bat_profit_l1697_169734


namespace factorization_proof_l1697_169737

theorem factorization_proof (x : ℝ) : 180 * x^2 + 36 * x + 4 = 4 * (3 * x + 1) * (15 * x + 1) := by
  sorry

end factorization_proof_l1697_169737


namespace third_vertex_coordinates_l1697_169796

/-- Given a triangle with vertices (2,3), (0,0), and (0,y) where y > 0,
    if the area of the triangle is 36 square units, then y = 39 -/
theorem third_vertex_coordinates (y : ℝ) (h1 : y > 0) : 
  (1/2 : ℝ) * |2 * (3 - y)| = 36 → y = 39 := by
  sorry

end third_vertex_coordinates_l1697_169796


namespace problem_statement_l1697_169730

theorem problem_statement :
  (∀ a b : ℝ, a > 0 → b > 0 → 1 / (a + b) ≤ (1 / 4) * (1 / a + 1 / b)) ∧
  (∀ x₁ x₂ x₃ : ℝ, x₁ > 0 → x₂ > 0 → x₃ > 0 → 
    1 / x₁ + 1 / x₂ + 1 / x₃ = 1 →
    (x₁ + x₂ + x₃) / (x₁ * x₃ + x₃ * x₂) + 
    (x₁ + x₂ + x₃) / (x₁ * x₂ + x₃ * x₁) + 
    (x₁ + x₂ + x₃) / (x₂ * x₁ + x₃ * x₂) ≤ 3 / 2) := by
  sorry

end problem_statement_l1697_169730


namespace distilled_water_amount_l1697_169763

/-- Given the initial mixture ratios and the required amount of final solution,
    prove that the amount of distilled water needed is 0.2 liters. -/
theorem distilled_water_amount
  (nutrient_concentrate : ℝ)
  (initial_distilled_water : ℝ)
  (initial_total_solution : ℝ)
  (required_solution : ℝ)
  (h1 : nutrient_concentrate = 0.05)
  (h2 : initial_distilled_water = 0.025)
  (h3 : initial_total_solution = 0.075)
  (h4 : required_solution = 0.6)
  (h5 : initial_total_solution = nutrient_concentrate + initial_distilled_water) :
  (required_solution * (initial_distilled_water / initial_total_solution)) = 0.2 :=
by sorry

end distilled_water_amount_l1697_169763


namespace brennan_pepper_usage_l1697_169745

/-- The amount of pepper Brennan used to make scrambled eggs -/
def pepper_used (initial : ℝ) (remaining : ℝ) : ℝ :=
  initial - remaining

/-- Theorem stating that Brennan used 0.16 grams of pepper -/
theorem brennan_pepper_usage :
  pepper_used 0.25 0.09 = 0.16 := by
  sorry

end brennan_pepper_usage_l1697_169745


namespace point_Q_coordinates_l1697_169748

/-- Given a point P in ℝ² and a length l, this function returns the two possible
    points Q such that PQ is parallel to the x-axis and has length l -/
def possible_Q (P : ℝ × ℝ) (l : ℝ) : Set (ℝ × ℝ) :=
  {(P.1 + l, P.2), (P.1 - l, P.2)}

theorem point_Q_coordinates :
  let P : ℝ × ℝ := (2, 1)
  let l : ℝ := 3
  possible_Q P l = {(5, 1), (-1, 1)} := by
  sorry

#check point_Q_coordinates

end point_Q_coordinates_l1697_169748


namespace ball_count_proof_l1697_169792

theorem ball_count_proof (a : ℕ) (red_balls : ℕ) (probability : ℝ) 
  (h1 : red_balls = 5)
  (h2 : probability = 0.20)
  (h3 : (red_balls : ℝ) / a = probability) : 
  a = 25 := by
  sorry

end ball_count_proof_l1697_169792


namespace min_major_axis_length_l1697_169767

/-- An ellipse with the property that the maximum area of a triangle formed by 
    a point on the ellipse and its two foci is 1 -/
structure SpecialEllipse where
  /-- The semi-major axis length -/
  a : ℝ
  /-- The semi-minor axis length -/
  b : ℝ
  /-- The semi-focal distance -/
  c : ℝ
  /-- The maximum triangle area is 1 -/
  max_triangle_area : b * c = 1
  /-- Relationship between a, b, and c in an ellipse -/
  ellipse_property : a^2 = b^2 + c^2

/-- The minimum length of the major axis of a SpecialEllipse is 2√2 -/
theorem min_major_axis_length (e : SpecialEllipse) : 
  2 * e.a ≥ 2 * Real.sqrt 2 := by
  sorry

end min_major_axis_length_l1697_169767


namespace rebecca_hours_less_than_toby_l1697_169795

theorem rebecca_hours_less_than_toby (x : ℕ) : 
  x + (2 * x - 10) + 56 = 157 → 64 - 56 = 8 := by
  sorry

end rebecca_hours_less_than_toby_l1697_169795


namespace prob_at_least_7_heads_in_9_flips_is_correct_l1697_169719

/-- The probability of getting at least 7 heads in 9 flips of a fair coin -/
def prob_at_least_7_heads_in_9_flips : ℚ :=
  46 / 512

/-- Theorem stating that the probability of getting at least 7 heads in 9 flips of a fair coin is 46/512 -/
theorem prob_at_least_7_heads_in_9_flips_is_correct :
  prob_at_least_7_heads_in_9_flips = 46 / 512 := by
  sorry

end prob_at_least_7_heads_in_9_flips_is_correct_l1697_169719


namespace prairie_area_l1697_169761

/-- The total area of a prairie given the area covered by a dust storm and the area left untouched -/
theorem prairie_area (dust_covered : ℕ) (untouched : ℕ) : dust_covered = 64535 → untouched = 522 → dust_covered + untouched = 65057 := by
  sorry

#check prairie_area

end prairie_area_l1697_169761


namespace five_card_draw_probability_l1697_169773

/-- Represents a standard deck of 52 cards -/
def StandardDeck : ℕ := 52

/-- Represents the number of cards to be drawn -/
def CardsDrawn : ℕ := 5

/-- Represents the number of suits in a standard deck -/
def NumSuits : ℕ := 4

/-- Represents the number of cards in each suit -/
def CardsPerSuit : ℕ := 13

/-- The probability of drawing 5 cards from a standard 52-card deck without replacement,
    such that there is one card from each suit and the fifth card is from hearts -/
theorem five_card_draw_probability :
  (1 : ℚ) * (CardsPerSuit : ℚ) / (StandardDeck - 1 : ℚ) *
  (CardsPerSuit : ℚ) / (StandardDeck - 2 : ℚ) *
  (CardsPerSuit : ℚ) / (StandardDeck - 3 : ℚ) *
  (CardsPerSuit - 1 : ℚ) / (StandardDeck - 4 : ℚ) =
  2197 / 83300 :=
by sorry

end five_card_draw_probability_l1697_169773


namespace right_triangle_areas_l1697_169727

theorem right_triangle_areas (a b : ℝ) (ha : a = 6) (hb : b = 8) :
  let c := Real.sqrt (a^2 + b^2)
  let s := (a * b) / 2
  let s1 := (s * b^2) / c^2
  let s2 := s - s1
  (s1 = 15.36 ∧ s2 = 8.64) := by sorry

end right_triangle_areas_l1697_169727


namespace saheed_earnings_l1697_169712

theorem saheed_earnings (vika_earnings kayla_earnings saheed_earnings : ℕ) : 
  vika_earnings = 84 →
  kayla_earnings = vika_earnings - 30 →
  saheed_earnings = 4 * kayla_earnings →
  saheed_earnings = 216 := by
  sorry

end saheed_earnings_l1697_169712


namespace smallest_integer_with_remainders_l1697_169711

theorem smallest_integer_with_remainders : ∃ n : ℕ, 
  (∀ m : ℕ, m > 0 ∧ m < n → 
    (m % 6 ≠ 3 ∨ m % 8 ≠ 5 ∨ m % 9 ≠ 2)) ∧
  n % 6 = 3 ∧ 
  n % 8 = 5 ∧ 
  n % 9 = 2 ∧
  n = 237 :=
by sorry

end smallest_integer_with_remainders_l1697_169711


namespace toy_store_revenue_ratio_l1697_169726

theorem toy_store_revenue_ratio :
  ∀ (november december january : ℝ),
  january = (1 / 6) * november →
  december = 2.857142857142857 * (november + january) / 2 →
  november / december = 3 / 5 := by
sorry

end toy_store_revenue_ratio_l1697_169726


namespace max_difference_three_digit_mean_505_l1697_169704

/-- The maximum difference between two three-digit integers with a mean of 505 -/
theorem max_difference_three_digit_mean_505 :
  ∃ (x y : ℕ),
    100 ≤ x ∧ x < 1000 ∧
    100 ≤ y ∧ y < 1000 ∧
    (x + y) / 2 = 505 ∧
    ∀ (a b : ℕ),
      100 ≤ a ∧ a < 1000 ∧
      100 ≤ b ∧ b < 1000 ∧
      (a + b) / 2 = 505 →
      x - y ≥ a - b ∧
    x - y = 810 :=
by sorry

end max_difference_three_digit_mean_505_l1697_169704


namespace circle_and_line_equations_l1697_169716

-- Define the circle C
def circle_C (x y : ℝ) : Prop :=
  ∃ (h : ℝ) (k : ℝ), (x - h)^2 + (y - k)^2 = 16 ∧ 2*h - k - 5 = 0

-- Define the line l
def line_l (x y : ℝ) : Prop :=
  x = 1 ∨ 3*x + 4*y - 23 = 0

-- Theorem statement
theorem circle_and_line_equations :
  ∀ (x y : ℝ),
    (circle_C x y ↔ (x - 3)^2 + (y - 1)^2 = 16) ∧
    (line_l x y ↔ (x = 1 ∨ 3*x + 4*y - 23 = 0)) :=
by sorry

end circle_and_line_equations_l1697_169716


namespace dividend_calculation_l1697_169746

theorem dividend_calculation (divisor quotient remainder : ℕ) 
  (h1 : divisor = 18) 
  (h2 : quotient = 9) 
  (h3 : remainder = 4) : 
  divisor * quotient + remainder = 166 := by
sorry

end dividend_calculation_l1697_169746


namespace project_hours_difference_l1697_169705

theorem project_hours_difference (total_hours kate_hours pat_hours mark_hours : ℕ) : 
  total_hours = 144 →
  pat_hours = 2 * kate_hours →
  pat_hours * 3 = mark_hours →
  total_hours = kate_hours + pat_hours + mark_hours →
  mark_hours - kate_hours = 80 := by
sorry

end project_hours_difference_l1697_169705


namespace circle_properties_l1697_169735

-- Define the circle equation
def circle_equation (x y m : ℝ) : Prop :=
  x^2 + y^2 - 2*x + 4*y + 4*m = 0

-- Define the range of m for which the equation represents a circle
def is_circle (m : ℝ) : Prop :=
  m < 5/4

-- Define the symmetric circle when m = 1
def symmetric_circle (x y : ℝ) : Prop :=
  (x + 1)^2 + (y + 2)^2 = 1

-- Define the line
def line (x y : ℝ) : Prop :=
  x + y - 1 = 0

-- Theorem statement
theorem circle_properties :
  (∀ m, is_circle m ↔ ∀ x y, ∃ r > 0, circle_equation x y m ↔ (x - 1)^2 + (y + 2)^2 = r^2) ∧
  (∀ x y, symmetric_circle x y →
    (∃ d, d = 2*Real.sqrt 2 + 1 ∧ ∀ x' y', line x' y' → d ≥ Real.sqrt ((x - x')^2 + (y - y')^2)) ∧
    (∃ d, d = 2*Real.sqrt 2 - 1 ∧ ∀ x' y', line x' y' → d ≤ Real.sqrt ((x - x')^2 + (y - y')^2))) :=
by sorry

end circle_properties_l1697_169735


namespace smallest_k_for_64_power_greater_than_4_power_17_l1697_169768

theorem smallest_k_for_64_power_greater_than_4_power_17 : 
  ∀ k : ℕ, (64 ^ k > 4 ^ 17 ∧ ∀ m : ℕ, m < k → 64 ^ m ≤ 4 ^ 17) ↔ k = 6 :=
by sorry

end smallest_k_for_64_power_greater_than_4_power_17_l1697_169768


namespace intersection_point_unique_l1697_169775

/-- The first line equation: 2x + 3y - 7 = 0 -/
def line1 (x y : ℝ) : Prop := 2 * x + 3 * y - 7 = 0

/-- The second line equation: 5x - y - 9 = 0 -/
def line2 (x y : ℝ) : Prop := 5 * x - y - 9 = 0

/-- The intersection point (2, 1) -/
def intersection_point : ℝ × ℝ := (2, 1)

/-- Theorem stating that (2, 1) is the unique intersection point of the two lines -/
theorem intersection_point_unique :
  (∃! p : ℝ × ℝ, line1 p.1 p.2 ∧ line2 p.1 p.2) ∧
  (line1 intersection_point.1 intersection_point.2) ∧
  (line2 intersection_point.1 intersection_point.2) :=
sorry

end intersection_point_unique_l1697_169775


namespace after_tax_dividend_amount_l1697_169790

def expected_earnings : ℝ := 0.80
def actual_earnings : ℝ := 1.10
def additional_dividend_rate : ℝ := 0.04
def additional_earnings_threshold : ℝ := 0.10
def tax_rate_threshold : ℝ := 1.00
def low_tax_rate : ℝ := 0.15
def high_tax_rate : ℝ := 0.20
def num_shares : ℕ := 300

def calculate_after_tax_dividend (
  expected_earnings : ℝ)
  (actual_earnings : ℝ)
  (additional_dividend_rate : ℝ)
  (additional_earnings_threshold : ℝ)
  (tax_rate_threshold : ℝ)
  (low_tax_rate : ℝ)
  (high_tax_rate : ℝ)
  (num_shares : ℕ) : ℝ :=
  sorry

theorem after_tax_dividend_amount :
  calculate_after_tax_dividend
    expected_earnings
    actual_earnings
    additional_dividend_rate
    additional_earnings_threshold
    tax_rate_threshold
    low_tax_rate
    high_tax_rate
    num_shares = 124.80 := by sorry

end after_tax_dividend_amount_l1697_169790


namespace donnys_spending_l1697_169723

/-- Donny's spending on Thursday given his savings from Monday to Wednesday -/
theorem donnys_spending (monday_savings : ℕ) (tuesday_savings : ℕ) (wednesday_savings : ℕ)
  (h1 : monday_savings = 15)
  (h2 : tuesday_savings = 28)
  (h3 : wednesday_savings = 13) :
  (monday_savings + tuesday_savings + wednesday_savings) / 2 = 28 := by
  sorry

end donnys_spending_l1697_169723


namespace contest_scores_order_l1697_169751

theorem contest_scores_order (A B C D : ℕ) 
  (eq1 : A + B = C + D)
  (eq2 : D + B = A + C + 10)
  (eq3 : C = A + D + 5) :
  B > C ∧ C > D ∧ D > A := by
  sorry

end contest_scores_order_l1697_169751


namespace lena_video_game_time_l1697_169785

/-- Proves that Lena played video games for 3.5 hours given the conditions of the problem -/
theorem lena_video_game_time (lena_time brother_time : ℕ) : 
  brother_time = lena_time + 17 →
  lena_time + brother_time = 437 →
  (lena_time : ℚ) / 60 = 3.5 := by
    sorry

end lena_video_game_time_l1697_169785


namespace fruit_seller_pricing_l1697_169760

/-- Given a fruit seller's pricing scenario, calculate the current selling price. -/
theorem fruit_seller_pricing (loss_percentage : ℝ) (profit_percentage : ℝ) (profit_price : ℝ) :
  loss_percentage = 20 →
  profit_percentage = 5 →
  profit_price = 10.5 →
  ∃ (current_price : ℝ),
    current_price = (1 - loss_percentage / 100) * (profit_price / (1 + profit_percentage / 100)) ∧
    current_price = 8 := by
  sorry

end fruit_seller_pricing_l1697_169760


namespace correct_calculation_l1697_169780

theorem correct_calculation (x y : ℝ) : 3 * x^2 * y - 2 * x^2 * y = x^2 * y := by
  sorry

end correct_calculation_l1697_169780


namespace function_inequality_l1697_169708

/-- Given a function f(x) = axe^x where a ≠ 0 and a ≥ 4/e^2, 
    prove that f(x)/(x+1) - (x+1)ln(x) > 0 for x > 0 -/
theorem function_inequality (a : ℝ) (h1 : a ≠ 0) (h2 : a ≥ 4 / Real.exp 2) :
  ∀ x > 0, (a * x * Real.exp x) / (x + 1) - (x + 1) * Real.log x > 0 := by
  sorry

end function_inequality_l1697_169708


namespace price_restoration_percentage_l1697_169750

theorem price_restoration_percentage (original_price : ℝ) (h : original_price > 0) :
  let reduced_price := 0.85 * original_price
  let restoration_factor := original_price / reduced_price
  let percentage_increase := (restoration_factor - 1) * 100
  ∃ ε > 0, abs (percentage_increase - 17.65) < ε :=
by sorry

end price_restoration_percentage_l1697_169750


namespace bacteria_eliminated_l1697_169774

/-- Number of bacteria on a given day -/
def bacteria_count (day : ℕ) : ℤ :=
  50 - 6 * (day - 1)

/-- The day when bacteria are eliminated -/
def elimination_day : ℕ := 10

/-- Theorem stating that bacteria are eliminated on the 10th day -/
theorem bacteria_eliminated :
  bacteria_count elimination_day ≤ 0 ∧
  ∀ d : ℕ, d < elimination_day → bacteria_count d > 0 :=
sorry

end bacteria_eliminated_l1697_169774


namespace sum_of_absolute_coefficients_l1697_169762

theorem sum_of_absolute_coefficients (a a₁ a₂ a₃ a₄ a₅ a₆ a₇ : ℝ) : 
  (∀ x : ℝ, (1 - x)^7 = a + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5 + a₆*x^6 + a₇*x^7) →
  |a| + |a₁| + |a₂| + |a₃| + |a₄| + |a₅| + |a₆| + |a₇| = 2^7 := by
  sorry

end sum_of_absolute_coefficients_l1697_169762
