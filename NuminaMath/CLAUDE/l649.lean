import Mathlib

namespace merchant_gross_profit_l649_64933

/-- The merchant's gross profit on a jacket sale --/
theorem merchant_gross_profit (purchase_price : ℝ) (markup_percent : ℝ) (discount_percent : ℝ) : 
  purchase_price = 42 ∧ 
  markup_percent = 0.3 ∧ 
  discount_percent = 0.2 → 
  let selling_price := purchase_price / (1 - markup_percent)
  let discounted_price := selling_price * (1 - discount_percent)
  let gross_profit := discounted_price - purchase_price
  gross_profit = 6 := by
sorry


end merchant_gross_profit_l649_64933


namespace max_value_abc_l649_64901

theorem max_value_abc (a b : Real) (c : Fin 2) 
  (ha : 0 ≤ a ∧ a ≤ 1) (hb : 0 ≤ b ∧ b ≤ 1) :
  ∃ (a₀ b₀ : Real) (c₀ : Fin 2) (ha₀ : 0 ≤ a₀ ∧ a₀ ≤ 1) (hb₀ : 0 ≤ b₀ ∧ b₀ ≤ 1),
    Real.sqrt (a * b * c.val) + Real.sqrt ((1 - a) * (1 - b) * (1 - c.val)) ≤ 1 ∧
    Real.sqrt (a₀ * b₀ * c₀.val) + Real.sqrt ((1 - a₀) * (1 - b₀) * (1 - c₀.val)) = 1 :=
by sorry

end max_value_abc_l649_64901


namespace student_multiplication_error_l649_64966

/-- Represents a repeating decimal of the form 1.abababab... -/
def repeating_decimal (a b : ℕ) : ℚ :=
  1 + (10 * a + b : ℚ) / 99

/-- Represents the decimal 1.ab -/
def non_repeating_decimal (a b : ℕ) : ℚ :=
  1 + (a : ℚ) / 10 + (b : ℚ) / 100

theorem student_multiplication_error (a b : ℕ) :
  a < 10 → b < 10 →
  66 * (repeating_decimal a b - non_repeating_decimal a b) = (1 : ℚ) / 2 →
  a * 10 + b = 75 := by
  sorry

end student_multiplication_error_l649_64966


namespace necessary_but_not_sufficient_l649_64923

theorem necessary_but_not_sufficient (a b c : ℝ) :
  (∀ a b c : ℝ, a * c^2 > b * c^2 → a > b) ∧
  ¬(∀ a b c : ℝ, a > b → a * c^2 > b * c^2) :=
by sorry

end necessary_but_not_sufficient_l649_64923


namespace simplify_nested_expression_l649_64983

theorem simplify_nested_expression (x : ℝ) : 1 - (1 + (1 - (1 + (1 - x)))) = 1 - x := by
  sorry

end simplify_nested_expression_l649_64983


namespace a_10_value_l649_64951

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem a_10_value (a : ℕ → ℝ) :
  arithmetic_sequence a → a 2 = 2 → a 3 = 4 → a 10 = 18 := by
  sorry

end a_10_value_l649_64951


namespace royal_family_theorem_l649_64993

/-- Represents the royal family -/
structure RoyalFamily where
  king_age : ℕ
  queen_age : ℕ
  num_sons : ℕ
  num_daughters : ℕ
  children_total_age : ℕ

/-- The conditions of the problem -/
def royal_family_conditions (family : RoyalFamily) : Prop :=
  family.king_age = 35 ∧
  family.queen_age = 35 ∧
  family.num_sons = 3 ∧
  family.num_daughters ≥ 1 ∧
  family.children_total_age = 35 ∧
  family.num_sons + family.num_daughters ≤ 20

/-- The theorem to be proved -/
theorem royal_family_theorem (family : RoyalFamily) 
  (h : royal_family_conditions family) :
  family.num_sons + family.num_daughters = 7 ∨
  family.num_sons + family.num_daughters = 9 :=
by
  sorry

end royal_family_theorem_l649_64993


namespace final_product_is_twelve_l649_64914

/-- Represents the count of each number on the blackboard -/
structure BoardState :=
  (ones : ℕ)
  (twos : ℕ)
  (threes : ℕ)
  (fours : ℕ)

/-- The operation performed on the board -/
def performOperation (state : BoardState) : BoardState :=
  { ones := state.ones - 1,
    twos := state.twos - 1,
    threes := state.threes - 1,
    fours := state.fours + 2 }

/-- Predicate to check if an operation can be performed -/
def canPerformOperation (state : BoardState) : Prop :=
  state.ones > 0 ∧ state.twos > 0 ∧ state.threes > 0

/-- Predicate to check if the board is in its final state -/
def isFinalState (state : BoardState) : Prop :=
  ¬(canPerformOperation state) ∧ 
  (state.ones + state.twos + state.threes + state.fours = 3)

/-- The initial state of the board -/
def initialState : BoardState :=
  { ones := 11, twos := 22, threes := 33, fours := 44 }

/-- The main theorem to prove -/
theorem final_product_is_twelve :
  ∃ (finalState : BoardState),
    (isFinalState finalState) ∧
    (finalState.ones * finalState.twos * finalState.threes * finalState.fours = 12) := by
  sorry

end final_product_is_twelve_l649_64914


namespace root_relation_implies_k_value_l649_64965

theorem root_relation_implies_k_value (k : ℝ) : 
  (∃ r s : ℝ, r^2 + k*r + 8 = 0 ∧ s^2 + k*s + 8 = 0 ∧
   (r+3)^2 - k*(r+3) + 8 = 0 ∧ (s+3)^2 - k*(s+3) + 8 = 0) →
  k = 3 := by
sorry

end root_relation_implies_k_value_l649_64965


namespace cyclic_fourth_root_sum_inequality_l649_64981

theorem cyclic_fourth_root_sum_inequality (a₁ a₂ a₃ a₄ a₅ a₆ : ℝ) 
  (ha₁ : a₁ > 0) (ha₂ : a₂ > 0) (ha₃ : a₃ > 0) (ha₄ : a₄ > 0) (ha₅ : a₅ > 0) (ha₆ : a₆ > 0) : 
  (a₁ / (a₂ + a₃ + a₄)) ^ (1/4 : ℝ) + 
  (a₂ / (a₃ + a₄ + a₅)) ^ (1/4 : ℝ) + 
  (a₃ / (a₄ + a₅ + a₆)) ^ (1/4 : ℝ) + 
  (a₄ / (a₅ + a₆ + a₁)) ^ (1/4 : ℝ) + 
  (a₅ / (a₆ + a₁ + a₂)) ^ (1/4 : ℝ) + 
  (a₆ / (a₁ + a₂ + a₃)) ^ (1/4 : ℝ) ≥ 2 := by
  sorry

end cyclic_fourth_root_sum_inequality_l649_64981


namespace partition_displacement_is_one_sixth_length_l649_64943

/-- Represents a cylindrical vessel with a movable partition -/
structure Vessel where
  length : ℝ
  initial_partition_position : ℝ
  final_partition_position : ℝ

/-- Calculates the displacement of the partition -/
def partition_displacement (v : Vessel) : ℝ :=
  v.initial_partition_position - v.final_partition_position

/-- Theorem stating the displacement of the partition -/
theorem partition_displacement_is_one_sixth_length (v : Vessel) 
  (h1 : v.length > 0)
  (h2 : v.initial_partition_position = 2 * v.length / 3)
  (h3 : v.final_partition_position = v.length / 2) :
  partition_displacement v = v.length / 6 := by
  sorry

#check partition_displacement_is_one_sixth_length

end partition_displacement_is_one_sixth_length_l649_64943


namespace min_value_of_fraction_l649_64982

theorem min_value_of_fraction (a : ℝ) (h : a > 1) :
  (a^2 - a + 1) / (a - 1) ≥ 3 ∧
  ∃ b > 1, (b^2 - b + 1) / (b - 1) = 3 :=
sorry

end min_value_of_fraction_l649_64982


namespace sum_number_and_square_l649_64994

/-- If a number is 16, then the sum of this number and its square is 272. -/
theorem sum_number_and_square (x : ℕ) : x = 16 → x + x^2 = 272 := by
  sorry

end sum_number_and_square_l649_64994


namespace robin_gum_packages_l649_64995

/-- Represents the number of pieces of gum in each package -/
def pieces_per_package : ℕ := 23

/-- Represents the number of extra pieces of gum Robin has -/
def extra_pieces : ℕ := 8

/-- Represents the total number of pieces of gum Robin has -/
def total_pieces : ℕ := 997

/-- Represents the number of packages Robin has -/
def num_packages : ℕ := (total_pieces - extra_pieces) / pieces_per_package

theorem robin_gum_packages : num_packages = 43 := by
  sorry

end robin_gum_packages_l649_64995


namespace gcd_102_238_l649_64956

theorem gcd_102_238 : Nat.gcd 102 238 = 34 := by
  sorry

end gcd_102_238_l649_64956


namespace log_sequence_l649_64919

theorem log_sequence (a b c : ℝ) (ha : a = Real.log 3 / Real.log 4)
    (hb : b = Real.log 6 / Real.log 4) (hc : c = Real.log 12 / Real.log 4) :
  (b - a = c - b) ∧ ¬(b / a = c / b) := by
  sorry

end log_sequence_l649_64919


namespace kendra_pens_l649_64908

/-- Proves that Kendra has 4 packs of pens given the problem conditions -/
theorem kendra_pens (kendra_packs : ℕ) : 
  let tony_packs : ℕ := 2
  let pens_per_pack : ℕ := 3
  let pens_kept_each : ℕ := 2
  let friends_given_pens : ℕ := 14
  kendra_packs * pens_per_pack - pens_kept_each + 
    (tony_packs * pens_per_pack - pens_kept_each) = friends_given_pens →
  kendra_packs = 4 := by
sorry

end kendra_pens_l649_64908


namespace barons_claim_correct_l649_64979

/-- Sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- Theorem stating the existence of two different 10-digit numbers satisfying the Baron's claim -/
theorem barons_claim_correct : 
  ∃ (a b : ℕ), 
    a ≠ b ∧ 
    10^9 ≤ a ∧ a < 10^10 ∧
    10^9 ≤ b ∧ b < 10^10 ∧
    a % 10 ≠ 0 ∧
    b % 10 ≠ 0 ∧
    a + sum_of_digits (a^2) = b + sum_of_digits (b^2) :=
by sorry

end barons_claim_correct_l649_64979


namespace x_value_l649_64987

theorem x_value (x y : ℝ) (h1 : x - y = 8) (h2 : x + y = 10) : x = 9 := by
  sorry

end x_value_l649_64987


namespace square_equals_product_sum_solutions_l649_64998

theorem square_equals_product_sum_solutions :
  ∀ (a b : ℤ), a ≥ 0 → b ≥ 0 → a^2 = b * (b + 7) → (a = 0 ∧ b = 0) ∨ (a = 12 ∧ b = 9) := by
  sorry

end square_equals_product_sum_solutions_l649_64998


namespace polynomial_simplification_l649_64963

theorem polynomial_simplification (p : ℝ) :
  (4 * p^4 + 2 * p^3 - 7 * p + 3) + (5 * p^3 - 8 * p^2 + 3 * p + 2) =
  4 * p^4 + 7 * p^3 - 8 * p^2 - 4 * p + 5 := by
  sorry

end polynomial_simplification_l649_64963


namespace tims_income_percentage_l649_64906

theorem tims_income_percentage (tim mary juan : ℝ) 
  (h1 : mary = tim * 1.6)
  (h2 : mary = juan * 0.6400000000000001) : 
  tim = juan * 0.4 := by
  sorry

end tims_income_percentage_l649_64906


namespace circle_equation_proof_l649_64902

theorem circle_equation_proof (x y : ℝ) :
  let circle_eq := (x - 5/3)^2 + y^2 = 25/9
  let line_eq := 3*x + y - 5 = 0
  let origin := (0, 0)
  let point := (3, -1)
  (∃ (center : ℝ × ℝ), 
    (center.1 - 5/3)^2 + center.2^2 = 25/9 ∧ 
    3*center.1 + center.2 - 5 = 0) ∧
  ((0 - 5/3)^2 + 0^2 = 25/9) ∧
  ((3 - 5/3)^2 + (-1)^2 = 25/9) →
  circle_eq
:= by sorry

end circle_equation_proof_l649_64902


namespace room_volume_example_l649_64986

/-- The volume of a rectangular room -/
def room_volume (length width height : ℝ) : ℝ := length * width * height

/-- Theorem: The volume of a room with dimensions 100 m * 10 m * 10 m is 10,000 cubic meters -/
theorem room_volume_example : room_volume 100 10 10 = 10000 := by
  sorry

end room_volume_example_l649_64986


namespace not_right_triangle_2_3_4_l649_64961

/-- A function that checks if three numbers can form a right triangle -/
def is_right_triangle (a b c : ℝ) : Prop :=
  a^2 + b^2 = c^2 ∨ a^2 + c^2 = b^2 ∨ b^2 + c^2 = a^2

/-- Theorem stating that the set {2, 3, 4} cannot form a right triangle -/
theorem not_right_triangle_2_3_4 : ¬ is_right_triangle 2 3 4 := by
  sorry

#check not_right_triangle_2_3_4

end not_right_triangle_2_3_4_l649_64961


namespace final_sum_after_operations_l649_64924

/-- Given two real numbers with sum S, prove that adding 5 to each and then tripling results in a sum of 3S + 30 -/
theorem final_sum_after_operations (a b S : ℝ) (h : a + b = S) :
  3 * (a + 5) + 3 * (b + 5) = 3 * S + 30 := by
  sorry


end final_sum_after_operations_l649_64924


namespace caterer_sundae_order_l649_64970

/-- Represents the problem of determining the number of sundaes ordered by a caterer --/
theorem caterer_sundae_order (total_price : ℚ) (ice_cream_bars : ℕ) (ice_cream_price : ℚ) (sundae_price : ℚ)
  (h1 : total_price = 200)
  (h2 : ice_cream_bars = 225)
  (h3 : ice_cream_price = 60/100)
  (h4 : sundae_price = 52/100) :
  ∃ (sundaes : ℕ), sundaes = 125 ∧ total_price = ice_cream_bars * ice_cream_price + sundaes * sundae_price :=
by sorry

end caterer_sundae_order_l649_64970


namespace smallest_divisor_is_number_itself_l649_64954

def form_number (a b : Nat) (digit : Nat) : Nat :=
  a * 1000 + digit * 100 + b

theorem smallest_divisor_is_number_itself :
  let complete_number := form_number 761 829 3
  complete_number % complete_number = 0 ∧
  ∀ d : Nat, d > 0 ∧ d < complete_number → complete_number % d ≠ 0 :=
by sorry

end smallest_divisor_is_number_itself_l649_64954


namespace hyperbola_real_axis_length_l649_64976

/-- Properties of a hyperbola -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  e : ℝ
  angle_F1PF2 : ℝ
  area_F1PF2 : ℝ
  a_pos : 0 < a
  b_pos : 0 < b
  e_eq : e = 2
  angle_eq : angle_F1PF2 = Real.pi / 2
  area_eq : area_F1PF2 = 3

/-- The length of the real axis of a hyperbola -/
def real_axis_length (h : Hyperbola) : ℝ := 2 * h.a

/-- Theorem: The length of the real axis of the given hyperbola is 2 -/
theorem hyperbola_real_axis_length (h : Hyperbola) : real_axis_length h = 2 := by
  sorry

end hyperbola_real_axis_length_l649_64976


namespace school_students_count_l649_64928

/-- Given the number of pencils and erasers ordered, and the number of each item given to each student,
    calculate the number of students in the school. -/
def calculate_students (total_pencils : ℕ) (total_erasers : ℕ) (pencils_per_student : ℕ) (erasers_per_student : ℕ) : ℕ :=
  min (total_pencils / pencils_per_student) (total_erasers / erasers_per_student)

/-- Theorem stating that the number of students in the school is 65. -/
theorem school_students_count : calculate_students 195 65 3 1 = 65 := by
  sorry

end school_students_count_l649_64928


namespace square_root_16_divided_by_2_l649_64978

theorem square_root_16_divided_by_2 : Real.sqrt 16 / 2 = 2 := by
  sorry

end square_root_16_divided_by_2_l649_64978


namespace olivia_chocolate_sales_l649_64946

/-- Calculates the money made from selling chocolate bars -/
def money_made (total_bars : ℕ) (bars_left : ℕ) (price_per_bar : ℕ) : ℕ :=
  (total_bars - bars_left) * price_per_bar

/-- Theorem stating that Olivia would make $9 from selling the chocolate bars -/
theorem olivia_chocolate_sales : 
  money_made 7 4 3 = 9 := by
  sorry

end olivia_chocolate_sales_l649_64946


namespace fisherman_daily_earnings_l649_64949

/-- Calculates the daily earnings of a fisherman based on their catch and fish prices -/
theorem fisherman_daily_earnings (red_snapper_count : ℕ) (tuna_count : ℕ) (red_snapper_price : ℕ) (tuna_price : ℕ) : 
  red_snapper_count = 8 → 
  tuna_count = 14 → 
  red_snapper_price = 3 → 
  tuna_price = 2 → 
  red_snapper_count * red_snapper_price + tuna_count * tuna_price = 52 := by
sorry

end fisherman_daily_earnings_l649_64949


namespace fourth_root_of_409600000_l649_64929

theorem fourth_root_of_409600000 : (409600000 : ℝ) ^ (1/4 : ℝ) = 80 := by
  sorry

end fourth_root_of_409600000_l649_64929


namespace simplify_expression_l649_64926

theorem simplify_expression :
  ∀ w x : ℝ, 3*w + 6*w + 9*w + 12*w + 15*w - 2*x - 4*x - 6*x - 8*x - 10*x + 24 = 45*w - 30*x + 24 :=
by
  sorry

end simplify_expression_l649_64926


namespace quadratic_inequality_range_l649_64920

theorem quadratic_inequality_range (m : ℝ) : 
  (∀ x : ℝ, (m^2 + 4*m - 5)*x^2 - 4*(m-1)*x + 3 > 0) ↔ (1 ≤ m ∧ m < 19) :=
sorry

end quadratic_inequality_range_l649_64920


namespace units_digit_sum_of_powers_l649_64997

theorem units_digit_sum_of_powers : ∃ n : ℕ, n < 10 ∧ (45^125 + 7^87) % 10 = n ∧ n = 8 := by
  sorry

end units_digit_sum_of_powers_l649_64997


namespace dodecagon_rectangle_area_equality_l649_64959

/-- The area of a regular dodecagon inscribed in a circle of radius r -/
def area_inscribed_dodecagon (r : ℝ) : ℝ := 3 * r^2

/-- The area of a rectangle with sides r and 3r -/
def area_rectangle (r : ℝ) : ℝ := r * (3 * r)

theorem dodecagon_rectangle_area_equality (r : ℝ) :
  area_inscribed_dodecagon r = area_rectangle r :=
by sorry

end dodecagon_rectangle_area_equality_l649_64959


namespace c_range_l649_64990

def p (c : ℝ) : Prop := ∀ x y : ℝ, x < y → (c - 1) * x + 1 < (c - 1) * y + 1

def q (c : ℝ) : Prop := ∀ x : ℝ, x^2 - x + c > 0

theorem c_range (c : ℝ) (hp : p c) (hq : q c) : c > 1 := by
  sorry

end c_range_l649_64990


namespace identical_geometric_sequences_l649_64938

/-- Two geometric sequences with the same first term -/
def geometric_sequence (a₀ : ℝ) (q : ℝ) (n : ℕ) : ℝ := a₀ * q^n

theorem identical_geometric_sequences
  (a₀ : ℝ) (q r : ℝ) :
  (∀ n : ℕ, ∃ s : ℝ, geometric_sequence a₀ q n + geometric_sequence a₀ r n = geometric_sequence (2 * a₀) s n) →
  q = r :=
sorry

end identical_geometric_sequences_l649_64938


namespace mark_bought_three_weeks_of_food_l649_64989

/-- Calculates the number of weeks of dog food purchased given the total cost,
    puppy cost, daily food consumption, bag size, and bag cost. -/
def weeks_of_food (total_cost puppy_cost daily_food_cups bag_size_cups bag_cost : ℚ) : ℚ :=
  let food_cost := total_cost - puppy_cost
  let bags_bought := food_cost / bag_cost
  let total_cups := bags_bought * bag_size_cups
  let days_of_food := total_cups / daily_food_cups
  days_of_food / 7

/-- Theorem stating that under the given conditions, Mark bought food for 3 weeks. -/
theorem mark_bought_three_weeks_of_food :
  weeks_of_food 14 10 (1/3) (7/2) 2 = 3 := by
  sorry


end mark_bought_three_weeks_of_food_l649_64989


namespace expected_value_binomial_l649_64955

/-- The number of missile launches -/
def n : ℕ := 10

/-- The probability of an accident in a single launch -/
def p : ℝ := 0.01

/-- The random variable representing the number of accidents -/
def ξ : Nat → ℝ := sorry

theorem expected_value_binomial :
  Finset.sum (Finset.range (n + 1)) (fun k => k * (n.choose k : ℝ) * p^k * (1 - p)^(n - k)) = n * p :=
sorry

end expected_value_binomial_l649_64955


namespace quadratic_inequality_range_l649_64932

theorem quadratic_inequality_range (x : ℝ) :
  (∃ a : ℝ, a ∈ Set.Icc 2 4 ∧ a * x^2 + (a - 3) * x - 3 > 0) →
  x ∈ Set.Ioi (-1) ∪ Set.Iio (3/4) := by
sorry

end quadratic_inequality_range_l649_64932


namespace lcm_hcf_problem_l649_64969

theorem lcm_hcf_problem (A B : ℕ+) : 
  Nat.lcm A B = 2310 →
  Nat.gcd A B = 30 →
  A = 462 →
  B = 150 := by
sorry

end lcm_hcf_problem_l649_64969


namespace tip_percentage_calculation_l649_64980

theorem tip_percentage_calculation (meal_cost drink_cost payment change : ℚ) : 
  meal_cost = 10 →
  drink_cost = 5/2 →
  payment = 20 →
  change = 5 →
  ((payment - change) - (meal_cost + drink_cost)) / (meal_cost + drink_cost) * 100 = 20 := by
  sorry

end tip_percentage_calculation_l649_64980


namespace sum_of_digits_of_triangular_array_rows_l649_64936

def triangular_sum (n : ℕ) : ℕ := n * (n + 1) / 2

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

theorem sum_of_digits_of_triangular_array_rows : ∃ N : ℕ, 
  triangular_sum N = 3003 ∧ sum_of_digits N = 14 := by
  sorry

end sum_of_digits_of_triangular_array_rows_l649_64936


namespace ice_cream_distribution_l649_64934

theorem ice_cream_distribution (total_sandwiches : ℕ) (num_nieces : ℕ) 
  (h1 : total_sandwiches = 143)
  (h2 : num_nieces = 11) :
  total_sandwiches / num_nieces = 13 :=
by
  sorry

end ice_cream_distribution_l649_64934


namespace tank_capacity_tank_capacity_proof_l649_64917

/-- Given a tank where adding 130 gallons when it's 1/6 full makes it 3/5 full,
    prove that the tank's total capacity is 300 gallons. -/
theorem tank_capacity : ℝ → Prop :=
  fun capacity =>
    (capacity / 6 + 130 = 3 * capacity / 5) → capacity = 300

-- The proof is omitted
theorem tank_capacity_proof : ∃ capacity, tank_capacity capacity :=
  sorry

end tank_capacity_tank_capacity_proof_l649_64917


namespace truck_toll_theorem_l649_64903

/-- Calculates the toll for a truck given the number of axles -/
def toll (axles : ℕ) : ℚ :=
  2.50 + 0.50 * (axles - 2)

/-- Calculates the number of axles for a truck given the total number of wheels,
    wheels on the front axle, and wheels on each other axle -/
def calculateAxles (totalWheels frontAxleWheels otherAxleWheels : ℕ) : ℕ :=
  1 + (totalWheels - frontAxleWheels) / otherAxleWheels

theorem truck_toll_theorem (totalWheels frontAxleWheels otherAxleWheels : ℕ)
    (h1 : totalWheels = 18)
    (h2 : frontAxleWheels = 2)
    (h3 : otherAxleWheels = 4) :
  toll (calculateAxles totalWheels frontAxleWheels otherAxleWheels) = 4 :=
by
  sorry

#eval toll (calculateAxles 18 2 4)

end truck_toll_theorem_l649_64903


namespace james_writing_speed_l649_64931

/-- James writes some pages an hour. -/
def pages_per_hour : ℝ := sorry

/-- James writes 5 pages a day to 2 different people. -/
def pages_per_day : ℝ := 5 * 2

/-- James spends 7 hours a week writing. -/
def hours_per_week : ℝ := 7

/-- The number of days in a week. -/
def days_per_week : ℝ := 7

theorem james_writing_speed :
  pages_per_hour = 10 :=
sorry

end james_writing_speed_l649_64931


namespace nine_students_left_l649_64999

/-- The number of students left after some were checked out early -/
def students_left (initial : ℕ) (checked_out : ℕ) : ℕ :=
  initial - checked_out

/-- Theorem: Given 16 initial students and 7 checked out early, 9 students are left -/
theorem nine_students_left :
  students_left 16 7 = 9 := by
  sorry

end nine_students_left_l649_64999


namespace product_sqrt_inequality_l649_64991

theorem product_sqrt_inequality (a b c : ℝ) (ha : a > 1) (hb : b > 1) (hc : c > 1) (hsum : a + b + c = 9) :
  Real.sqrt (a * b + b * c + c * a) ≤ Real.sqrt a + Real.sqrt b + Real.sqrt c := by
  sorry

end product_sqrt_inequality_l649_64991


namespace unique_function_satisfying_condition_l649_64974

theorem unique_function_satisfying_condition :
  ∃! f : ℝ → ℝ, (∀ x y z : ℝ, f (x * y) + f (x * z) + f x * f (y * z) ≥ 3) ∧
  (∀ x : ℝ, f x = 1) := by
  sorry

end unique_function_satisfying_condition_l649_64974


namespace gcd_n_cube_plus_25_and_n_plus_3_l649_64900

theorem gcd_n_cube_plus_25_and_n_plus_3 (n : ℕ) (h : n > 3^2) :
  Nat.gcd (n^3 + 5^2) (n + 3) = if (n + 3) % 2 = 0 then 2 else 1 := by
  sorry

end gcd_n_cube_plus_25_and_n_plus_3_l649_64900


namespace least_prime_factor_of_p6_minus_p5_l649_64973

theorem least_prime_factor_of_p6_minus_p5 (p : ℕ) (hp : Nat.Prime p) :
  Nat.minFac (p^6 - p^5) = 2 := by
sorry

end least_prime_factor_of_p6_minus_p5_l649_64973


namespace solution_set_implies_a_values_l649_64972

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x - a| + 3 * x

-- State the theorem
theorem solution_set_implies_a_values :
  (∀ x : ℝ, f a x ≤ 0 ↔ x ≤ -1) →
  (a = 2 ∨ a = -4) :=
by sorry

end solution_set_implies_a_values_l649_64972


namespace sqrt_equation_solution_l649_64940

theorem sqrt_equation_solution (y : ℝ) : 
  Real.sqrt (5 * y + 15) = 15 → y = 42 := by sorry

end sqrt_equation_solution_l649_64940


namespace power_2021_representation_l649_64988

theorem power_2021_representation (n : ℕ+) :
  (∃ (x y : ℤ), (2021 : ℤ)^(n : ℕ) = x^4 - 4*y^4) ↔ 4 ∣ n := by
  sorry

end power_2021_representation_l649_64988


namespace a_values_l649_64915

def A : Set ℝ := {x | x^2 - 8*x + 15 = 0}
def B (a : ℝ) : Set ℝ := {x | a*x - 1 = 0}

theorem a_values (h : ∀ a : ℝ, B a ⊆ A) : 
  {a : ℝ | B a ⊆ A} = {0, 1/3, 1/5} := by sorry

end a_values_l649_64915


namespace polynomial_with_prime_roots_l649_64992

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

theorem polynomial_with_prime_roots (s : ℕ) :
  (∃ p q : ℕ, is_prime p ∧ is_prime q ∧ p + q = 7 ∧ p * q = s) →
  s = 10 :=
by sorry

end polynomial_with_prime_roots_l649_64992


namespace cos_2x_plus_sin_pi_half_minus_x_properties_l649_64911

/-- The function f(x) = cos(2x) + sin(π/2 - x) has both maximum and minimum values and is an even function. -/
theorem cos_2x_plus_sin_pi_half_minus_x_properties :
  ∃ (f : ℝ → ℝ), 
    (∀ x, f x = Real.cos (2 * x) + Real.sin (Real.pi / 2 - x)) ∧
    (∃ (max min : ℝ), ∀ x, min ≤ f x ∧ f x ≤ max) ∧
    (∀ x, f (-x) = f x) := by
  sorry


end cos_2x_plus_sin_pi_half_minus_x_properties_l649_64911


namespace nines_in_hundred_l649_64935

/-- Count of digit 9 in a single number -/
def count_nines (n : ℕ) : ℕ := sorry

/-- Sum of count_nines for all numbers from 1 to n -/
def total_nines (n : ℕ) : ℕ := sorry

/-- Theorem: The count of the digit 9 in all numbers from 1 to 100 (inclusive) is 20 -/
theorem nines_in_hundred : total_nines 100 = 20 := by sorry

end nines_in_hundred_l649_64935


namespace point_plane_line_sphere_ratio_l649_64968

/-- Given a point (a,b,c) on a plane and a line through the origin, 
    and (p,q,r) as the center of a sphere passing through specific points,
    prove that (a+b+c)/(p+q+r) = 1 -/
theorem point_plane_line_sphere_ratio 
  (a b c d e f p q r : ℝ) 
  (h1 : ∃ (t : ℝ), a = t * d ∧ b = t * e ∧ c = t * f)  -- (a,b,c) on line with direction (d,e,f)
  (h2 : ∃ (α β γ : ℝ), α ≠ 0 ∧ β ≠ 0 ∧ γ ≠ 0 ∧        -- A, B, C distinct from O
        a/α + b/β + c/γ = 1)                          -- (a,b,c) on plane through A, B, C
  (h3 : p^2 + q^2 + r^2 = (p - α)^2 + q^2 + r^2)       -- O and A equidistant from (p,q,r)
  (h4 : p^2 + q^2 + r^2 = p^2 + (q - β)^2 + r^2)       -- O and B equidistant from (p,q,r)
  (h5 : p^2 + q^2 + r^2 = p^2 + q^2 + (r - γ)^2)       -- O and C equidistant from (p,q,r)
  (h6 : p ≠ 0 ∧ q ≠ 0 ∧ r ≠ 0)                        -- Avoid division by zero
  : (a + b + c) / (p + q + r) = 1 := by
  sorry

end point_plane_line_sphere_ratio_l649_64968


namespace complex_point_on_real_axis_l649_64945

theorem complex_point_on_real_axis (a : ℝ) : 
  (Complex.I + 1) * (Complex.I + a) ∈ Set.range Complex.ofReal → a = -1 := by
  sorry

end complex_point_on_real_axis_l649_64945


namespace raisin_count_l649_64996

theorem raisin_count (total : ℕ) (box1 : ℕ) (box345 : ℕ) (h1 : total = 437) 
  (h2 : box1 = 72) (h3 : box345 = 97) : 
  total - box1 - 3 * box345 = 74 := by
  sorry

end raisin_count_l649_64996


namespace point_position_l649_64937

/-- A point in a 2D coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- Predicate to check if a point is to the right of the y-axis -/
def isRightOfYAxis (p : Point) : Prop := p.x > 0

/-- Predicate to check if a point is below the x-axis -/
def isBelowXAxis (p : Point) : Prop := p.y < 0

/-- Theorem stating that if a point is to the right of the y-axis and below the x-axis,
    then its x-coordinate is positive and its y-coordinate is negative -/
theorem point_position (P : Point) 
  (h1 : isRightOfYAxis P) (h2 : isBelowXAxis P) : 
  P.x > 0 ∧ P.y < 0 := by
  sorry

#check point_position

end point_position_l649_64937


namespace least_reducible_fraction_l649_64910

def is_reducible (n : ℕ) : Prop :=
  n > 0 ∧ (n - 17).gcd (7 * n + 5) > 1

theorem least_reducible_fraction :
  (∀ m : ℕ, m < 48 → ¬(is_reducible m)) ∧ is_reducible 48 := by
  sorry

end least_reducible_fraction_l649_64910


namespace remainder_problem_l649_64962

theorem remainder_problem (N : ℕ) : 
  (∃ r : ℕ, N = 44 * 432 + r ∧ r < 44) → 
  (∃ q : ℕ, N = 30 * q + 18) → 
  N % 44 = 18 := by
sorry

end remainder_problem_l649_64962


namespace tantrix_impossibility_l649_64922

/-- Represents a tile in the Tantrix Solitaire game -/
structure Tile where
  blue_lines : Nat
  red_lines : Nat

/-- Represents the game board -/
structure Board where
  tiles : List Tile
  blue_loop : Bool
  no_gaps : Bool
  red_intersections : Nat

/-- Checks if a board configuration is valid -/
def is_valid_board (b : Board) : Prop :=
  b.tiles.length = 13 ∧ b.blue_loop ∧ b.no_gaps ∧ b.red_intersections = 3

/-- Theorem stating the impossibility of arranging 13 tiles to form a valid board -/
theorem tantrix_impossibility : ¬ ∃ (b : Board), is_valid_board b := by
  sorry

end tantrix_impossibility_l649_64922


namespace regular_polygon_sides_l649_64909

theorem regular_polygon_sides (n : ℕ) (h_regular : n ≥ 3) 
  (h_interior_angle : (n - 2) * 180 / n = 144) : n = 10 := by
  sorry

end regular_polygon_sides_l649_64909


namespace power_function_property_l649_64947

-- Define a power function
def isPowerFunction (f : ℝ → ℝ) : Prop :=
  ∃ α : ℝ, ∀ x : ℝ, x > 0 → f x = x ^ α

-- State the theorem
theorem power_function_property (f : ℝ → ℝ) 
  (h_power : isPowerFunction f) 
  (h_condition : f 4 = 2 * f 2) : 
  f 3 = 3 := by
sorry

end power_function_property_l649_64947


namespace combined_alloy_force_problem_solution_l649_64942

/-- Represents an alloy of two metals -/
structure Alloy where
  mass : ℝ
  ratio : ℝ
  force : ℝ

/-- Theorem stating that the force exerted by a combination of two alloys
    is equal to the sum of their individual forces -/
theorem combined_alloy_force (A B : Alloy) :
  let C : Alloy := ⟨A.mass + B.mass, (A.mass * A.ratio + B.mass * B.ratio) / (A.mass + B.mass), A.force + B.force⟩
  C.force = A.force + B.force := by
  sorry

/-- Given alloys A and B with specified properties, prove that their combination
    exerts a force of 40 N -/
theorem problem_solution (A B : Alloy)
  (hA_mass : A.mass = 6)
  (hA_ratio : A.ratio = 2)
  (hA_force : A.force = 30)
  (hB_mass : B.mass = 3)
  (hB_ratio : B.ratio = 1/5)
  (hB_force : B.force = 10) :
  let C : Alloy := ⟨A.mass + B.mass, (A.mass * A.ratio + B.mass * B.ratio) / (A.mass + B.mass), A.force + B.force⟩
  C.force = 40 := by
  sorry

end combined_alloy_force_problem_solution_l649_64942


namespace absolute_value_inequality_l649_64939

theorem absolute_value_inequality (x : ℝ) : 
  |x - 1| + |x - 2| > 5 ↔ x < -1 ∨ x > 4 := by
  sorry

end absolute_value_inequality_l649_64939


namespace other_number_is_two_l649_64953

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

theorem other_number_is_two :
  ∃ n : ℕ, factorial 8 / factorial (8 - n) = 56 ∧ n = 2 := by
  sorry

end other_number_is_two_l649_64953


namespace door_width_calculation_l649_64941

/-- Calculates the width of a door given room dimensions and whitewashing costs -/
theorem door_width_calculation (room_length room_width room_height : ℝ)
  (door_height : ℝ) (window_width window_height : ℝ) (num_windows : ℕ)
  (cost_per_sqft total_cost : ℝ) : 
  room_length = 25 ∧ room_width = 15 ∧ room_height = 12 ∧
  door_height = 6 ∧ window_width = 4 ∧ window_height = 3 ∧
  num_windows = 3 ∧ cost_per_sqft = 9 ∧ total_cost = 8154 →
  ∃ (door_width : ℝ),
    (2 * (room_length * room_height + room_width * room_height) - 
     (door_height * door_width + num_windows * window_width * window_height)) * cost_per_sqft = total_cost ∧
    door_width = 3 := by
  sorry

end door_width_calculation_l649_64941


namespace running_track_l649_64905

/-- Given two concentric circles with radii r₁ and r₂, where the difference in their circumferences is 24π feet, prove the width of the track and the enclosed area. -/
theorem running_track (r₁ r₂ : ℝ) (h : 2 * Real.pi * r₁ - 2 * Real.pi * r₂ = 24 * Real.pi) :
  r₁ - r₂ = 12 ∧ Real.pi * (r₁^2 - r₂^2) = Real.pi * (24 * r₂ + 144) :=
by sorry

end running_track_l649_64905


namespace billys_weekend_activities_l649_64930

/-- Billy's weekend activities theorem -/
theorem billys_weekend_activities :
  -- Define the given conditions
  let free_time_per_day : ℕ := 8
  let weekend_days : ℕ := 2
  let pages_per_hour : ℕ := 60
  let pages_per_book : ℕ := 80
  let books_read : ℕ := 3

  -- Calculate total free time
  let total_free_time : ℕ := free_time_per_day * weekend_days

  -- Calculate total pages read
  let total_pages_read : ℕ := pages_per_book * books_read

  -- Calculate time spent reading
  let reading_time : ℕ := total_pages_read / pages_per_hour

  -- Calculate time spent playing video games
  let gaming_time : ℕ := total_free_time - reading_time

  -- Calculate percentage of time spent playing video games
  let gaming_percentage : ℚ := (gaming_time : ℚ) / (total_free_time : ℚ) * 100

  -- Prove that Billy spends 75% of his time playing video games
  gaming_percentage = 75 := by sorry

end billys_weekend_activities_l649_64930


namespace mirror_wall_area_ratio_l649_64907

/-- Proves that the ratio of the area of a square mirror to the area of a rectangular wall is 1:2 -/
theorem mirror_wall_area_ratio (mirror_side : ℝ) (wall_width wall_length : ℝ)
  (h1 : mirror_side = 18)
  (h2 : wall_width = 32)
  (h3 : wall_length = 20.25) :
  (mirror_side^2) / (wall_width * wall_length) = 1 / 2 := by
sorry

end mirror_wall_area_ratio_l649_64907


namespace twice_a_minus_four_nonnegative_l649_64975

theorem twice_a_minus_four_nonnegative (a : ℝ) :
  (2 * a - 4 ≥ 0) ↔ (∃ (x : ℝ), x ≥ 0 ∧ x = 2 * a - 4) :=
by sorry

end twice_a_minus_four_nonnegative_l649_64975


namespace counterfeit_coin_identification_l649_64913

/-- Represents the result of a weighing operation -/
inductive WeighingResult
  | Left : WeighingResult  -- Left side is heavier
  | Right : WeighingResult -- Right side is heavier
  | Equal : WeighingResult -- Both sides are equal

/-- Represents a weighing operation -/
def Weighing := Nat → Nat → WeighingResult

/-- Represents a strategy to find the counterfeit coin -/
def Strategy := List (Nat × Nat) → Nat

/-- Checks if a strategy correctly identifies the counterfeit coin -/
def isValidStrategy (n : Nat) (strategy : Strategy) : Prop :=
  ∀ (counterfeit : Nat), counterfeit < n →
    ∃ (weighings : List (Nat × Nat)),
      (∀ w ∈ weighings, w.1 < n ∧ w.2 < n) ∧
      (weighings.length ≤ 3) ∧
      (strategy weighings = counterfeit)

theorem counterfeit_coin_identification (n : Nat) (h : n = 10 ∨ n = 27) :
  ∃ (strategy : Strategy), isValidStrategy n strategy :=
sorry

end counterfeit_coin_identification_l649_64913


namespace peach_cost_per_pound_l649_64944

def initial_amount : ℚ := 20
def final_amount : ℚ := 14
def pounds_of_peaches : ℚ := 3

theorem peach_cost_per_pound :
  (initial_amount - final_amount) / pounds_of_peaches = 2 := by
  sorry

end peach_cost_per_pound_l649_64944


namespace simplify_sqrt_expression_l649_64971

theorem simplify_sqrt_expression : 
  (Real.sqrt 450 / Real.sqrt 200) + (Real.sqrt 392 / Real.sqrt 98) = 7 / 2 := by
  sorry

end simplify_sqrt_expression_l649_64971


namespace min_value_sum_reciprocals_l649_64964

theorem min_value_sum_reciprocals (x y z : ℝ) 
  (pos_x : 0 < x) (pos_y : 0 < y) (pos_z : 0 < z) 
  (sum_eq_3 : x + y + z = 3) : 
  (1/x) + (4/y) + (9/z) ≥ 12 ∧ 
  ∃ (a b c : ℝ), 0 < a ∧ 0 < b ∧ 0 < c ∧ 
    a + b + c = 3 ∧ (1/a) + (4/b) + (9/c) = 12 := by
  sorry

end min_value_sum_reciprocals_l649_64964


namespace geometric_sequence_terms_l649_64985

/-- Given a geometric sequence with third term 12 and fourth term 18, prove that the first term is 16/3 and the second term is 8. -/
theorem geometric_sequence_terms (a : ℕ → ℚ) (q : ℚ) :
  (∀ n, a (n + 1) = a n * q) →  -- Definition of geometric sequence
  a 3 = 12 →                    -- Third term is 12
  a 4 = 18 →                    -- Fourth term is 18
  a 1 = 16 / 3 ∧ a 2 = 8 :=     -- First term is 16/3 and second term is 8
by sorry

end geometric_sequence_terms_l649_64985


namespace total_weekly_meals_l649_64984

/-- The number of days in a week -/
def daysInWeek : ℕ := 7

/-- The number of meals served daily by the first restaurant -/
def restaurant1Meals : ℕ := 20

/-- The number of meals served daily by the second restaurant -/
def restaurant2Meals : ℕ := 40

/-- The number of meals served daily by the third restaurant -/
def restaurant3Meals : ℕ := 50

/-- Theorem stating that the total number of meals served per week by the three restaurants is 770 -/
theorem total_weekly_meals :
  (restaurant1Meals * daysInWeek) + (restaurant2Meals * daysInWeek) + (restaurant3Meals * daysInWeek) = 770 := by
  sorry

end total_weekly_meals_l649_64984


namespace triangle_inequality_l649_64957

theorem triangle_inequality (A B C : Real) (h : A + B + C = π) :
  (Real.sqrt (Real.sin A * Real.sin B)) / (Real.sin (C / 2)) ≥ 3 * Real.sqrt 3 * Real.tan (A / 2) * Real.tan (B / 2) := by
  sorry

end triangle_inequality_l649_64957


namespace expected_adjacent_pairs_l649_64925

/-- The expected number of adjacent boy-girl pairs in a random permutation of boys and girls -/
theorem expected_adjacent_pairs (b g : ℕ) (hb : b = 8) (hg : g = 12) :
  let total := b + g
  let prob_bg := (b : ℚ) / total * (g : ℚ) / (total - 1)
  let prob_pair := 2 * prob_bg
  let num_pairs := total - 1
  num_pairs * prob_pair = 912 / 95 := by sorry

end expected_adjacent_pairs_l649_64925


namespace problem_solution_l649_64967

theorem problem_solution (x y : ℝ) (hx : x > 0) (hy : y > 0)
  (h1 : Real.log x / Real.log y + Real.log y / Real.log x = 4)
  (h2 : x * y = 64) :
  (x + y) / 2 = 13 * Real.sqrt 3 := by
  sorry

end problem_solution_l649_64967


namespace division_simplification_l649_64927

theorem division_simplification (x y : ℝ) (h : x ≠ 0) :
  6 * x^3 * y^2 / (3 * x) = 2 * x^2 * y^2 := by
  sorry

end division_simplification_l649_64927


namespace brick_surface_area_l649_64950

/-- The surface area of a rectangular prism -/
def surface_area (length width height : ℝ) : ℝ :=
  2 * (length * width + length * height + width * height)

/-- Theorem: The surface area of a 8 cm x 6 cm x 2 cm brick is 152 cm² -/
theorem brick_surface_area :
  surface_area 8 6 2 = 152 := by
  sorry

end brick_surface_area_l649_64950


namespace gym_membership_ratio_l649_64912

theorem gym_membership_ratio (f m : ℕ) (h1 : f > 0) (h2 : m > 0) : 
  (35 : ℝ) * f + 20 * m = 25 * (f + m) → f / m = 1 / 2 := by
  sorry

end gym_membership_ratio_l649_64912


namespace right_triangle_arm_square_l649_64977

theorem right_triangle_arm_square (a c : ℝ) (h1 : c = a + 2) :
  ∃ b : ℝ, a^2 + b^2 = c^2 ∧ b^2 = 4*a + 4 := by
  sorry

end right_triangle_arm_square_l649_64977


namespace movie_expenses_split_l649_64952

theorem movie_expenses_split (num_friends : ℕ) (ticket_price popcorn_price parking_fee milk_tea_price candy_bar_price : ℚ)
  (num_tickets num_popcorn num_milk_tea num_candy_bars : ℕ) :
  num_friends = 4 ∧
  ticket_price = 7 ∧
  popcorn_price = 3/2 ∧
  parking_fee = 4 ∧
  milk_tea_price = 3 ∧
  candy_bar_price = 2 ∧
  num_tickets = 4 ∧
  num_popcorn = 2 ∧
  num_milk_tea = 3 ∧
  num_candy_bars = 4 →
  (num_tickets * ticket_price + num_popcorn * popcorn_price + parking_fee +
   num_milk_tea * milk_tea_price + num_candy_bars * candy_bar_price) / num_friends = 13 :=
by sorry

end movie_expenses_split_l649_64952


namespace no_valid_arrangement_l649_64948

def is_valid_arrangement (perm : List Nat) : Prop :=
  perm.length = 8 ∧
  (∀ n, n ∈ perm → n ∈ [1, 2, 3, 4, 5, 6, 8, 9]) ∧
  (∀ i, i < 7 → (10 * perm[i]! + perm[i+1]!) % 7 = 0)

theorem no_valid_arrangement : ¬∃ perm : List Nat, is_valid_arrangement perm := by
  sorry

end no_valid_arrangement_l649_64948


namespace change_calculation_l649_64916

def laptop_price : ℝ := 600
def smartphone_price : ℝ := 400
def tablet_price : ℝ := 250
def headphone_price : ℝ := 100
def discount_rate : ℝ := 0.1
def tax_rate : ℝ := 0.05
def num_laptops : ℕ := 2
def num_smartphones : ℕ := 4
def num_tablets : ℕ := 3
def num_headphones : ℕ := 5
def initial_amount : ℝ := 5000

theorem change_calculation : 
  let total_before_discount := 
    num_laptops * laptop_price + 
    num_smartphones * smartphone_price + 
    num_tablets * tablet_price + 
    num_headphones * headphone_price
  let discount := 
    discount_rate * (num_laptops * laptop_price + num_tablets * tablet_price)
  let total_after_discount := total_before_discount - discount
  let tax := tax_rate * total_after_discount
  let final_price := total_after_discount + tax
  initial_amount - final_price = 952.25 := by sorry

end change_calculation_l649_64916


namespace max_average_annual_profit_l649_64918

/-- Represents the total profit (in million yuan) for operating 4 buses for x years -/
def total_profit (x : ℕ+) : ℚ :=
  16 * (-2 * x^2 + 23 * x - 50)

/-- Represents the average annual profit (in million yuan) for operating 4 buses for x years -/
def average_annual_profit (x : ℕ+) : ℚ :=
  total_profit x / x

/-- Theorem stating that the average annual profit is maximized when x = 5 -/
theorem max_average_annual_profit :
  ∀ x : ℕ+, average_annual_profit 5 ≥ average_annual_profit x :=
sorry

end max_average_annual_profit_l649_64918


namespace largest_divisor_of_10000_l649_64904

theorem largest_divisor_of_10000 :
  ∀ n : ℕ, n ∣ 10000 ∧ ¬(n ∣ 9999) → n ≤ 10000 :=
by
  sorry

end largest_divisor_of_10000_l649_64904


namespace distance_before_meeting_is_100_l649_64921

/-- The distance between two trains one hour before they meet -/
def distance_before_meeting (total_distance : ℝ) (speed_A speed_B : ℝ) (delay : ℝ) : ℝ :=
  let relative_speed := speed_A + speed_B
  let time_to_meet := (total_distance - speed_A * delay) / relative_speed
  relative_speed

/-- Theorem stating the distance between trains one hour before meeting -/
theorem distance_before_meeting_is_100 :
  distance_before_meeting 435 45 55 (40/60) = 100 := by
  sorry

end distance_before_meeting_is_100_l649_64921


namespace state_quarter_fraction_l649_64958

theorem state_quarter_fraction :
  ∀ (total_quarters state_quarters pennsylvania_quarters : ℕ),
    total_quarters = 35 →
    pennsylvania_quarters = 7 →
    2 * pennsylvania_quarters = state_quarters →
    (state_quarters : ℚ) / total_quarters = 2 / 5 := by
  sorry

end state_quarter_fraction_l649_64958


namespace pure_imaginary_condition_l649_64960

open Complex

theorem pure_imaginary_condition (θ : ℝ) :
  let Z : ℂ := 1 / (sin θ + cos θ * I) - (1 : ℂ) / 2
  (∃ y : ℝ, Z = y * I) →
  (∃ k : ℤ, θ = π / 6 + 2 * k * π ∨ θ = 5 * π / 6 + 2 * k * π) :=
by sorry

end pure_imaginary_condition_l649_64960
