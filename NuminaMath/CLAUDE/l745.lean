import Mathlib

namespace furniture_purchase_price_l745_74516

theorem furniture_purchase_price 
  (marked_price : ℝ) 
  (discount_rate : ℝ) 
  (profit_rate : ℝ) 
  (purchase_price : ℝ) : 
  marked_price = 132 ∧ 
  discount_rate = 0.1 ∧ 
  profit_rate = 0.1 ∧ 
  marked_price * (1 - discount_rate) = purchase_price * (1 + profit_rate) → 
  purchase_price = 108 := by
sorry

end furniture_purchase_price_l745_74516


namespace elite_cheaper_at_min_shirts_l745_74527

/-- Elite T-Shirt Company's pricing structure -/
def elite_cost (n : ℕ) : ℚ := 30 + 8 * n

/-- Omega T-Shirt Company's pricing structure -/
def omega_cost (n : ℕ) : ℚ := 10 + 12 * n

/-- The minimum number of shirts for which Elite is cheaper than Omega -/
def min_shirts_for_elite : ℕ := 6

theorem elite_cheaper_at_min_shirts :
  elite_cost min_shirts_for_elite < omega_cost min_shirts_for_elite ∧
  ∀ k : ℕ, k < min_shirts_for_elite → elite_cost k ≥ omega_cost k :=
by sorry

end elite_cheaper_at_min_shirts_l745_74527


namespace staff_avg_age_l745_74508

def robotics_camp (total_members : ℕ) (overall_avg_age : ℝ)
  (num_girls num_boys num_adults num_staff : ℕ)
  (avg_age_girls avg_age_boys avg_age_adults : ℝ) : Prop :=
  total_members = 50 ∧
  overall_avg_age = 20 ∧
  num_girls = 22 ∧
  num_boys = 18 ∧
  num_adults = 5 ∧
  num_staff = 5 ∧
  avg_age_girls = 18 ∧
  avg_age_boys = 19 ∧
  avg_age_adults = 30

theorem staff_avg_age
  (h : robotics_camp 50 20 22 18 5 5 18 19 30) :
  (50 * 20 - (22 * 18 + 18 * 19 + 5 * 30)) / 5 = 22.4 :=
by sorry

end staff_avg_age_l745_74508


namespace ice_cream_stacking_permutations_l745_74550

theorem ice_cream_stacking_permutations : Nat.factorial 5 = 120 := by
  sorry

end ice_cream_stacking_permutations_l745_74550


namespace triangle_angle_sum_rounded_l745_74574

-- Define a structure for a triangle with actual and rounded angles
structure Triangle where
  P' : ℝ
  Q' : ℝ
  R' : ℝ
  P : ℤ
  Q : ℤ
  R : ℤ

-- Define the properties of a valid triangle
def is_valid_triangle (t : Triangle) : Prop :=
  t.P' + t.Q' + t.R' = 180 ∧
  t.P' > 0 ∧ t.Q' > 0 ∧ t.R' > 0 ∧
  (t.P' - 0.5 ≤ t.P ∧ t.P ≤ t.P' + 0.5) ∧
  (t.Q' - 0.5 ≤ t.Q ∧ t.Q ≤ t.Q' + 0.5) ∧
  (t.R' - 0.5 ≤ t.R ∧ t.R ≤ t.R' + 0.5)

-- Theorem statement
theorem triangle_angle_sum_rounded (t : Triangle) :
  is_valid_triangle t → (t.P + t.Q + t.R = 179 ∨ t.P + t.Q + t.R = 180 ∨ t.P + t.Q + t.R = 181) :=
by sorry

end triangle_angle_sum_rounded_l745_74574


namespace f_injective_f_property_inverse_f_512_l745_74507

/-- A function satisfying f(5) = 2 and f(2x) = 2f(x) for all x -/
def f : ℝ → ℝ :=
  sorry

/-- f is injective -/
theorem f_injective : Function.Injective f :=
  sorry

/-- The inverse function of f -/
noncomputable def f_inv : ℝ → ℝ := Function.invFun f

theorem f_property (x : ℝ) : f (2 * x) = 2 * f x :=
  sorry

axiom f_5 : f 5 = 2

/-- The main theorem: f⁻¹(512) = 1280 -/
theorem inverse_f_512 : f_inv 512 = 1280 := by
  sorry

end f_injective_f_property_inverse_f_512_l745_74507


namespace arithmetic_sequence_2017th_term_l745_74559

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

def geometric_sequence (b : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, b (n + 1) = b n * r

theorem arithmetic_sequence_2017th_term 
  (a : ℕ → ℝ) 
  (h_arith : arithmetic_sequence a)
  (h_a1 : a 1 = 2)
  (h_geom : geometric_sequence (λ n => match n with
    | 1 => a 1 - 1
    | 2 => a 3
    | 3 => a 5 + 5
    | _ => 0
  )) :
  a 2017 = 1010 := by
sorry

end arithmetic_sequence_2017th_term_l745_74559


namespace descending_order_proof_l745_74529

def original_numbers : List ℝ := [1.64, 2.1, 0.09, 1.2]
def sorted_numbers : List ℝ := [2.1, 1.64, 1.2, 0.09]

theorem descending_order_proof :
  (sorted_numbers.zip (sorted_numbers.tail!)).all (fun (a, b) => a ≥ b) ∧
  sorted_numbers.toFinset = original_numbers.toFinset :=
by sorry

end descending_order_proof_l745_74529


namespace composite_function_solution_l745_74568

theorem composite_function_solution (f g : ℝ → ℝ) (a : ℝ) 
  (hf : ∀ x, f x = x / 3 + 2)
  (hg : ∀ x, g x = 5 - 2 * x)
  (h : f (g a) = 4) :
  a = -1/2 := by
sorry

end composite_function_solution_l745_74568


namespace all_statements_imply_target_l745_74557

theorem all_statements_imply_target (p q r : Prop) :
  ((¬p ∧ ¬r ∧ q) → ((p ∧ q) → ¬r)) ∧
  ((p ∧ ¬r ∧ ¬q) → ((p ∧ q) → ¬r)) ∧
  ((¬p ∧ r ∧ q) → ((p ∧ q) → ¬r)) ∧
  ((p ∧ r ∧ ¬q) → ((p ∧ q) → ¬r)) :=
by sorry

end all_statements_imply_target_l745_74557


namespace randy_store_trips_l745_74501

/-- The number of trips Randy makes to the store each month -/
def trips_per_month (initial_amount : ℕ) (amount_per_trip : ℕ) (remaining_amount : ℕ) (months_per_year : ℕ) : ℕ :=
  ((initial_amount - remaining_amount) / amount_per_trip) / months_per_year

/-- Proof that Randy makes 4 trips to the store each month -/
theorem randy_store_trips :
  trips_per_month 200 2 104 12 = 4 := by
  sorry

end randy_store_trips_l745_74501


namespace compound_molecular_weight_l745_74515

/-- Atomic weight of Calcium -/
def Ca_weight : ℝ := 40.08

/-- Atomic weight of Oxygen -/
def O_weight : ℝ := 16.00

/-- Atomic weight of Hydrogen -/
def H_weight : ℝ := 1.008

/-- Atomic weight of Nitrogen -/
def N_weight : ℝ := 14.01

/-- Atomic weight of Carbon-12 -/
def C12_weight : ℝ := 12.00

/-- Atomic weight of Carbon-13 -/
def C13_weight : ℝ := 13.003

/-- Percentage of Carbon-12 in the compound -/
def C12_percentage : ℝ := 0.95

/-- Percentage of Carbon-13 in the compound -/
def C13_percentage : ℝ := 0.05

/-- Average atomic weight of Carbon in the compound -/
def C_avg_weight : ℝ := C12_percentage * C12_weight + C13_percentage * C13_weight

/-- Number of Calcium atoms in the compound -/
def Ca_count : ℕ := 2

/-- Number of Oxygen atoms in the compound -/
def O_count : ℕ := 3

/-- Number of Hydrogen atoms in the compound -/
def H_count : ℕ := 2

/-- Number of Nitrogen atoms in the compound -/
def N_count : ℕ := 1

/-- Number of Carbon atoms in the compound -/
def C_count : ℕ := 1

/-- Molecular weight of the compound -/
def molecular_weight : ℝ :=
  Ca_count * Ca_weight + O_count * O_weight + H_count * H_weight +
  N_count * N_weight + C_count * C_avg_weight

theorem compound_molecular_weight :
  molecular_weight = 156.22615 := by sorry

end compound_molecular_weight_l745_74515


namespace saree_discount_problem_l745_74512

/-- Calculates the second discount percentage given the original price, first discount percentage, and final sale price. -/
def second_discount_percentage (original_price first_discount_percent final_price : ℚ) : ℚ :=
  let price_after_first_discount := original_price * (1 - first_discount_percent / 100)
  let second_discount_amount := price_after_first_discount - final_price
  (second_discount_amount / price_after_first_discount) * 100

/-- Theorem stating that for the given conditions, the second discount percentage is 15%. -/
theorem saree_discount_problem :
  second_discount_percentage 450 20 306 = 15 := by sorry

end saree_discount_problem_l745_74512


namespace not_all_points_follow_linear_relation_l745_74541

-- Define the type for our data points
structure DataPoint where
  n : Nat
  w : Nat

-- Define our dataset
def dataset : List DataPoint := [
  { n := 1, w := 55 },
  { n := 2, w := 110 },
  { n := 3, w := 160 },
  { n := 4, w := 200 },
  { n := 5, w := 254 },
  { n := 6, w := 300 },
  { n := 7, w := 350 }
]

-- Theorem statement
theorem not_all_points_follow_linear_relation :
  ∃ point : DataPoint, point ∈ dataset ∧ point.w ≠ 55 * point.n := by
  sorry


end not_all_points_follow_linear_relation_l745_74541


namespace bracket_six_times_bracket_three_l745_74553

-- Define a function for the square bracket operation
def bracket (x : ℕ) : ℕ :=
  if x % 2 = 0 then
    x / 2 + 1
  else
    2 * x + 1

-- Theorem statement
theorem bracket_six_times_bracket_three : bracket 6 * bracket 3 = 28 := by
  sorry

end bracket_six_times_bracket_three_l745_74553


namespace tangent_line_curve_l745_74526

-- Define the line equation
def line (x y : ℝ) : Prop := x - y + 2 = 0

-- Define the curve equation
def curve (x y a : ℝ) : Prop := y = Real.log x + a

-- Define the tangency condition
def is_tangent (a : ℝ) : Prop :=
  ∃ x y : ℝ, line x y ∧ curve x y a ∧
    (∀ x' y' : ℝ, x' ≠ x → line x' y' → curve x' y' a → (y' - y) / (x' - x) ≠ 1 / x)

-- Theorem statement
theorem tangent_line_curve (a : ℝ) : is_tangent a → a = 3 := by
  sorry

end tangent_line_curve_l745_74526


namespace chord_length_is_sqrt_6_l745_74565

-- Define the circle C
def circle_C (x y : ℝ) : Prop := x + y^2 + 4*x - 4*y + 6 = 0

-- Define the line l
def line_l (k x y : ℝ) : Prop := k*x + y + 4 = 0

-- Define the line m
def line_m (k x y : ℝ) : Prop := y = x + k

-- Theorem statement
theorem chord_length_is_sqrt_6 (k : ℝ) :
  (∃ x y : ℝ, line_l k x y ∧ circle_C x y) →  -- l is a symmetric axis of C
  (∃ x y : ℝ, line_m k x y ∧ circle_C x y) →  -- m intersects C
  (∃ x1 y1 x2 y2 : ℝ, 
    line_m k x1 y1 ∧ circle_C x1 y1 ∧
    line_m k x2 y2 ∧ circle_C x2 y2 ∧
    (x1 - x2)^2 + (y1 - y2)^2 = 6) :=
by sorry

end chord_length_is_sqrt_6_l745_74565


namespace partial_fraction_decomposition_l745_74530

theorem partial_fraction_decomposition :
  ∃! (P Q R : ℚ),
    ∀ (x : ℚ), x ≠ 1 ∧ x ≠ 4 ∧ x ≠ 6 →
      (x^2 - 9) / ((x - 1) * (x - 4) * (x - 6)) =
      P / (x - 1) + Q / (x - 4) + R / (x - 6) ∧
      P = -8/15 ∧ Q = -7/6 ∧ R = 27/10 := by
  sorry

end partial_fraction_decomposition_l745_74530


namespace probability_two_shirts_one_shorts_one_socks_l745_74536

def num_shirts : ℕ := 3
def num_shorts : ℕ := 7
def num_socks : ℕ := 4
def num_selected : ℕ := 4

def total_articles : ℕ := num_shirts + num_shorts + num_socks

def favorable_outcomes : ℕ := (num_shirts.choose 2) * (num_shorts.choose 1) * (num_socks.choose 1)
def total_outcomes : ℕ := total_articles.choose num_selected

theorem probability_two_shirts_one_shorts_one_socks :
  (favorable_outcomes : ℚ) / total_outcomes = 84 / 1001 :=
sorry

end probability_two_shirts_one_shorts_one_socks_l745_74536


namespace linda_coin_fraction_l745_74531

/-- The fraction of Linda's coins representing states that joined the union during 1790-1799 -/
def fraction_of_coins (total_coins : ℕ) (states_joined : ℕ) : ℚ :=
  states_joined / total_coins

/-- Proof that the fraction of Linda's coins representing states from 1790-1799 is 4/15 -/
theorem linda_coin_fraction :
  fraction_of_coins 30 8 = 4 / 15 := by
  sorry

end linda_coin_fraction_l745_74531


namespace angle_of_inclination_special_line_l745_74519

/-- The angle of inclination of a line passing through points (1,0) and (0,-1) is π/4 -/
theorem angle_of_inclination_special_line : 
  let A : ℝ × ℝ := (1, 0)
  let B : ℝ × ℝ := (0, -1)
  let m : ℝ := (B.2 - A.2) / (B.1 - A.1)
  Real.arctan m = π / 4 := by
  sorry

end angle_of_inclination_special_line_l745_74519


namespace total_new_games_is_92_l745_74598

/-- The number of new games Katie has -/
def katie_new_games : ℕ := 84

/-- The number of new games Katie's friends have -/
def friends_new_games : ℕ := 8

/-- The total number of new games Katie and her friends have together -/
def total_new_games : ℕ := katie_new_games + friends_new_games

/-- Theorem stating that the total number of new games is 92 -/
theorem total_new_games_is_92 : total_new_games = 92 := by
  sorry

end total_new_games_is_92_l745_74598


namespace bus_seating_capacity_l745_74534

/-- Calculates the total number of people who can sit in a bus with the given seating arrangement. -/
theorem bus_seating_capacity
  (left_seats : ℕ)
  (right_seats_difference : ℕ)
  (people_per_seat : ℕ)
  (back_seat_capacity : ℕ)
  (h1 : left_seats = 15)
  (h2 : right_seats_difference = 3)
  (h3 : people_per_seat = 3)
  (h4 : back_seat_capacity = 10) :
  left_seats * people_per_seat +
  (left_seats - right_seats_difference) * people_per_seat +
  back_seat_capacity = 91 := by
  sorry

end bus_seating_capacity_l745_74534


namespace product_equals_zero_l745_74561

theorem product_equals_zero (n : ℤ) (h : n = 1) : (n - 2) * (n - 1) * n * (n + 1) * (n + 2) = 0 := by
  sorry

end product_equals_zero_l745_74561


namespace problem_solution_l745_74546

noncomputable def AC (x : ℝ) : ℝ × ℝ := (Real.cos (x/2) + Real.sin (x/2), Real.sin (x/2))

noncomputable def BC (x : ℝ) : ℝ × ℝ := (Real.sin (x/2) - Real.cos (x/2), 2 * Real.cos (x/2))

noncomputable def f (x : ℝ) : ℝ := (AC x).1 * (BC x).1 + (AC x).2 * (BC x).2

theorem problem_solution :
  (∃ (T : ℝ), T > 0 ∧ ∀ (x : ℝ), f (x + T) = f x ∧ ∀ (S : ℝ), S > 0 ∧ (∀ (x : ℝ), f (x + S) = f x) → T ≤ S) ∧
  (∀ (x : ℝ), x ∈ Set.Icc 0 (Real.pi / 2) → f x ≤ 1) ∧
  (∀ (x : ℝ), x ∈ Set.Icc 0 (Real.pi / 2) → f x ≥ -1) ∧
  (∃ (x : ℝ), x ∈ Set.Icc 0 (Real.pi / 2) ∧ f x = 1) ∧
  (∃ (x : ℝ), x ∈ Set.Icc 0 (Real.pi / 2) ∧ f x = -1) ∧
  (∀ (x₁ x₂ : ℝ), x₁ ∈ Set.Ioo Real.pi (3 * Real.pi) ∧ x₂ ∈ Set.Ioo Real.pi (3 * Real.pi) ∧
    f x₁ = Real.sqrt 6 / 2 ∧ f x₂ = Real.sqrt 6 / 2 ∧ x₁ ≠ x₂ →
    x₁ + x₂ = 11 * Real.pi / 2) :=
by sorry

end problem_solution_l745_74546


namespace trigonometric_identity_l745_74524

theorem trigonometric_identity (α : Real) 
  (h1 : Real.tan (α + π/4) = 1/2) 
  (h2 : -π/2 < α) 
  (h3 : α < 0) : 
  Real.sin (2*α) + 2 * (Real.sin α)^2 = -2/5 := by
  sorry

end trigonometric_identity_l745_74524


namespace baseball_distribution_l745_74543

theorem baseball_distribution (total : ℕ) (classes : ℕ) (h1 : total = 43) (h2 : classes = 6) :
  total % classes = 1 := by
  sorry

end baseball_distribution_l745_74543


namespace max_distance_PQ_l745_74540

noncomputable section

-- Define the real parameters m and n
variables (m n : ℝ)

-- Define the lines l₁ and l₂
def l₁ (x y : ℝ) : Prop := m * x - n * y - 5 * m + n = 0
def l₂ (x y : ℝ) : Prop := n * x + m * y - 5 * m - n = 0

-- Define the circle C
def C (x y : ℝ) : Prop := (x + 1)^2 + y^2 = 1

-- Define the intersection point P
def P (x y : ℝ) : Prop := l₁ m n x y ∧ l₂ m n x y

-- Define point Q on circle C
def Q (x y : ℝ) : Prop := C x y

-- State the theorem
theorem max_distance_PQ (hm : m^2 + n^2 ≠ 0) :
  ∃ (px py qx qy : ℝ), P m n px py ∧ Q qx qy ∧
  ∀ (px' py' qx' qy' : ℝ), P m n px' py' → Q qx' qy' →
  (px - qx)^2 + (py - qy)^2 ≤ (6 + 2 * Real.sqrt 2)^2 :=
sorry

end max_distance_PQ_l745_74540


namespace max_area_rectangle_l745_74520

/-- The maximum area of a rectangle with perimeter P is P²/16 -/
theorem max_area_rectangle (P : ℝ) (h : P > 0) : 
  ∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ 2*x + 2*y = P ∧ 
  x*y = P^2/16 ∧ 
  ∀ (a b : ℝ), a > 0 → b > 0 → 2*a + 2*b = P → a*b ≤ P^2/16 := by
sorry

end max_area_rectangle_l745_74520


namespace emily_numbers_l745_74548

theorem emily_numbers (n : ℕ) : 
  (n % 5 = 0 ∧ n % 10 = 0) → 
  (∃ d : ℕ, d < 10 ∧ d ≠ 0 ∧ n / 10 % 10 = d) →
  (∃ count : ℕ, count = 9 ∧ 
    ∀ d : ℕ, d < 10 ∧ d ≠ 0 → 
    ∃ m : ℕ, m % 5 = 0 ∧ m % 10 = 0 ∧ m / 10 % 10 = d) :=
by
  sorry

#check emily_numbers

end emily_numbers_l745_74548


namespace ab_value_l745_74500

theorem ab_value (a b : ℝ) (h1 : a * Real.exp a = Real.exp 2) (h2 : Real.log (b / Real.exp 1) = Real.exp 3 / b) : a * b = Real.exp 3 := by
  sorry

end ab_value_l745_74500


namespace solve_quadratic_equation_l745_74589

theorem solve_quadratic_equation (B : ℝ) :
  5 * B^2 + 5 = 30 → B = Real.sqrt 5 ∨ B = -Real.sqrt 5 := by
  sorry

end solve_quadratic_equation_l745_74589


namespace largest_n_binomial_sum_l745_74594

theorem largest_n_binomial_sum : 
  (∃ n : ℕ, (Nat.choose 10 3 + Nat.choose 10 4 = Nat.choose 11 n) ∧ 
    (∀ m : ℕ, m > n → Nat.choose 10 3 + Nat.choose 10 4 ≠ Nat.choose 11 m)) → 
  (∃ n : ℕ, n = 7 ∧ (Nat.choose 10 3 + Nat.choose 10 4 = Nat.choose 11 n) ∧ 
    (∀ m : ℕ, m > n → Nat.choose 10 3 + Nat.choose 10 4 ≠ Nat.choose 11 m)) :=
by sorry

end largest_n_binomial_sum_l745_74594


namespace exterior_angle_sum_l745_74597

theorem exterior_angle_sum (angle1 angle2 angle3 angle4 : ℝ) :
  angle1 = 100 →
  angle2 = 60 →
  angle3 = 90 →
  angle1 + angle2 + angle3 + angle4 = 360 →
  angle4 = 110 := by
  sorry

end exterior_angle_sum_l745_74597


namespace exactly_two_approve_probability_l745_74584

def approval_rate : ℝ := 0.6

def num_voters : ℕ := 4

def num_approving : ℕ := 2

def probability_exactly_two_approve (p : ℝ) (n : ℕ) (k : ℕ) : ℝ :=
  (Nat.choose n k : ℝ) * p^k * (1 - p)^(n - k)

theorem exactly_two_approve_probability :
  probability_exactly_two_approve approval_rate num_voters num_approving = 0.3456 := by
  sorry

end exactly_two_approve_probability_l745_74584


namespace rectangle_width_l745_74525

theorem rectangle_width (area : ℝ) (length width : ℝ) : 
  area = 63 →
  width = length - 2 →
  area = length * width →
  width = 7 := by
sorry

end rectangle_width_l745_74525


namespace marys_hourly_rate_l745_74510

/-- Represents Mary's work schedule and pay structure --/
structure WorkSchedule where
  maxHours : ℕ
  regularHours : ℕ
  overtimeRate : ℚ
  maxEarnings : ℚ

/-- Calculates the regular hourly rate given a work schedule --/
def regularHourlyRate (schedule : WorkSchedule) : ℚ :=
  let overtimeHours := schedule.maxHours - schedule.regularHours
  let x := schedule.maxEarnings / (schedule.regularHours + overtimeHours * schedule.overtimeRate)
  x

/-- Theorem stating that Mary's regular hourly rate is $8 --/
theorem marys_hourly_rate :
  let schedule := WorkSchedule.mk 80 20 1.25 760
  regularHourlyRate schedule = 8 := by
  sorry

end marys_hourly_rate_l745_74510


namespace picnic_age_problem_l745_74502

theorem picnic_age_problem (initial_count : ℕ) (new_count : ℕ) (new_avg_age : ℝ) (final_avg_age : ℝ) :
  initial_count = 15 →
  new_count = 15 →
  new_avg_age = 15 →
  final_avg_age = 15.5 →
  ∃ (initial_avg_age : ℝ),
    initial_avg_age * initial_count + new_avg_age * new_count = 
    final_avg_age * (initial_count + new_count) ∧
    initial_avg_age = 16 := by
  sorry

end picnic_age_problem_l745_74502


namespace chocolate_candy_price_difference_l745_74556

/-- Proves the difference in cost between a discounted chocolate and a taxed candy bar --/
theorem chocolate_candy_price_difference 
  (initial_money : ℝ)
  (chocolate_price gum_price candy_price soda_price : ℝ)
  (chocolate_discount gum_candy_tax : ℝ) :
  initial_money = 20 →
  chocolate_price = 7 →
  gum_price = 3 →
  candy_price = 2 →
  soda_price = 1.5 →
  chocolate_discount = 0.15 →
  gum_candy_tax = 0.08 →
  chocolate_price * (1 - chocolate_discount) - (candy_price * (1 + gum_candy_tax)) = 3.95 := by
  sorry

end chocolate_candy_price_difference_l745_74556


namespace f_2023_equals_107_l745_74545

-- Define the property of the function f
def has_property (f : ℕ → ℝ) : Prop :=
  ∀ (a b n : ℕ), a > 0 → b > 0 → n > 0 → a + b = 2^n → f a + f b = (n^2 + 1 : ℝ)

-- Theorem statement
theorem f_2023_equals_107 (f : ℕ → ℝ) (h : has_property f) : f 2023 = 107 := by
  sorry

end f_2023_equals_107_l745_74545


namespace smallest_multiple_of_7_greater_than_neg_50_l745_74554

theorem smallest_multiple_of_7_greater_than_neg_50 :
  ∀ n : ℤ, n > -50 ∧ n % 7 = 0 → n ≥ -49 :=
by
  sorry

end smallest_multiple_of_7_greater_than_neg_50_l745_74554


namespace characterize_solution_set_l745_74596

/-- A function satisfying the given functional equation -/
def SatisfiesFunctionalEquation (f : ℝ → ℝ) : Prop :=
  ∃ (n : ℕ), n ≥ 2 ∧ ∀ (x y : ℝ), f (x + y^n) = f x + (f y)^n

/-- The set of functions that satisfy the functional equation -/
def SolutionSet : Set (ℝ → ℝ) :=
  {f | SatisfiesFunctionalEquation f}

/-- The zero function -/
def ZeroFunction : ℝ → ℝ := fun _ ↦ 0

/-- The identity function -/
def IdentityFunction : ℝ → ℝ := fun x ↦ x

/-- The negation function -/
def NegationFunction : ℝ → ℝ := fun x ↦ -x

/-- The main theorem characterizing the solution set -/
theorem characterize_solution_set :
  SolutionSet = {ZeroFunction, IdentityFunction, NegationFunction} := by sorry

end characterize_solution_set_l745_74596


namespace inequality_range_l745_74523

theorem inequality_range (b : ℝ) : 
  (b > 0 ∧ ∃ y : ℝ, |y - 5| + 2 * |y - 2| > b) → 0 < b ∧ b < 3 := by
  sorry

end inequality_range_l745_74523


namespace power_sum_equality_l745_74514

theorem power_sum_equality : (-1 : ℤ) ^ 53 + 3 ^ (2^3 + 5^2 - 7^2) = -1 := by
  sorry

end power_sum_equality_l745_74514


namespace base_conversion_subtraction_l745_74582

/-- Converts a number from base 5 to base 10 -/
def base5ToBase10 (n : ℕ) : ℕ :=
  (n / 10000) * 3125 + ((n / 1000) % 10) * 625 + ((n / 100) % 10) * 125 + ((n / 10) % 10) * 25 + (n % 10) * 5

/-- Converts a number from base 8 to base 10 -/
def base8ToBase10 (n : ℕ) : ℕ :=
  (n / 1000) * 512 + ((n / 100) % 10) * 64 + ((n / 10) % 10) * 8 + (n % 10)

theorem base_conversion_subtraction :
  base5ToBase10 52143 - base8ToBase10 4310 = 1175 := by
  sorry

end base_conversion_subtraction_l745_74582


namespace opposite_of_negative_eleven_l745_74578

theorem opposite_of_negative_eleven : 
  ∀ x : ℤ, x + (-11) = 0 → x = 11 := by
  sorry

end opposite_of_negative_eleven_l745_74578


namespace plot_length_is_58_l745_74573

/-- Represents a rectangular plot with given properties -/
structure RectangularPlot where
  breadth : ℝ
  length : ℝ
  fencing_cost_per_meter : ℝ
  total_fencing_cost : ℝ
  length_breadth_difference : ℝ

/-- Calculates the length of the plot given the conditions -/
def calculate_plot_length (plot : RectangularPlot) : ℝ :=
  plot.breadth + plot.length_breadth_difference

/-- Theorem stating that the length of the plot is 58 meters under given conditions -/
theorem plot_length_is_58 (plot : RectangularPlot) 
  (h1 : plot.length = plot.breadth + 16)
  (h2 : plot.fencing_cost_per_meter = 26.5)
  (h3 : plot.total_fencing_cost = 5300)
  (h4 : plot.length_breadth_difference = 16) : 
  calculate_plot_length plot = 58 := by
  sorry

#eval calculate_plot_length { breadth := 42, length := 58, fencing_cost_per_meter := 26.5, total_fencing_cost := 5300, length_breadth_difference := 16 }

end plot_length_is_58_l745_74573


namespace vector_collinearity_l745_74505

/-- Given vectors m, n, and k in ℝ², prove that if m - 2n is collinear with k, then t = 1 -/
theorem vector_collinearity (m n k : ℝ × ℝ) (t : ℝ) 
  (hm : m = (Real.sqrt 3, 1)) 
  (hn : n = (0, -1)) 
  (hk : k = (t, Real.sqrt 3)) 
  (hcol : ∃ (c : ℝ), c • (m - 2 • n) = k) : 
  t = 1 := by
  sorry

end vector_collinearity_l745_74505


namespace solution_set_y_geq_4_min_value_reciprocal_sum_l745_74549

-- Define the quadratic function
def y (a b x : ℝ) : ℝ := a * x^2 + (b - 2) * x + 3

-- Part 1
theorem solution_set_y_geq_4 (a b : ℝ) :
  (∀ x : ℝ, y a b x > 0 ↔ -1 < x ∧ x < 3) →
  (∀ x : ℝ, y a b x ≥ 4 ↔ x = 1) :=
sorry

-- Part 2
theorem min_value_reciprocal_sum (a b : ℝ) :
  a > 0 →
  b > 0 →
  y a b 1 = 2 →
  (∀ a' b' : ℝ, a' > 0 → b' > 0 → y a' b' 1 = 2 → 1/a' + 4/b' ≥ 1/a + 4/b) →
  1/a + 4/b = 9 :=
sorry

end solution_set_y_geq_4_min_value_reciprocal_sum_l745_74549


namespace sqrt_sum_equals_eleven_sqrt_two_over_six_l745_74599

theorem sqrt_sum_equals_eleven_sqrt_two_over_six :
  Real.sqrt (9 / 2) + Real.sqrt (2 / 9) = 11 * Real.sqrt 2 / 6 := by
  sorry

end sqrt_sum_equals_eleven_sqrt_two_over_six_l745_74599


namespace quadratic_distinct_roots_l745_74551

/-- A quadratic equation x^2 + 5x + k = 0 has distinct real roots if and only if k < 25/4 -/
theorem quadratic_distinct_roots (k : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ x^2 + 5*x + k = 0 ∧ y^2 + 5*y + k = 0) ↔ k < 25/4 := by
  sorry

end quadratic_distinct_roots_l745_74551


namespace student_count_l745_74588

theorem student_count : ℕ :=
  let avg_age : ℕ := 20
  let group1_count : ℕ := 5
  let group1_avg : ℕ := 14
  let group2_count : ℕ := 9
  let group2_avg : ℕ := 16
  let last_student_age : ℕ := 186
  let total_students : ℕ := group1_count + group2_count + 1
  let total_age : ℕ := group1_count * group1_avg + group2_count * group2_avg + last_student_age
  have h1 : avg_age * total_students = total_age := by sorry
  20

end student_count_l745_74588


namespace point_on_line_l745_74560

/-- Given three points M, N, and P in the 2D plane, where P lies on the line passing through M and N,
    prove that the y-coordinate of P is 2. -/
theorem point_on_line (M N P : ℝ × ℝ) : 
  M = (2, -1) → N = (4, 5) → P.1 = 3 → 
  (P.2 - M.2) / (P.1 - M.1) = (N.2 - M.2) / (N.1 - M.1) → 
  P.2 = 2 := by
sorry

end point_on_line_l745_74560


namespace zeros_imply_a_range_l745_74522

/-- The function h(x) = ax² - x - ln(x) has two distinct zeros -/
def has_two_distinct_zeros (a : ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁ > 0 ∧ x₂ > 0 ∧
  a * x₁^2 - x₁ - Real.log x₁ = 0 ∧
  a * x₂^2 - x₂ - Real.log x₂ = 0

/-- If h(x) has two distinct zeros, then 0 < a < 1 -/
theorem zeros_imply_a_range (a : ℝ) (h : a ≠ 0) :
  has_two_distinct_zeros a → 0 < a ∧ a < 1 :=
by sorry

end zeros_imply_a_range_l745_74522


namespace original_laborers_l745_74583

/-- Given a piece of work that can be completed by x laborers in 15 days,
    if 5 laborers are absent and the remaining laborers complete the work in 20 days,
    then x = 20. -/
theorem original_laborers (x : ℕ) 
  (h1 : x * 15 = (x - 5) * 20) : x = 20 := by
  sorry

#check original_laborers

end original_laborers_l745_74583


namespace calculation_proof_l745_74595

theorem calculation_proof : 2 * (-1/4) - |1 - Real.sqrt 3| + (-2023)^0 = 3/2 - Real.sqrt 3 := by
  sorry

end calculation_proof_l745_74595


namespace sin_shift_equivalence_l745_74570

open Real

theorem sin_shift_equivalence (x : ℝ) : 
  sin (2 * x + π / 6) = sin (2 * (x + π / 4) - π / 3) := by sorry

end sin_shift_equivalence_l745_74570


namespace jigi_score_l745_74563

theorem jigi_score (max_score : ℕ) (gibi_percent mike_percent lizzy_percent : ℚ) 
  (average_mark : ℕ) (h1 : max_score = 700) (h2 : gibi_percent = 59/100) 
  (h3 : mike_percent = 99/100) (h4 : lizzy_percent = 67/100) (h5 : average_mark = 490) : 
  (4 * average_mark - (gibi_percent + mike_percent + lizzy_percent) * max_score) / max_score = 55/100 :=
sorry

end jigi_score_l745_74563


namespace equation_solutions_l745_74532

theorem equation_solutions :
  (∀ x : ℝ, (x + 4) * (x - 2) = 3 * (x - 2) ↔ x = -1 ∨ x = 2) ∧
  (∀ x : ℝ, x^2 - x - 3 = 0 ↔ x = (1 + Real.sqrt 13) / 2 ∨ x = (1 - Real.sqrt 13) / 2) := by
  sorry

end equation_solutions_l745_74532


namespace simplify_fraction_l745_74571

theorem simplify_fraction : (180 / 16) * (5 / 120) * (8 / 3) = 5 / 4 := by
  sorry

end simplify_fraction_l745_74571


namespace g_difference_l745_74538

noncomputable def g (n : ℤ) : ℝ :=
  (2 + Real.sqrt 2) / 4 * ((1 + Real.sqrt 2) / 2) ^ n + 
  (2 - Real.sqrt 2) / 4 * ((1 - Real.sqrt 2) / 2) ^ n

theorem g_difference (n : ℤ) : g (n + 1) - g (n - 1) = (Real.sqrt 2 / 2) * g n := by
  sorry

end g_difference_l745_74538


namespace ricks_road_trip_l745_74504

/-- Rick's road trip problem -/
theorem ricks_road_trip (D : ℝ) : 
  D > 0 ∧ 
  40 = D / 2 → 
  D + 2 * D + 40 + 2 * (D + 2 * D + 40) = 840 := by
  sorry

end ricks_road_trip_l745_74504


namespace spinner_probability_l745_74592

theorem spinner_probability (p_A p_B p_C p_D : ℚ) : 
  p_A = 3/8 →
  p_B = 1/4 →
  p_C = p_D →
  p_A + p_B + p_C + p_D = 1 →
  p_C = 3/16 := by
sorry

end spinner_probability_l745_74592


namespace contrapositive_equivalence_l745_74564

theorem contrapositive_equivalence (a b : ℝ) :
  (¬(a = 0 ∧ b = 0) → a^2 + b^2 ≠ 0) ↔ (a^2 + b^2 = 0 → a = 0 ∧ b = 0) :=
by sorry

end contrapositive_equivalence_l745_74564


namespace power_six_mod_eleven_l745_74581

theorem power_six_mod_eleven : 6^2045 % 11 = 10 := by
  sorry

end power_six_mod_eleven_l745_74581


namespace cosine_function_properties_l745_74513

theorem cosine_function_properties (a b c d : ℝ) (ha : a > 0) :
  (∀ x, ∃ y, y = a * Real.cos (b * x + c) + d) →
  (a = 4) →
  (2 * Real.pi / b = Real.pi / 2) →
  (b = 4 ∧ ∀ c₁ c₂, ∃ b', 
    (∀ x, ∃ y, y = a * Real.cos (b' * x + c₁) + d) ∧
    (∀ x, ∃ y, y = a * Real.cos (b' * x + c₂) + d) ∧
    (2 * Real.pi / b' = Real.pi / 2)) :=
by sorry

end cosine_function_properties_l745_74513


namespace quadrilateral_to_square_l745_74577

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a quadrilateral -/
structure Quadrilateral where
  A : Point
  B : Point
  C : Point
  D : Point

/-- Represents a trapezoid -/
structure Trapezoid where
  base1 : ℝ
  base2 : ℝ
  height : ℝ

/-- Function to cut a quadrilateral into two trapezoids -/
def cutQuadrilateral (q : Quadrilateral) : (Trapezoid × Trapezoid) :=
  sorry

/-- Function to check if two trapezoids can form a square -/
def canFormSquare (t1 t2 : Trapezoid) : Prop :=
  sorry

/-- Theorem stating that the quadrilateral can be cut and rearranged into a square -/
theorem quadrilateral_to_square (q : Quadrilateral) :
  ∃ (t1 t2 : Trapezoid), 
    (t1, t2) = cutQuadrilateral q ∧ 
    canFormSquare t1 t2 ∧
    ∃ (side : ℝ), side = t1.height ∧ side * side = t1.base1 * t1.height + t2.base1 * t2.height :=
  sorry

end quadrilateral_to_square_l745_74577


namespace account_balance_first_year_l745_74533

/-- Proves that given an initial deposit and interest accrued, the account balance
    at the end of the first year is the sum of the initial deposit and interest accrued. -/
theorem account_balance_first_year
  (initial_deposit : ℝ)
  (interest_accrued : ℝ)
  (h1 : initial_deposit = 1000)
  (h2 : interest_accrued = 100) :
  initial_deposit + interest_accrued = 1100 := by
  sorry

end account_balance_first_year_l745_74533


namespace ceiling_painting_fraction_l745_74555

def total_ceilings : ℕ := 28
def first_week_ceilings : ℕ := 12
def remaining_ceilings : ℕ := 13

theorem ceiling_painting_fraction :
  (total_ceilings - first_week_ceilings - remaining_ceilings) / first_week_ceilings = 1/4 := by
  sorry

end ceiling_painting_fraction_l745_74555


namespace adolfo_blocks_l745_74585

theorem adolfo_blocks (initial_blocks added_blocks : ℕ) 
  (h1 : initial_blocks = 35)
  (h2 : added_blocks = 30) :
  initial_blocks + added_blocks = 65 := by
  sorry

end adolfo_blocks_l745_74585


namespace next_simultaneous_event_is_180_lcm_9_60_is_180_l745_74580

/-- Represents the interval in minutes between lighting up events -/
def light_interval : ℕ := 9

/-- Represents the interval in minutes between chiming events -/
def chime_interval : ℕ := 60

/-- Calculates the next time both events occur simultaneously -/
def next_simultaneous_event : ℕ := Nat.lcm light_interval chime_interval

/-- Theorem stating that the next simultaneous event occurs after 180 minutes -/
theorem next_simultaneous_event_is_180 : next_simultaneous_event = 180 := by
  sorry

/-- Theorem stating that 180 minutes is the least common multiple of 9 and 60 -/
theorem lcm_9_60_is_180 : Nat.lcm 9 60 = 180 := by
  sorry

end next_simultaneous_event_is_180_lcm_9_60_is_180_l745_74580


namespace inscribed_circle_radius_l745_74579

/-- Given three mutually externally tangent circles with radii a, b, and c,
    the radius r of the inscribed circle satisfies the given equation. -/
theorem inscribed_circle_radius (a b c : ℝ) (ha : a = 3) (hb : b = 6) (hc : c = 18) :
  let r := (1 / a + 1 / b + 1 / c + 2 * Real.sqrt (1 / (a * b) + 1 / (a * c) + 1 / (b * c)))⁻¹
  r = 9 / 8 := by sorry

end inscribed_circle_radius_l745_74579


namespace negative_marks_for_wrong_answer_l745_74528

def total_questions : ℕ := 150
def correct_answers : ℕ := 120
def total_score : ℕ := 420
def correct_score : ℕ := 4

theorem negative_marks_for_wrong_answer :
  ∃ (x : ℚ), 
    (correct_score * correct_answers : ℚ) - 
    (x * (total_questions - correct_answers)) = total_score ∧
    x = 2 := by sorry

end negative_marks_for_wrong_answer_l745_74528


namespace distance_along_stream_is_16_l745_74575

-- Define the boat's speed in still water
def boat_speed : ℝ := 11

-- Define the distance traveled against the stream in one hour
def distance_against_stream : ℝ := 6

-- Define the stream speed
def stream_speed : ℝ := boat_speed - distance_against_stream

-- Define the boat's speed along the stream
def speed_along_stream : ℝ := boat_speed + stream_speed

-- Theorem to prove
theorem distance_along_stream_is_16 : speed_along_stream = 16 := by
  sorry

end distance_along_stream_is_16_l745_74575


namespace father_son_age_sum_l745_74572

/-- Given the ages of a father and son 25 years ago and their current age ratio,
    prove that the sum of their present ages is 300 years. -/
theorem father_son_age_sum : ℕ → ℕ → Prop :=
  fun (s f : ℕ) =>
    (f = 4 * s) →                  -- 25 years ago, father was 4 times as old as son
    (f + 25 = 3 * (s + 25)) →      -- Now, father is 3 times as old as son
    ((s + 25) + (f + 25) = 300)    -- Sum of their present ages is 300

/-- Proof of the theorem -/
lemma prove_father_son_age_sum : ∃ (s f : ℕ), father_son_age_sum s f := by
  sorry

#check prove_father_son_age_sum

end father_son_age_sum_l745_74572


namespace fraction_equality_l745_74506

theorem fraction_equality : ∀ x : ℝ, x ≠ 0 ∧ x^2 + 1 ≠ 0 →
  (x^2 + 5*x - 6) / (x^4 + x^2) = (-6 : ℝ) / x^2 + (0*x + 7) / (x^2 + 1) := by
  sorry

end fraction_equality_l745_74506


namespace derivative_at_pi_third_l745_74511

theorem derivative_at_pi_third (f : ℝ → ℝ) (hf : Differentiable ℝ f) 
  (h : ∀ x, f x = x^2 * (deriv f (π/3)) + Real.sin x) : 
  deriv f (π/3) = 3 / (6 - 4*π) := by
  sorry

end derivative_at_pi_third_l745_74511


namespace rectangular_field_width_l745_74562

theorem rectangular_field_width (width length : ℝ) (perimeter : ℝ) : 
  length = (7 / 5) * width →
  perimeter = 2 * length + 2 * width →
  perimeter = 240 →
  width = 50 := by
sorry

end rectangular_field_width_l745_74562


namespace gabby_fruit_ratio_l745_74503

/-- Represents the number of fruits Gabby harvested -/
structure FruitHarvest where
  watermelons : ℕ
  peaches : ℕ
  plums : ℕ

/-- Conditions of Gabby's fruit harvest -/
def gabbyHarvest : FruitHarvest where
  watermelons := 1
  peaches := 13
  plums := 39

theorem gabby_fruit_ratio :
  let h := gabbyHarvest
  h.watermelons = 1 ∧
  h.peaches = h.watermelons + 12 ∧
  h.watermelons + h.peaches + h.plums = 53 →
  h.plums / h.peaches = 3 := by
  sorry

end gabby_fruit_ratio_l745_74503


namespace jack_has_forty_dollars_l745_74518

/-- Represents the cost of a pair of socks in dollars -/
def sock_cost : ℚ := 9.5

/-- Represents the cost of the soccer shoes in dollars -/
def shoe_cost : ℚ := 92

/-- Represents the additional amount Jack needs in dollars -/
def additional_needed : ℚ := 71

/-- Calculates Jack's initial money based on the given costs and additional amount needed -/
def jack_initial_money : ℚ :=
  2 * sock_cost + shoe_cost - additional_needed

/-- Theorem stating that Jack's initial money is $40 -/
theorem jack_has_forty_dollars :
  jack_initial_money = 40 := by sorry

end jack_has_forty_dollars_l745_74518


namespace ivan_apple_purchase_l745_74552

theorem ivan_apple_purchase (mini_pies : ℕ) (apples_per_mini_pie : ℚ) (leftover_apples : ℕ) 
  (h1 : mini_pies = 24)
  (h2 : apples_per_mini_pie = 1/2)
  (h3 : leftover_apples = 36) :
  (mini_pies : ℚ) * apples_per_mini_pie + leftover_apples = 48 := by
  sorry

end ivan_apple_purchase_l745_74552


namespace census_objects_eq_population_l745_74537

/-- The entirety of objects under investigation in a census -/
def census_objects : Type := Unit

/-- The term "population" in statistical context -/
def population : Type := Unit

/-- Theorem stating that census objects are equivalent to population -/
theorem census_objects_eq_population : census_objects ≃ population := sorry

end census_objects_eq_population_l745_74537


namespace store_prices_l745_74535

def price_X : ℝ := 80 * (1 + 0.12)
def price_Y : ℝ := price_X * (1 - 0.15)
def price_Z : ℝ := price_Y * (1 + 0.25)

theorem store_prices :
  price_X = 89.6 ∧ price_Y = 76.16 ∧ price_Z = 95.20 := by
  sorry

end store_prices_l745_74535


namespace lawn_mowing_time_l745_74587

theorem lawn_mowing_time (mary_rate tom_rate : ℚ) (mary_time : ℚ) : 
  mary_rate = 1 / 3 →
  tom_rate = 1 / 6 →
  mary_time = 2 →
  (1 - mary_rate * mary_time) / tom_rate = 2 :=
by sorry

end lawn_mowing_time_l745_74587


namespace defect_probability_l745_74576

/-- The probability of a randomly chosen unit being defective from two machines -/
theorem defect_probability
  (machine_a_ratio : ℝ)
  (machine_b_ratio : ℝ)
  (machine_a_defect_rate : ℝ)
  (machine_b_defect_rate : ℝ)
  (h1 : machine_a_ratio = 0.4)
  (h2 : machine_b_ratio = 0.6)
  (h3 : machine_a_ratio + machine_b_ratio = 1)
  (h4 : machine_a_defect_rate = 9 / 1000)
  (h5 : machine_b_defect_rate = 1 / 50) :
  machine_a_ratio * machine_a_defect_rate + machine_b_ratio * machine_b_defect_rate = 0.0156 :=
by sorry


end defect_probability_l745_74576


namespace larger_segment_is_70_l745_74542

/-- A triangle with sides 40, 50, and 90 units -/
structure Triangle where
  side_a : ℝ
  side_b : ℝ
  side_c : ℝ
  ha : side_a = 40
  hb : side_b = 50
  hc : side_c = 90

/-- The altitude dropped on the side of length 90 -/
def altitude (t : Triangle) : ℝ := sorry

/-- The larger segment cut off on the side of length 90 -/
def larger_segment (t : Triangle) : ℝ := sorry

/-- Theorem stating that the larger segment is 70 units -/
theorem larger_segment_is_70 (t : Triangle) : larger_segment t = 70 := by sorry

end larger_segment_is_70_l745_74542


namespace roger_shelves_theorem_l745_74547

def shelves_needed (total_books : ℕ) (books_taken : ℕ) (books_per_shelf : ℕ) : ℕ :=
  let remaining_books := total_books - books_taken
  (remaining_books + books_per_shelf - 1) / books_per_shelf

theorem roger_shelves_theorem :
  shelves_needed 24 3 4 = 6 := by
  sorry

end roger_shelves_theorem_l745_74547


namespace quadratic_equation_solution_l745_74521

theorem quadratic_equation_solution (b : ℝ) : 
  (2 * (-5)^2 + b * (-5) - 20 = 0) → b = 6 := by
  sorry

end quadratic_equation_solution_l745_74521


namespace intersection_is_empty_l745_74544

-- Define the sets A and B
def A : Set String := {s | s = "line"}
def B : Set String := {s | s = "ellipse"}

-- Theorem statement
theorem intersection_is_empty : A ∩ B = ∅ := by
  sorry

end intersection_is_empty_l745_74544


namespace residue_11_2048_mod_17_l745_74558

theorem residue_11_2048_mod_17 : 11^2048 % 17 = 1 := by
  sorry

end residue_11_2048_mod_17_l745_74558


namespace equation_solution_l745_74569

theorem equation_solution :
  ∃! x : ℚ, x ≠ -3 ∧ (x^2 + 3*x + 4) / (x + 3) = x + 6 :=
by
  use -7/3
  sorry

end equation_solution_l745_74569


namespace vector_from_origin_to_line_l745_74539

/-- A line parameterized by t -/
structure ParametricLine where
  x : ℝ → ℝ
  y : ℝ → ℝ

/-- The given line -/
def givenLine : ParametricLine where
  x := λ t => 4 * t + 2
  y := λ t => t + 2

/-- Check if a vector is parallel to another vector -/
def isParallel (v w : ℝ × ℝ) : Prop :=
  ∃ (k : ℝ), v.1 = k * w.1 ∧ v.2 = k * w.2

/-- Check if a point lies on the given line -/
def liesOnLine (p : ℝ × ℝ) : Prop :=
  ∃ (t : ℝ), p.1 = givenLine.x t ∧ p.2 = givenLine.y t

theorem vector_from_origin_to_line :
  liesOnLine (6, 3) ∧ isParallel (6, 3) (2, 1) := by
  sorry

end vector_from_origin_to_line_l745_74539


namespace maria_towels_l745_74567

theorem maria_towels (green_towels white_towels given_to_mother : ℕ) 
  (h1 : green_towels = 58)
  (h2 : white_towels = 43)
  (h3 : given_to_mother = 87) :
  green_towels + white_towels - given_to_mother = 14 :=
by sorry

end maria_towels_l745_74567


namespace lucy_crayons_count_l745_74591

/-- The number of crayons Willy has -/
def willys_crayons : ℕ := 5092

/-- The difference between Willy's and Lucy's crayons -/
def difference : ℕ := 1121

/-- The number of crayons Lucy has -/
def lucys_crayons : ℕ := willys_crayons - difference

theorem lucy_crayons_count : lucys_crayons = 3971 := by
  sorry

end lucy_crayons_count_l745_74591


namespace rs_length_l745_74593

-- Define the triangle XYZ
structure Triangle :=
  (X Y Z : ℝ × ℝ)

-- Define the properties of the triangle
def isValidTriangle (t : Triangle) : Prop :=
  let XY := dist t.X t.Y
  let YZ := dist t.Y t.Z
  let ZX := dist t.Z t.X
  XY = 13 ∧ YZ = 14 ∧ ZX = 15

-- Define the median XM
def isMedian (t : Triangle) (M : ℝ × ℝ) : Prop :=
  dist t.X M = dist M ((t.Y.1 + t.Z.1, t.Y.2 + t.Z.2))

-- Define points G and F
def isOnSide (A B P : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, 0 < k ∧ k < 1 ∧ P = (k * A.1 + (1 - k) * B.1, k * A.2 + (1 - k) * B.2)

-- Define angle bisectors
def isAngleBisector (t : Triangle) (P : ℝ × ℝ) (V : ℝ × ℝ) : Prop :=
  ∃ G F : ℝ × ℝ, 
    isOnSide t.Z t.X G ∧ 
    isOnSide t.X t.Y F ∧
    dist t.Y G * dist t.Z V = dist t.Z G * dist t.Y V ∧
    dist t.Z F * dist t.X V = dist t.X F * dist t.Z V

-- Define the theorem
theorem rs_length (t : Triangle) (M R S : ℝ × ℝ) :
  isValidTriangle t →
  isMedian t M →
  isAngleBisector t R t.Y →
  isAngleBisector t S t.Z →
  dist R S = 129 / 203 :=
sorry


end rs_length_l745_74593


namespace convex_polygons_contain_center_l745_74517

-- Define a square
structure Square where
  side : ℝ
  center : ℝ × ℝ

-- Define a convex polygon
structure ConvexPolygon where
  vertices : List (ℝ × ℝ)
  is_convex : Bool

-- Define a function to check if a point is inside a polygon
def is_inside (p : ℝ × ℝ) (poly : ConvexPolygon) : Prop :=
  sorry

-- Define a function to check if a polygon is inside a square
def is_inside_square (poly : ConvexPolygon) (sq : Square) : Prop :=
  sorry

-- Theorem statement
theorem convex_polygons_contain_center 
  (sq : Square) 
  (poly1 poly2 poly3 : ConvexPolygon) 
  (h1 : is_inside_square poly1 sq)
  (h2 : is_inside_square poly2 sq)
  (h3 : is_inside_square poly3 sq) :
  is_inside sq.center poly1 ∧ is_inside sq.center poly2 ∧ is_inside sq.center poly3 :=
sorry

end convex_polygons_contain_center_l745_74517


namespace work_completion_time_l745_74590

theorem work_completion_time (original_laborers : ℕ) (absent_laborers : ℕ) (actual_days : ℕ) : 
  original_laborers = 20 → 
  absent_laborers = 5 → 
  actual_days = 20 → 
  ∃ (original_days : ℕ), 
    original_days * original_laborers = actual_days * (original_laborers - absent_laborers) ∧ 
    original_days = 15 := by
  sorry

end work_completion_time_l745_74590


namespace xiao_hong_math_probability_expected_value_X_xiao_hong_more_likely_math_noon_l745_74586

-- Define the students
inductive Student : Type
| XiaoHong : Student
| XiaoMing : Student

-- Define the subjects
inductive Subject : Type
| Math : Subject
| Physics : Subject

-- Define the time of day
inductive TimeOfDay : Type
| Noon : TimeOfDay
| Evening : TimeOfDay

-- Define the choice of subjects for a day
structure DailyChoice :=
  (noon : Subject)
  (evening : Subject)

-- Define the probabilities for each student's choices
def choice_probability (s : Student) (dc : DailyChoice) : ℚ :=
  match s, dc with
  | Student.XiaoHong, ⟨Subject.Math, Subject.Math⟩ => 1/4
  | Student.XiaoHong, ⟨Subject.Math, Subject.Physics⟩ => 1/5
  | Student.XiaoHong, ⟨Subject.Physics, Subject.Math⟩ => 7/20
  | Student.XiaoHong, ⟨Subject.Physics, Subject.Physics⟩ => 1/10
  | Student.XiaoMing, ⟨Subject.Math, Subject.Math⟩ => 1/5
  | Student.XiaoMing, ⟨Subject.Math, Subject.Physics⟩ => 1/4
  | Student.XiaoMing, ⟨Subject.Physics, Subject.Math⟩ => 3/20
  | Student.XiaoMing, ⟨Subject.Physics, Subject.Physics⟩ => 3/10

-- Define the number of subjects chosen in a day
def subjects_chosen (s : Student) (dc : DailyChoice) : ℕ :=
  match dc with
  | ⟨Subject.Math, Subject.Math⟩ => 2
  | ⟨Subject.Math, Subject.Physics⟩ => 2
  | ⟨Subject.Physics, Subject.Math⟩ => 2
  | ⟨Subject.Physics, Subject.Physics⟩ => 2

-- Theorem 1: Probability of Xiao Hong choosing math for both noon and evening for exactly 3 out of 5 days
theorem xiao_hong_math_probability : 
  (Finset.sum (Finset.range 6) (λ k => if k = 3 then Nat.choose 5 k * (1/4)^k * (3/4)^(5-k) else 0)) = 45/512 :=
sorry

-- Theorem 2: Expected value of X
theorem expected_value_X :
  (1/100 * 0 + 33/200 * 1 + 33/40 * 2) = 363/200 :=
sorry

-- Theorem 3: Xiao Hong is more likely to choose math at noon when doing physics in the evening
theorem xiao_hong_more_likely_math_noon :
  (choice_probability Student.XiaoHong ⟨Subject.Math, Subject.Physics⟩) / 
  (choice_probability Student.XiaoHong ⟨Subject.Physics, Subject.Physics⟩ + 
   choice_probability Student.XiaoHong ⟨Subject.Math, Subject.Physics⟩) >
  (choice_probability Student.XiaoMing ⟨Subject.Math, Subject.Physics⟩) / 
  (choice_probability Student.XiaoMing ⟨Subject.Physics, Subject.Physics⟩ + 
   choice_probability Student.XiaoMing ⟨Subject.Math, Subject.Physics⟩) :=
sorry

end xiao_hong_math_probability_expected_value_X_xiao_hong_more_likely_math_noon_l745_74586


namespace solution_is_two_lines_l745_74509

-- Define the equation
def equation (x y : ℝ) : Prop := (x + y)^2 = x^2 + y^2 + 4*x

-- Define the solution set
def solution_set : Set (ℝ × ℝ) :=
  {p | equation p.1 p.2}

-- Define the two lines
def y_axis : Set (ℝ × ℝ) := {p | p.1 = 0}
def horizontal_line : Set (ℝ × ℝ) := {p | p.2 = 2}

-- Theorem statement
theorem solution_is_two_lines :
  solution_set = y_axis ∪ horizontal_line :=
sorry

end solution_is_two_lines_l745_74509


namespace expression_evaluation_1_expression_evaluation_2_l745_74566

theorem expression_evaluation_1 : (1 * (-4.5) - (-5 - (2/3)) - 2.5 - (7 + (2/3))) = -9 := by sorry

theorem expression_evaluation_2 : (-4^2 / (-2)^3 - (4/9) * (-3/2)^2) = 1 := by sorry

end expression_evaluation_1_expression_evaluation_2_l745_74566
