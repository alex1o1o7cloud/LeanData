import Mathlib

namespace NUMINAMATH_CALUDE_loyalty_program_benefits_l3146_314616

-- Define the structure for a bank
structure Bank where
  cardUsage : ℝ
  customerLoyalty : ℝ
  transactionVolume : ℝ

-- Define the structure for the Central Bank
structure CentralBank where
  nationalPaymentSystemUsage : ℝ
  consumerSpending : ℝ

-- Define the effect of the loyalty program
def loyaltyProgramEffect (bank : Bank) (centralBank : CentralBank) : Bank × CentralBank :=
  let newBank : Bank := {
    cardUsage := bank.cardUsage * 1.2,
    customerLoyalty := bank.customerLoyalty * 1.15,
    transactionVolume := bank.transactionVolume * 1.25
  }
  let newCentralBank : CentralBank := {
    nationalPaymentSystemUsage := centralBank.nationalPaymentSystemUsage * 1.3,
    consumerSpending := centralBank.consumerSpending * 1.1
  }
  (newBank, newCentralBank)

-- Theorem stating the benefits of the loyalty program
theorem loyalty_program_benefits 
  (bank : Bank) 
  (centralBank : CentralBank) :
  let (newBank, newCentralBank) := loyaltyProgramEffect bank centralBank
  newBank.cardUsage > bank.cardUsage ∧
  newBank.customerLoyalty > bank.customerLoyalty ∧
  newBank.transactionVolume > bank.transactionVolume ∧
  newCentralBank.nationalPaymentSystemUsage > centralBank.nationalPaymentSystemUsage ∧
  newCentralBank.consumerSpending > centralBank.consumerSpending :=
by
  sorry


end NUMINAMATH_CALUDE_loyalty_program_benefits_l3146_314616


namespace NUMINAMATH_CALUDE_ball_bounce_ratio_l3146_314696

theorem ball_bounce_ratio (h₀ : ℝ) (h₅ : ℝ) (r : ℝ) :
  h₀ = 96 →
  h₅ = 3 →
  h₅ = h₀ * r^5 →
  r = Real.sqrt 2 / 4 := by
sorry

end NUMINAMATH_CALUDE_ball_bounce_ratio_l3146_314696


namespace NUMINAMATH_CALUDE_faster_train_speed_l3146_314638

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

end NUMINAMATH_CALUDE_faster_train_speed_l3146_314638


namespace NUMINAMATH_CALUDE_melody_cutouts_l3146_314676

/-- Given that Melody planned to paste 4 cut-outs on each card and made 6 cards in total,
    prove that the total number of cut-outs she made is 24. -/
theorem melody_cutouts (cutouts_per_card : ℕ) (total_cards : ℕ) 
  (h1 : cutouts_per_card = 4) 
  (h2 : total_cards = 6) : 
  cutouts_per_card * total_cards = 24 := by
  sorry

end NUMINAMATH_CALUDE_melody_cutouts_l3146_314676


namespace NUMINAMATH_CALUDE_parabola_shift_theorem_l3146_314690

/-- Represents a parabola in the form y = ax^2 + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Shifts a parabola horizontally and vertically -/
def shift (p : Parabola) (h : ℝ) (v : ℝ) : Parabola :=
  { a := p.a
    b := p.b - 2 * p.a * h
    c := p.a * h^2 - p.b * h + p.c + v }

theorem parabola_shift_theorem (p : Parabola) :
  p.a = 1 ∧ p.b = -4 ∧ p.c = -4 →
  let p' := shift p 3 3
  p'.a = 1 ∧ p'.b = 2 ∧ p'.c = -5 := by sorry

end NUMINAMATH_CALUDE_parabola_shift_theorem_l3146_314690


namespace NUMINAMATH_CALUDE_y_plus_2z_positive_l3146_314632

theorem y_plus_2z_positive (x y z : ℝ) 
  (hx : 0 < x ∧ x < 2) 
  (hy : -2 < y ∧ y < 0) 
  (hz : 0 < z ∧ z < 3) : 
  y + 2*z > 0 := by
  sorry

end NUMINAMATH_CALUDE_y_plus_2z_positive_l3146_314632


namespace NUMINAMATH_CALUDE_cards_selection_count_l3146_314612

/-- The number of ways to select 3 cards from 12 cards (3 each of red, yellow, green, and blue) 
    such that they are not all the same color and there is at most 1 blue card. -/
def select_cards : ℕ := sorry

/-- The total number of cards -/
def total_cards : ℕ := 12

/-- The number of cards of each color -/
def cards_per_color : ℕ := 3

/-- The number of colors -/
def num_colors : ℕ := 4

/-- The number of cards to be selected -/
def cards_to_select : ℕ := 3

theorem cards_selection_count : 
  select_cards = Nat.choose total_cards cards_to_select - 
                 num_colors - 
                 (Nat.choose cards_per_color 2 * Nat.choose (total_cards - cards_per_color) 1) := by
  sorry

end NUMINAMATH_CALUDE_cards_selection_count_l3146_314612


namespace NUMINAMATH_CALUDE_pills_remaining_l3146_314641

def initial_pills : ℕ := 200
def daily_dose : ℕ := 12
def days : ℕ := 14

theorem pills_remaining : initial_pills - (daily_dose * days) = 32 := by
  sorry

end NUMINAMATH_CALUDE_pills_remaining_l3146_314641


namespace NUMINAMATH_CALUDE_best_sampling_methods_l3146_314622

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

end NUMINAMATH_CALUDE_best_sampling_methods_l3146_314622


namespace NUMINAMATH_CALUDE_total_eggs_theorem_l3146_314692

/-- Represents the number of eggs used for a family member's breakfast on a given day type --/
structure EggUsage where
  children : ℕ  -- eggs per child
  husband : ℕ   -- eggs for husband
  lisa : ℕ      -- eggs for Lisa

/-- Represents the egg usage patterns for different days of the week --/
structure WeeklyEggUsage where
  monday_tuesday : EggUsage
  wednesday : EggUsage
  thursday : EggUsage
  friday : EggUsage

/-- Calculates the total eggs used in a year based on the given parameters --/
def total_eggs_per_year (
  num_children : ℕ
  ) (weekly_usage : WeeklyEggUsage
  ) (num_holidays : ℕ
  ) (holiday_usage : EggUsage
  ) : ℕ :=
  sorry

/-- The main theorem stating the total number of eggs used in a year --/
theorem total_eggs_theorem : 
  total_eggs_per_year 
    4  -- number of children
    {  -- weekly egg usage
      monday_tuesday := { children := 2, husband := 3, lisa := 2 },
      wednesday := { children := 3, husband := 4, lisa := 3 },
      thursday := { children := 1, husband := 2, lisa := 1 },
      friday := { children := 2, husband := 3, lisa := 2 }
    }
    8  -- number of holidays
    { children := 2, husband := 2, lisa := 2 }  -- holiday egg usage
  = 3476 := by
  sorry

end NUMINAMATH_CALUDE_total_eggs_theorem_l3146_314692


namespace NUMINAMATH_CALUDE_quadratic_equation_coefficient_l3146_314636

theorem quadratic_equation_coefficient (a : ℝ) : 
  (∀ x, ∃ y, y = (a - 3) * x^2 - 3 * x - 4) → a ≠ 3 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_coefficient_l3146_314636


namespace NUMINAMATH_CALUDE_max_students_planting_trees_l3146_314686

theorem max_students_planting_trees (a b : ℕ) : 
  3 * a + 5 * b = 115 → a + b ≤ 37 := by
  sorry

end NUMINAMATH_CALUDE_max_students_planting_trees_l3146_314686


namespace NUMINAMATH_CALUDE_deal_or_no_deal_probability_l3146_314680

theorem deal_or_no_deal_probability (total_boxes : ℕ) (desired_boxes : ℕ) (eliminated_boxes : ℕ) : 
  total_boxes = 30 →
  desired_boxes = 6 →
  eliminated_boxes = 18 →
  (desired_boxes : ℚ) / (total_boxes - eliminated_boxes : ℚ) ≥ 1/2 :=
by sorry

end NUMINAMATH_CALUDE_deal_or_no_deal_probability_l3146_314680


namespace NUMINAMATH_CALUDE_inequality_condition_l3146_314617

def f (x : ℝ) := x^2 + 3*x + 2

theorem inequality_condition (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∀ x : ℝ, |x + 2| < b → |f x + 4| < a) ↔ b ≤ a/7 := by sorry

end NUMINAMATH_CALUDE_inequality_condition_l3146_314617


namespace NUMINAMATH_CALUDE_intersection_equals_specific_set_l3146_314654

-- Define the set P
def P : Set ℝ := {x | ∃ k : ℤ, 2 * k * Real.pi ≤ x ∧ x ≤ (2 * k + 1) * Real.pi}

-- Define the set Q
def Q : Set ℝ := {α | -4 ≤ α ∧ α ≤ 4}

-- Define the intersection set
def intersection_set : Set ℝ := {α | (-4 ≤ α ∧ α ≤ -Real.pi) ∨ (0 ≤ α ∧ α ≤ Real.pi)}

-- Theorem statement
theorem intersection_equals_specific_set : P ∩ Q = intersection_set := by sorry

end NUMINAMATH_CALUDE_intersection_equals_specific_set_l3146_314654


namespace NUMINAMATH_CALUDE_intersection_N_complement_M_l3146_314669

open Set Real

-- Define the sets M and N
def M : Set ℝ := {x | ∃ y, y = Real.log (1 - 2/x)}
def N : Set ℝ := {x | ∃ y, y = Real.sqrt (x - 1)}

-- State the theorem
theorem intersection_N_complement_M :
  N ∩ (univ \ M) = Icc 1 2 := by sorry

end NUMINAMATH_CALUDE_intersection_N_complement_M_l3146_314669


namespace NUMINAMATH_CALUDE_output_for_twelve_l3146_314661

def function_machine (input : ℕ) : ℕ :=
  let step1 := input * 3
  if step1 > 25 then
    step1 - 5
  else
    step1 * 2

theorem output_for_twelve : function_machine 12 = 31 := by
  sorry

end NUMINAMATH_CALUDE_output_for_twelve_l3146_314661


namespace NUMINAMATH_CALUDE_sum_of_solutions_quadratic_l3146_314675

theorem sum_of_solutions_quadratic (x : ℝ) : 
  (x^2 = 9*x - 20) → (∃ y : ℝ, y^2 = 9*y - 20 ∧ x + y = 9) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_solutions_quadratic_l3146_314675


namespace NUMINAMATH_CALUDE_base_five_digits_of_1234_l3146_314604

theorem base_five_digits_of_1234 : ∃ n : ℕ, n > 0 ∧ 5^(n-1) ≤ 1234 ∧ 1234 < 5^n :=
  sorry

end NUMINAMATH_CALUDE_base_five_digits_of_1234_l3146_314604


namespace NUMINAMATH_CALUDE_dog_distance_l3146_314678

theorem dog_distance (s : ℝ) (ivan_speed dog_speed : ℝ) : 
  s > 0 → 
  ivan_speed > 0 → 
  dog_speed > 0 → 
  s = 3 → 
  dog_speed = 3 * ivan_speed → 
  (∃ t : ℝ, t > 0 ∧ ivan_speed * t = s / 4 ∧ dog_speed * t = 3 * s / 4) → 
  (dog_speed * (s / ivan_speed)) = 9 :=
by sorry

end NUMINAMATH_CALUDE_dog_distance_l3146_314678


namespace NUMINAMATH_CALUDE_inserted_numbers_sum_l3146_314608

theorem inserted_numbers_sum (a b : ℝ) : 
  a > 0 ∧ b > 0 ∧ 
  (∃ d : ℝ, a = 2 + d ∧ b = 2 + 2*d) ∧ 
  (∃ r : ℝ, b = a * r ∧ 18 = b * r) →
  a + b = 16 := by
sorry

end NUMINAMATH_CALUDE_inserted_numbers_sum_l3146_314608


namespace NUMINAMATH_CALUDE_quadratic_degeneracy_l3146_314663

/-- Represents a quadratic equation ax² + bx + c = 0 -/
structure QuadraticEquation where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Represents the roots of an equation -/
inductive Root
  | Finite (x : ℝ)
  | Infinity

/-- 
Given a quadratic equation ax² + bx + c = 0 where a = 0,
prove that it has one finite root -c/b and one root at infinity.
-/
theorem quadratic_degeneracy (eq : QuadraticEquation) (h : eq.a = 0) :
  ∃ (r₁ r₂ : Root), 
    r₁ = Root.Finite (-eq.c / eq.b) ∧ 
    r₂ = Root.Infinity ∧
    eq.b * (-eq.c / eq.b) + eq.c = 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_degeneracy_l3146_314663


namespace NUMINAMATH_CALUDE_wall_length_calculation_l3146_314639

/-- Given a square mirror and a rectangular wall, if the mirror's area is exactly half the wall's area,
    prove that the length of the wall is approximately 43 inches. -/
theorem wall_length_calculation (mirror_side : ℝ) (wall_width : ℝ) : 
  mirror_side = 34 →
  wall_width = 54 →
  (mirror_side ^ 2) * 2 = wall_width * (round ((mirror_side ^ 2) * 2 / wall_width)) :=
by sorry

end NUMINAMATH_CALUDE_wall_length_calculation_l3146_314639


namespace NUMINAMATH_CALUDE_remainder_problem_l3146_314689

theorem remainder_problem (N : ℤ) : ∃ (k : ℤ), N = 296 * k + 75 → ∃ (m : ℤ), N = 37 * m + 1 := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l3146_314689


namespace NUMINAMATH_CALUDE_prob_all_red_first_is_half_l3146_314687

/-- The number of red chips in the hat -/
def num_red_chips : ℕ := 3

/-- The number of green chips in the hat -/
def num_green_chips : ℕ := 3

/-- The total number of chips in the hat -/
def total_chips : ℕ := num_red_chips + num_green_chips

/-- The probability of drawing all red chips before all green chips -/
def prob_all_red_first : ℚ :=
  (Nat.choose (total_chips - 1) num_green_chips) / (Nat.choose total_chips num_red_chips)

/-- Theorem stating that the probability of drawing all red chips first is 1/2 -/
theorem prob_all_red_first_is_half : prob_all_red_first = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_prob_all_red_first_is_half_l3146_314687


namespace NUMINAMATH_CALUDE_convex_ngon_division_constant_l3146_314613

/-- A convex n-gon can be divided into triangles using non-intersecting diagonals -/
structure ConvexNGonDivision (n : ℕ) where
  (n_ge_3 : n ≥ 3)
  (triangles : ℕ)
  (diagonals : ℕ)

/-- The number of triangles and diagonals in any division of a convex n-gon is constant -/
theorem convex_ngon_division_constant (n : ℕ) (d : ConvexNGonDivision n) :
  d.triangles = n - 2 ∧ d.diagonals = n - 3 :=
sorry

end NUMINAMATH_CALUDE_convex_ngon_division_constant_l3146_314613


namespace NUMINAMATH_CALUDE_similar_triangles_ab_length_l3146_314606

/-- Two triangles are similar -/
def similar_triangles (t1 t2 : Set (Fin 3 → ℝ × ℝ)) : Prop := sorry

theorem similar_triangles_ab_length :
  ∀ (P Q R X Y Z A B C : ℝ × ℝ),
  let pqr : Set (Fin 3 → ℝ × ℝ) := {![P, Q, R]}
  let xyz : Set (Fin 3 → ℝ × ℝ) := {![X, Y, Z]}
  let abc : Set (Fin 3 → ℝ × ℝ) := {![A, B, C]}
  similar_triangles pqr xyz →
  similar_triangles xyz abc →
  dist P Q = 8 →
  dist Q R = 16 →
  dist B C = 24 →
  dist Y Z = 12 →
  dist A B = 12 :=
by sorry

end NUMINAMATH_CALUDE_similar_triangles_ab_length_l3146_314606


namespace NUMINAMATH_CALUDE_bob_spending_is_26_l3146_314670

-- Define the prices and quantities
def bread_price : ℚ := 2
def bread_quantity : ℕ := 4
def cheese_price : ℚ := 6
def cheese_quantity : ℕ := 2
def chocolate_price : ℚ := 3
def chocolate_quantity : ℕ := 3
def oil_price : ℚ := 10
def oil_quantity : ℕ := 1

-- Define the discount and coupon
def cheese_discount : ℚ := 0.25
def coupon_value : ℚ := 10
def coupon_threshold : ℚ := 30

-- Define Bob's spending function
def bob_spending : ℚ :=
  let bread_total := bread_price * bread_quantity
  let cheese_total := cheese_price * cheese_quantity * (1 - cheese_discount)
  let chocolate_total := chocolate_price * chocolate_quantity
  let oil_total := oil_price * oil_quantity
  let subtotal := bread_total + cheese_total + chocolate_total + oil_total
  if subtotal ≥ coupon_threshold then subtotal - coupon_value else subtotal

-- Theorem to prove
theorem bob_spending_is_26 : bob_spending = 26 := by sorry

end NUMINAMATH_CALUDE_bob_spending_is_26_l3146_314670


namespace NUMINAMATH_CALUDE_root_sum_fraction_equality_l3146_314611

theorem root_sum_fraction_equality (r s t : ℝ) : 
  r^3 - 6*r^2 + 11*r - 6 = 0 → 
  s^3 - 6*s^2 + 11*s - 6 = 0 → 
  t^3 - 6*t^2 + 11*t - 6 = 0 → 
  (r+s)/t + (s+t)/r + (t+r)/s = 25/3 :=
by
  sorry

end NUMINAMATH_CALUDE_root_sum_fraction_equality_l3146_314611


namespace NUMINAMATH_CALUDE_chord_length_l3146_314671

/-- The length of the chord formed by the intersection of a line and a circle -/
theorem chord_length (t : ℝ) : 
  let x := 1 + 2*t
  let y := 2 + t
  let circle := {(x, y) : ℝ × ℝ | x^2 + y^2 = 9}
  let line := {(x, y) : ℝ × ℝ | ∃ t, x = 1 + 2*t ∧ y = 2 + t}
  let intersection := circle ∩ line
  ∃ p q : ℝ × ℝ, p ∈ intersection ∧ q ∈ intersection ∧ 
    dist p q = 12/5 * Real.sqrt 5 :=
by
  sorry

end NUMINAMATH_CALUDE_chord_length_l3146_314671


namespace NUMINAMATH_CALUDE_number_solution_l3146_314621

theorem number_solution : ∃ (x : ℝ), 50 + (x * 12) / (180 / 3) = 51 ∧ x = 5 := by
  sorry

end NUMINAMATH_CALUDE_number_solution_l3146_314621


namespace NUMINAMATH_CALUDE_biancas_birthday_money_l3146_314650

theorem biancas_birthday_money (amount_per_friend : ℕ) (total_amount : ℕ) : 
  amount_per_friend = 6 → total_amount = 30 → total_amount / amount_per_friend = 5 := by
  sorry

end NUMINAMATH_CALUDE_biancas_birthday_money_l3146_314650


namespace NUMINAMATH_CALUDE_employed_females_percentage_l3146_314659

theorem employed_females_percentage (total_population : ℝ) 
  (employed_percentage : ℝ) (employed_males_percentage : ℝ) :
  employed_percentage = 96 →
  employed_males_percentage = 24 →
  (employed_percentage - employed_males_percentage) / employed_percentage * 100 = 75 := by
  sorry

end NUMINAMATH_CALUDE_employed_females_percentage_l3146_314659


namespace NUMINAMATH_CALUDE_equation_equiv_lines_l3146_314625

/-- The set of points satisfying the equation 2x^2 + y^2 + 3xy + 3x + y = 2 -/
def S : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | 2 * p.1^2 + p.2^2 + 3 * p.1 * p.2 + 3 * p.1 + p.2 = 2}

/-- The set of points on the line y = -x - 2 -/
def L1 : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2 = -p.1 - 2}

/-- The set of points on the line y = -2x + 1 -/
def L2 : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2 = -2 * p.1 + 1}

/-- Theorem stating that S is equivalent to the union of L1 and L2 -/
theorem equation_equiv_lines : S = L1 ∪ L2 := by
  sorry

end NUMINAMATH_CALUDE_equation_equiv_lines_l3146_314625


namespace NUMINAMATH_CALUDE_cube_with_cylindrical_hole_l3146_314679

/-- The surface area of a cube with a cylindrical hole --/
def surface_area (cube_edge : ℝ) (cylinder_radius : ℝ) (cylinder_height : ℝ) (π : ℝ) : ℝ :=
  6 * cube_edge^2 - 2 * π * cylinder_radius^2 + 2 * π * cylinder_radius * cylinder_height

/-- The volume of a cube with a cylindrical hole --/
def volume (cube_edge : ℝ) (cylinder_radius : ℝ) (cylinder_height : ℝ) (π : ℝ) : ℝ :=
  cube_edge^3 - π * cylinder_radius^2 * cylinder_height

/-- Theorem stating the surface area and volume of the resulting geometric figure --/
theorem cube_with_cylindrical_hole :
  let cube_edge : ℝ := 10
  let cylinder_radius : ℝ := 2
  let cylinder_height : ℝ := 10
  let π : ℝ := 3
  surface_area cube_edge cylinder_radius cylinder_height π = 696 ∧
  volume cube_edge cylinder_radius cylinder_height π = 880 := by
  sorry

end NUMINAMATH_CALUDE_cube_with_cylindrical_hole_l3146_314679


namespace NUMINAMATH_CALUDE_two_digit_number_digit_difference_l3146_314698

/-- Given a two-digit number where the difference between the original number
    and the number with interchanged digits is 45, prove that the difference
    between its two digits is 5. -/
theorem two_digit_number_digit_difference (x y : ℕ) : 
  x < 10 → y < 10 → (10 * x + y) - (10 * y + x) = 45 → x - y = 5 := by
  sorry

end NUMINAMATH_CALUDE_two_digit_number_digit_difference_l3146_314698


namespace NUMINAMATH_CALUDE_legoland_animals_l3146_314668

theorem legoland_animals (num_kangaroos : ℕ) (num_koalas : ℕ) : 
  num_kangaroos = 384 → 
  num_kangaroos = 8 * num_koalas → 
  num_kangaroos + num_koalas = 432 := by
sorry

end NUMINAMATH_CALUDE_legoland_animals_l3146_314668


namespace NUMINAMATH_CALUDE_percentage_problem_l3146_314600

theorem percentage_problem :
  ∃ x : ℝ, (18 : ℝ) / x = (45 : ℝ) / 100 ∧ x = 40 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l3146_314600


namespace NUMINAMATH_CALUDE_sin_2theta_value_l3146_314620

theorem sin_2theta_value (θ : ℝ) (h : Real.sin θ + Real.cos θ = 1/3) : 
  Real.sin (2 * θ) = -8/9 := by
  sorry

end NUMINAMATH_CALUDE_sin_2theta_value_l3146_314620


namespace NUMINAMATH_CALUDE_households_with_bike_only_l3146_314601

theorem households_with_bike_only 
  (total : ℕ) 
  (neither : ℕ) 
  (both : ℕ) 
  (with_car : ℕ) 
  (h1 : total = 90) 
  (h2 : neither = 11) 
  (h3 : both = 20) 
  (h4 : with_car = 44) : 
  total - neither - (with_car - both) - both = 35 := by
sorry

end NUMINAMATH_CALUDE_households_with_bike_only_l3146_314601


namespace NUMINAMATH_CALUDE_road_trip_distance_l3146_314643

/-- Road trip problem -/
theorem road_trip_distance (total_distance michelle_distance : ℕ) 
  (h1 : total_distance = 1000)
  (h2 : michelle_distance = 294)
  (h3 : ∃ (tracy_distance : ℕ), tracy_distance > 2 * michelle_distance)
  (h4 : ∃ (katie_distance : ℕ), michelle_distance = 3 * katie_distance) :
  ∃ (tracy_distance : ℕ), tracy_distance = total_distance - michelle_distance - (michelle_distance / 3) ∧ 
    tracy_distance - 2 * michelle_distance = 20 := by
  sorry

end NUMINAMATH_CALUDE_road_trip_distance_l3146_314643


namespace NUMINAMATH_CALUDE_number_equality_l3146_314603

theorem number_equality : ∃ x : ℝ, (0.4 * x = 0.3 * 50) ∧ (x = 37.5) := by
  sorry

end NUMINAMATH_CALUDE_number_equality_l3146_314603


namespace NUMINAMATH_CALUDE_gardening_project_cost_l3146_314693

-- Define constants for the given conditions
def rose_bushes : Nat := 20
def fruit_trees : Nat := 10
def ornamental_shrubs : Nat := 5
def rose_bush_cost : Nat := 150
def fertilizer_cost : Nat := 25
def fruit_tree_cost : Nat := 75
def ornamental_shrub_cost : Nat := 50
def gardener_hourly_rate : Nat := 30
def soil_cost_per_cubic_foot : Nat := 5
def soil_needed : Nat := 100
def tiller_cost_per_day : Nat := 40
def wheelbarrow_cost_per_day : Nat := 10
def rental_days : Nat := 3

def gardener_hours : List Nat := [6, 5, 4, 7]

-- Define functions for calculations
def rose_bush_total_cost : Nat :=
  let base_cost := rose_bushes * rose_bush_cost
  let discount := base_cost * 5 / 100
  base_cost - discount

def fertilizer_total_cost : Nat :=
  let base_cost := rose_bushes * fertilizer_cost
  let discount := base_cost * 10 / 100
  base_cost - discount

def fruit_tree_total_cost : Nat :=
  let free_trees := fruit_trees / 3
  let paid_trees := fruit_trees - free_trees
  paid_trees * fruit_tree_cost

def ornamental_shrub_total_cost : Nat :=
  ornamental_shrubs * ornamental_shrub_cost

def gardener_total_cost : Nat :=
  (gardener_hours.sum) * gardener_hourly_rate

def soil_total_cost : Nat :=
  soil_needed * soil_cost_per_cubic_foot

def tools_rental_total_cost : Nat :=
  (tiller_cost_per_day + wheelbarrow_cost_per_day) * rental_days

-- Define the total cost of the gardening project
def total_gardening_cost : Nat :=
  rose_bush_total_cost +
  fertilizer_total_cost +
  fruit_tree_total_cost +
  ornamental_shrub_total_cost +
  gardener_total_cost +
  soil_total_cost +
  tools_rental_total_cost

-- Theorem statement
theorem gardening_project_cost :
  total_gardening_cost = 6385 := by sorry

end NUMINAMATH_CALUDE_gardening_project_cost_l3146_314693


namespace NUMINAMATH_CALUDE_square_difference_theorem_l3146_314623

theorem square_difference_theorem : ∃ (n m : ℕ),
  (∀ k : ℕ, k^2 < 2018 → k ≤ n) ∧
  (n^2 < 2018) ∧
  (∀ k : ℕ, 2018 < k^2 → m ≤ k) ∧
  (2018 < m^2) ∧
  (m^2 - n^2 = 89) := by
sorry

end NUMINAMATH_CALUDE_square_difference_theorem_l3146_314623


namespace NUMINAMATH_CALUDE_birds_in_tree_l3146_314635

theorem birds_in_tree (initial_birds : ℕ) (new_birds : ℕ) (total_birds : ℕ) :
  initial_birds = 14 →
  new_birds = 21 →
  total_birds = initial_birds + new_birds →
  total_birds = 35 := by
sorry

end NUMINAMATH_CALUDE_birds_in_tree_l3146_314635


namespace NUMINAMATH_CALUDE_max_cars_ac_no_stripes_l3146_314691

theorem max_cars_ac_no_stripes 
  (total_cars : ℕ) 
  (cars_without_ac : ℕ) 
  (cars_with_stripes : ℕ) 
  (h1 : total_cars = 200)
  (h2 : cars_without_ac = 85)
  (h3 : cars_with_stripes ≥ 110) :
  ∃ (max_ac_no_stripes : ℕ),
    max_ac_no_stripes = 5 ∧
    max_ac_no_stripes ≤ total_cars - cars_without_ac ∧
    max_ac_no_stripes ≤ total_cars - cars_with_stripes :=
by
  sorry

end NUMINAMATH_CALUDE_max_cars_ac_no_stripes_l3146_314691


namespace NUMINAMATH_CALUDE_trig_identity_l3146_314626

theorem trig_identity : 
  (2 * Real.cos (10 * π / 180) - Real.sin (20 * π / 180)) / Real.sin (70 * π / 180) = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_l3146_314626


namespace NUMINAMATH_CALUDE_exists_congruent_polygons_l3146_314666

/-- A regular n-gon with colored vertices -/
structure ColoredRegularNGon (n : ℕ) (p : ℕ) where
  (n_ge_6 : n ≥ 6)
  (p_bounds : 3 ≤ p ∧ p < n - p)

/-- The set of red vertices -/
def R (n : ℕ) (p : ℕ) : Set (Fin n) :=
  {i | i.val < p}

/-- The set of black vertices -/
def B (n : ℕ) (p : ℕ) : Set (Fin n) :=
  {i | i.val ≥ p}

/-- Rotation of a vertex by i positions -/
def rotate (n : ℕ) (i : Fin n) (v : Fin n) : Fin n :=
  ⟨(v.val + i.val) % n, by sorry⟩

/-- The theorem to be proved -/
theorem exists_congruent_polygons (n p : ℕ) (h : ColoredRegularNGon n p) :
  ∃ (i : Fin n), (Set.image (rotate n i) (R n p) ∩ B n p).ncard > p / 2 := by
  sorry

end NUMINAMATH_CALUDE_exists_congruent_polygons_l3146_314666


namespace NUMINAMATH_CALUDE_simplify_expression_l3146_314628

theorem simplify_expression (x : ℝ) : 4 * (x^2 - 5*x) - 5 * (2*x^2 + 3*x) = -6*x^2 - 35*x := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3146_314628


namespace NUMINAMATH_CALUDE_diana_bottle_caps_l3146_314602

/-- The number of bottle caps Diana starts with -/
def initial_caps : ℕ := 65

/-- The number of bottle caps eaten by the hippopotamus -/
def eaten_caps : ℕ := 4

/-- The number of bottle caps Diana ends with -/
def final_caps : ℕ := initial_caps - eaten_caps

theorem diana_bottle_caps : final_caps = 61 := by
  sorry

end NUMINAMATH_CALUDE_diana_bottle_caps_l3146_314602


namespace NUMINAMATH_CALUDE_antons_number_l3146_314634

def matches_one_digit (a b : ℕ) : Prop :=
  (a / 100 = b / 100 ∧ a % 100 ≠ b % 100) ∨
  (a % 100 / 10 = b % 100 / 10 ∧ a / 100 ≠ b / 100 ∧ a % 10 ≠ b % 10) ∨
  (a % 10 = b % 10 ∧ a / 10 ≠ b / 10)

theorem antons_number (x : ℕ) :
  100 ≤ x ∧ x < 1000 ∧
  matches_one_digit x 109 ∧
  matches_one_digit x 704 ∧
  matches_one_digit x 124 →
  x = 729 := by
  sorry

end NUMINAMATH_CALUDE_antons_number_l3146_314634


namespace NUMINAMATH_CALUDE_coffee_mix_proof_l3146_314695

/-- The price of Colombian coffee beans in dollars per pound -/
def colombian_price : ℝ := 5.50

/-- The price of Peruvian coffee beans in dollars per pound -/
def peruvian_price : ℝ := 4.25

/-- The total weight of the mix in pounds -/
def total_weight : ℝ := 40

/-- The desired price of the mix in dollars per pound -/
def mix_price : ℝ := 4.60

/-- The amount of Colombian coffee beans in the mix -/
def colombian_amount : ℝ := 11.2

theorem coffee_mix_proof :
  colombian_amount * colombian_price + (total_weight - colombian_amount) * peruvian_price = 
  mix_price * total_weight :=
sorry

end NUMINAMATH_CALUDE_coffee_mix_proof_l3146_314695


namespace NUMINAMATH_CALUDE_sum_of_two_squares_condition_l3146_314664

theorem sum_of_two_squares_condition (p : ℕ) (hp : Nat.Prime p) :
  (∃ a b : ℤ, p = a^2 + b^2) ↔ p % 4 = 1 ∨ p = 2 :=
sorry

end NUMINAMATH_CALUDE_sum_of_two_squares_condition_l3146_314664


namespace NUMINAMATH_CALUDE_distance_is_8_sqrt_2_l3146_314682

/-- Two externally tangent circles with a common external tangent -/
structure TangentCircles where
  /-- Radius of the larger circle -/
  r₁ : ℝ
  /-- Radius of the smaller circle -/
  r₂ : ℝ
  /-- The circles are externally tangent -/
  tangent : r₁ > r₂
  /-- The radii are 8 and 2 units respectively -/
  radius_values : r₁ = 8 ∧ r₂ = 2

/-- The distance from the center of the larger circle to the point where 
    the common external tangent touches the smaller circle -/
def distance_to_tangent_point (c : TangentCircles) : ℝ :=
  sorry

/-- Theorem stating that the distance is 8√2 -/
theorem distance_is_8_sqrt_2 (c : TangentCircles) : 
  distance_to_tangent_point c = 8 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_distance_is_8_sqrt_2_l3146_314682


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l3146_314607

theorem quadratic_equation_roots (x : ℝ) : 
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ (2 * x₁^2 - 3 * x₁ - (3/2) = 0) ∧ (2 * x₂^2 - 3 * x₂ - (3/2) = 0) :=
by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l3146_314607


namespace NUMINAMATH_CALUDE_eleven_million_nine_hundred_thousand_scientific_notation_l3146_314610

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  valid : 1 ≤ coefficient ∧ coefficient < 10

/-- Converts a real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem eleven_million_nine_hundred_thousand_scientific_notation :
  toScientificNotation 11090000 = ScientificNotation.mk 1.109 7 (by norm_num) :=
sorry

end NUMINAMATH_CALUDE_eleven_million_nine_hundred_thousand_scientific_notation_l3146_314610


namespace NUMINAMATH_CALUDE_fifth_day_income_correct_l3146_314655

/-- Calculates the cab driver's income on the fifth day given the income for the first four days and the average income for five days. -/
def fifth_day_income (day1 day2 day3 day4 avg : ℚ) : ℚ :=
  5 * avg - (day1 + day2 + day3 + day4)

/-- Proves that the calculated fifth day income is correct given the income for the first four days and the average income for five days. -/
theorem fifth_day_income_correct (day1 day2 day3 day4 avg : ℚ) :
  let fifth_day := fifth_day_income day1 day2 day3 day4 avg
  (day1 + day2 + day3 + day4 + fifth_day) / 5 = avg :=
by sorry

#eval fifth_day_income 400 250 650 400 440

end NUMINAMATH_CALUDE_fifth_day_income_correct_l3146_314655


namespace NUMINAMATH_CALUDE_two_distinct_roots_iff_p_condition_l3146_314660

theorem two_distinct_roots_iff_p_condition (p : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ 
    ((x ≥ 0 → x^2 - 2*x - p = 0) ∧ (x < 0 → x^2 + 2*x - p = 0)) ∧
    ((y ≥ 0 → y^2 - 2*y - p = 0) ∧ (y < 0 → y^2 + 2*y - p = 0)))
  ↔ 
  (p > 0 ∨ p = -1) :=
sorry

end NUMINAMATH_CALUDE_two_distinct_roots_iff_p_condition_l3146_314660


namespace NUMINAMATH_CALUDE_greatest_prime_factor_of_product_l3146_314614

def x : ℕ := 2 * 4 * 6 * 8 * 10 * 12 * 14 * 16 * 18 * 20

theorem greatest_prime_factor_of_product (x : ℕ) : 
  x = 2 * 4 * 6 * 8 * 10 * 12 * 14 * 16 * 18 * 20 →
  ∃ p : ℕ, Nat.Prime p ∧ p ∣ (18 * x * 14 * x) ∧ 
  ∀ q : ℕ, Nat.Prime q → q ∣ (18 * x * 14 * x) → q ≤ p ∧ p = 7 :=
by sorry

end NUMINAMATH_CALUDE_greatest_prime_factor_of_product_l3146_314614


namespace NUMINAMATH_CALUDE_fraction_puzzle_l3146_314677

theorem fraction_puzzle : ∃ (x y : ℕ), 
  x + 35 = y ∧ 
  x ≠ 0 ∧ 
  y ≠ 0 ∧
  (x : ℚ) / y + (x.gcd y : ℚ) * x / ((y.gcd x) * y) = 16 / 13 ∧
  x = 56 ∧
  y = 91 := by
sorry

end NUMINAMATH_CALUDE_fraction_puzzle_l3146_314677


namespace NUMINAMATH_CALUDE_inequality_proof_l3146_314665

theorem inequality_proof (a b θ : Real) 
  (h1 : a > b) (h2 : b > 1) (h3 : 0 < θ) (h4 : θ < π / 2) :
  a * Real.log (Real.sin θ) / Real.log b < b * Real.log (Real.sin θ) / Real.log a :=
sorry

end NUMINAMATH_CALUDE_inequality_proof_l3146_314665


namespace NUMINAMATH_CALUDE_compound_weight_is_88_l3146_314667

/-- The molecular weight of the carbon part in the compound C4H8O2 -/
def carbon_weight : ℕ := 48

/-- The molecular weight of the hydrogen part in the compound C4H8O2 -/
def hydrogen_weight : ℕ := 8

/-- The molecular weight of the oxygen part in the compound C4H8O2 -/
def oxygen_weight : ℕ := 32

/-- The total molecular weight of the compound C4H8O2 -/
def total_molecular_weight : ℕ := carbon_weight + hydrogen_weight + oxygen_weight

theorem compound_weight_is_88 : total_molecular_weight = 88 := by
  sorry

end NUMINAMATH_CALUDE_compound_weight_is_88_l3146_314667


namespace NUMINAMATH_CALUDE_lily_shopping_ratio_l3146_314658

theorem lily_shopping_ratio (initial_balance shirt_cost final_balance : ℕ) 
  (h1 : initial_balance = 55)
  (h2 : shirt_cost = 7)
  (h3 : final_balance = 27) :
  (initial_balance - shirt_cost - final_balance) / shirt_cost = 3 := by
sorry

end NUMINAMATH_CALUDE_lily_shopping_ratio_l3146_314658


namespace NUMINAMATH_CALUDE_quilt_cost_is_450_l3146_314683

/-- Calculates the total cost of patches for a quilt with given dimensions and pricing rules. -/
def quilt_patch_cost (quilt_length : ℕ) (quilt_width : ℕ) (patch_area : ℕ) 
                     (first_patch_cost : ℕ) (first_patch_count : ℕ) : ℕ :=
  let total_area := quilt_length * quilt_width
  let total_patches := total_area / patch_area
  let remaining_patches := total_patches - first_patch_count
  let first_batch_cost := first_patch_count * first_patch_cost
  let remaining_cost := remaining_patches * (first_patch_cost / 2)
  first_batch_cost + remaining_cost

/-- Proves that the total cost of patches for the specified quilt is $450. -/
theorem quilt_cost_is_450 : quilt_patch_cost 16 20 4 10 10 = 450 := by
  sorry

end NUMINAMATH_CALUDE_quilt_cost_is_450_l3146_314683


namespace NUMINAMATH_CALUDE_fabric_width_l3146_314648

/-- Given a rectangular piece of fabric with area 24 square centimeters and length 8 centimeters,
    prove that its width is 3 centimeters. -/
theorem fabric_width (area : ℝ) (length : ℝ) (width : ℝ) 
    (h1 : area = 24) 
    (h2 : length = 8) 
    (h3 : area = length * width) : width = 3 := by
  sorry

end NUMINAMATH_CALUDE_fabric_width_l3146_314648


namespace NUMINAMATH_CALUDE_fraction_value_l3146_314631

theorem fraction_value (m n : ℝ) (h : |m - 1/4| + (n + 3)^2 = 0) : n / m = -12 := by
  sorry

end NUMINAMATH_CALUDE_fraction_value_l3146_314631


namespace NUMINAMATH_CALUDE_fraction_sum_l3146_314624

theorem fraction_sum : (2 : ℚ) / 5 + (3 : ℚ) / 8 = (31 : ℚ) / 40 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_l3146_314624


namespace NUMINAMATH_CALUDE_fish_count_l3146_314615

theorem fish_count (total_tables : ℕ) (special_table_fish : ℕ) (regular_table_fish : ℕ)
  (h1 : total_tables = 32)
  (h2 : special_table_fish = 3)
  (h3 : regular_table_fish = 2) :
  (total_tables - 1) * regular_table_fish + special_table_fish = 65 := by
  sorry

end NUMINAMATH_CALUDE_fish_count_l3146_314615


namespace NUMINAMATH_CALUDE_debby_photos_l3146_314646

/-- Calculates the number of photographs Debby kept after her vacation -/
theorem debby_photos (N : ℝ) : 
  let zoo_percent : ℝ := 0.60
  let museum_percent : ℝ := 0.25
  let gallery_percent : ℝ := 0.15
  let zoo_keep : ℝ := 0.70
  let museum_keep : ℝ := 0.50
  let gallery_keep : ℝ := 1

  let zoo_photos : ℝ := zoo_percent * N
  let museum_photos : ℝ := museum_percent * N
  let gallery_photos : ℝ := gallery_percent * N

  let kept_zoo : ℝ := zoo_keep * zoo_photos
  let kept_museum : ℝ := museum_keep * museum_photos
  let kept_gallery : ℝ := gallery_keep * gallery_photos

  let total_kept : ℝ := kept_zoo + kept_museum + kept_gallery

  total_kept = 0.695 * N :=
by sorry

end NUMINAMATH_CALUDE_debby_photos_l3146_314646


namespace NUMINAMATH_CALUDE_outfit_combinations_l3146_314688

theorem outfit_combinations (shirts : ℕ) (hats : ℕ) : shirts = 5 → hats = 3 → shirts * hats = 15 := by
  sorry

end NUMINAMATH_CALUDE_outfit_combinations_l3146_314688


namespace NUMINAMATH_CALUDE_product_scaling_l3146_314640

theorem product_scaling (a b c : ℝ) (h : (268 : ℝ) * 74 = 19832) :
  2.68 * 0.74 = 1.9832 := by
  sorry

end NUMINAMATH_CALUDE_product_scaling_l3146_314640


namespace NUMINAMATH_CALUDE_existence_of_special_set_l3146_314694

theorem existence_of_special_set (n : ℕ) (h : n ≥ 2) :
  ∃ (S : Finset ℤ), Finset.card S = n ∧
    ∀ (a b : ℤ), a ∈ S → b ∈ S → a ≠ b → (a - b)^2 ∣ (a * b) := by
  sorry

end NUMINAMATH_CALUDE_existence_of_special_set_l3146_314694


namespace NUMINAMATH_CALUDE_h_inverse_at_one_l3146_314672

def h (x : ℝ) : ℝ := 5 * x - 6

theorem h_inverse_at_one :
  ∃ b : ℝ, h b = 1 ∧ b = 7/5 := by
  sorry

end NUMINAMATH_CALUDE_h_inverse_at_one_l3146_314672


namespace NUMINAMATH_CALUDE_probability_of_white_ball_l3146_314609

/-- The probability of drawing a white ball from a bag containing white and red balls -/
theorem probability_of_white_ball (num_white : ℕ) (num_red : ℕ) : 
  num_white = 6 → num_red = 14 → (num_white : ℚ) / (num_white + num_red : ℚ) = 3 / 10 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_white_ball_l3146_314609


namespace NUMINAMATH_CALUDE_dave_initial_money_l3146_314633

-- Define the given amounts
def derek_initial : ℕ := 40
def derek_spend1 : ℕ := 14
def derek_spend2 : ℕ := 11
def derek_spend3 : ℕ := 5
def dave_spend : ℕ := 7
def dave_extra : ℕ := 33

-- Define Derek's total spending
def derek_total_spend : ℕ := derek_spend1 + derek_spend2 + derek_spend3

-- Define Derek's remaining money
def derek_remaining : ℕ := derek_initial - derek_total_spend

-- Define Dave's remaining money
def dave_remaining : ℕ := derek_remaining + dave_extra

-- Theorem to prove
theorem dave_initial_money : dave_remaining + dave_spend = 50 := by
  sorry

end NUMINAMATH_CALUDE_dave_initial_money_l3146_314633


namespace NUMINAMATH_CALUDE_complex_power_simplification_l3146_314662

theorem complex_power_simplification :
  (3 * (Complex.cos (30 * π / 180)) + 3 * Complex.I * (Complex.sin (30 * π / 180)))^4 =
  Complex.mk (-81/2) ((81 * Real.sqrt 3)/2) := by
sorry

end NUMINAMATH_CALUDE_complex_power_simplification_l3146_314662


namespace NUMINAMATH_CALUDE_max_profit_l3146_314651

def factory_price_A : ℕ := 10
def factory_price_B : ℕ := 18
def selling_price_A : ℕ := 12
def selling_price_B : ℕ := 22
def total_vehicles : ℕ := 130

def profit_function (x : ℕ) : ℤ :=
  -2 * x + 520

def is_valid_purchase (x : ℕ) : Prop :=
  x ≤ total_vehicles ∧ (total_vehicles - x) ≤ 2 * x

theorem max_profit :
  ∃ (x : ℕ), is_valid_purchase x ∧
    ∀ (y : ℕ), is_valid_purchase y → profit_function x ≥ profit_function y ∧
    profit_function x = 432 :=
  sorry

end NUMINAMATH_CALUDE_max_profit_l3146_314651


namespace NUMINAMATH_CALUDE_marble_problem_l3146_314649

/-- The number of white marbles in the bag -/
def white_marbles : ℕ := 3

/-- The probability that all 3 girls select the same colored marble -/
def same_color_prob : ℚ := 1/10

/-- The number of black marbles in the bag -/
def black_marbles : ℕ := 3

/-- The total number of marbles in the bag -/
def total_marbles : ℕ := white_marbles + black_marbles

/-- The probability of all girls selecting white marbles -/
def all_white_prob : ℚ := (white_marbles / total_marbles) * 
                          ((white_marbles - 1) / (total_marbles - 1)) * 
                          ((white_marbles - 2) / (total_marbles - 2))

/-- The probability of all girls selecting black marbles -/
def all_black_prob : ℚ := (black_marbles / total_marbles) * 
                          ((black_marbles - 1) / (total_marbles - 1)) * 
                          ((black_marbles - 2) / (total_marbles - 2))

theorem marble_problem : 
  all_white_prob + all_black_prob = same_color_prob :=
by sorry

end NUMINAMATH_CALUDE_marble_problem_l3146_314649


namespace NUMINAMATH_CALUDE_village_b_largest_population_l3146_314656

/-- Calculate the population after n years given initial population and growth rate -/
def futurePopulation (initialPop : ℝ) (growthRate : ℝ) (years : ℕ) : ℝ :=
  initialPop * (1 + growthRate) ^ years

/-- Theorem: Village B has the largest population after 3 years -/
theorem village_b_largest_population :
  let villageA := futurePopulation 12000 0.24 3
  let villageB := futurePopulation 15000 0.18 3
  let villageC := futurePopulation 18000 (-0.12) 3
  villageB > villageA ∧ villageB > villageC := by sorry

end NUMINAMATH_CALUDE_village_b_largest_population_l3146_314656


namespace NUMINAMATH_CALUDE_circle_and_max_distance_l3146_314685

-- Define the circle C
def Circle (a b r : ℝ) := {(x, y) : ℝ × ℝ | (x - a)^2 + (y - b)^2 = r^2}

-- Define the conditions for the circle
def CircleConditions (a b r : ℝ) : Prop :=
  3 * a - b = 0 ∧ 
  a ≥ 0 ∧ 
  |a - 4| = r ∧ 
  ((3 * a + 4 * b + 10)^2 / 25 + 12 = r^2)

-- Define points A and B
def A : ℝ × ℝ := (1, 1)
def B : ℝ × ℝ := (-2, 0)

-- Define the distance squared function
def DistanceSquared (p q : ℝ × ℝ) : ℝ :=
  (p.1 - q.1)^2 + (p.2 - q.2)^2

-- Theorem statement
theorem circle_and_max_distance :
  ∃ a b r : ℝ, 
    CircleConditions a b r → 
    (Circle a b r = Circle 0 0 4) ∧
    (∀ p ∈ Circle 0 0 4, DistanceSquared p A + DistanceSquared p B ≤ 38 + 8 * Real.sqrt 2) ∧
    (∃ p ∈ Circle 0 0 4, DistanceSquared p A + DistanceSquared p B = 38 + 8 * Real.sqrt 2) :=
  sorry

end NUMINAMATH_CALUDE_circle_and_max_distance_l3146_314685


namespace NUMINAMATH_CALUDE_function_solution_set_l3146_314605

theorem function_solution_set (a : ℝ) : 
  (∀ x : ℝ, (|2*x - a| + a ≤ 6 ↔ -2 ≤ x ∧ x ≤ 3)) → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_function_solution_set_l3146_314605


namespace NUMINAMATH_CALUDE_paths_count_l3146_314699

/-- The number of paths between two points given the number of right and down steps -/
def numPaths (right down : ℕ) : ℕ := sorry

/-- The total number of paths from A to D via B and C -/
def totalPaths : ℕ :=
  let pathsAB := numPaths 2 2  -- B is 2 right and 2 down from A
  let pathsBC := numPaths 1 3  -- C is 1 right and 3 down from B
  let pathsCD := numPaths 3 1  -- D is 3 right and 1 down from C
  pathsAB * pathsBC * pathsCD

theorem paths_count : totalPaths = 96 := by sorry

end NUMINAMATH_CALUDE_paths_count_l3146_314699


namespace NUMINAMATH_CALUDE_second_term_of_geometric_series_l3146_314657

theorem second_term_of_geometric_series 
  (r : ℝ) (S : ℝ) (a : ℝ) (h1 : r = (1 : ℝ) / 4)
  (h2 : S = 48) (h3 : S = a / (1 - r)) :
  a * r = 9 := by
sorry

end NUMINAMATH_CALUDE_second_term_of_geometric_series_l3146_314657


namespace NUMINAMATH_CALUDE_quadratic_roots_sequence_l3146_314673

theorem quadratic_roots_sequence (p q a b : ℝ) : 
  p > 0 → 
  q > 0 → 
  a ≠ b → 
  a^2 - p*a + q = 0 → 
  b^2 - p*b + q = 0 → 
  ((a + b = 2*(-2) ∨ a + (-2) = 2*b ∨ b + (-2) = 2*a) ∧ 
   (a * b = (-2)^2 ∨ a * (-2) = b^2 ∨ b * (-2) = a^2)) → 
  p + q = 9 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_sequence_l3146_314673


namespace NUMINAMATH_CALUDE_triangle_properties_l3146_314644

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The given condition -/
def condition (t : Triangle) : Prop :=
  (2 * t.a + t.b) * Real.cos t.C + t.c * Real.cos t.B = 0

theorem triangle_properties (t : Triangle) 
  (h : condition t) : 
  t.C = 2 * Real.pi / 3 ∧ 
  (t.c = 6 → ∃ (max_area : ℝ), max_area = 3 * Real.sqrt 3 ∧ 
    ∀ (area : ℝ), area ≤ max_area) :=
sorry

end NUMINAMATH_CALUDE_triangle_properties_l3146_314644


namespace NUMINAMATH_CALUDE_fill_time_three_pipes_l3146_314618

/-- Represents a pipe that can fill or empty a tank -/
structure Pipe where
  rate : ℚ  -- Rate at which the pipe fills (positive) or empties (negative) the tank per hour

/-- Represents a system of pipes filling a tank -/
def PipeSystem (pipes : List Pipe) : ℚ :=
  pipes.map (·.rate) |> List.sum

theorem fill_time_three_pipes (a b c : Pipe) 
  (ha : a.rate = 1/3)
  (hb : b.rate = 1/4)
  (hc : c.rate = -1/4) :
  (PipeSystem [a, b, c])⁻¹ = 3 := by
  sorry

#check fill_time_three_pipes

end NUMINAMATH_CALUDE_fill_time_three_pipes_l3146_314618


namespace NUMINAMATH_CALUDE_walnut_trees_after_planting_l3146_314627

/-- The number of walnut trees in the park after planting -/
def total_trees (initial_trees planted_trees : ℕ) : ℕ :=
  initial_trees + planted_trees

/-- Theorem: The total number of walnut trees after planting is 55 -/
theorem walnut_trees_after_planting :
  total_trees 22 33 = 55 := by
  sorry

end NUMINAMATH_CALUDE_walnut_trees_after_planting_l3146_314627


namespace NUMINAMATH_CALUDE_mendel_pea_experiment_l3146_314684

/-- Represents the genotype of a pea plant -/
inductive Genotype
| DD
| Dd
| dd

/-- Represents a generation of pea plants -/
structure Generation where
  DD_ratio : ℚ
  Dd_ratio : ℚ
  dd_ratio : ℚ
  sum_to_one : DD_ratio + Dd_ratio + dd_ratio = 1

/-- First generation with all Dd genotype -/
def first_gen : Generation where
  DD_ratio := 0
  Dd_ratio := 1
  dd_ratio := 0
  sum_to_one := by norm_num

/-- Function to calculate the next generation's ratios -/
def next_gen (g : Generation) : Generation where
  DD_ratio := g.DD_ratio^2 + g.DD_ratio * g.Dd_ratio + (g.Dd_ratio^2) / 4
  Dd_ratio := g.DD_ratio * g.Dd_ratio + g.Dd_ratio * g.dd_ratio + (g.Dd_ratio^2) / 2
  dd_ratio := g.dd_ratio^2 + g.dd_ratio * g.Dd_ratio + (g.Dd_ratio^2) / 4
  sum_to_one := by sorry

/-- Second generation -/
def second_gen : Generation := next_gen first_gen

/-- Third generation -/
def third_gen : Generation := next_gen second_gen

/-- Probability of dominant trait in a generation -/
def prob_dominant (g : Generation) : ℚ := g.DD_ratio + g.Dd_ratio

theorem mendel_pea_experiment :
  (third_gen.dd_ratio = 1/4) ∧
  (3 * (prob_dominant third_gen)^2 * (1 - prob_dominant third_gen) = 27/64) := by sorry

end NUMINAMATH_CALUDE_mendel_pea_experiment_l3146_314684


namespace NUMINAMATH_CALUDE_combined_flock_size_l3146_314674

def initial_flock : ℕ := 100
def net_increase_per_year : ℕ := 10
def years : ℕ := 5
def other_flock : ℕ := 150

theorem combined_flock_size :
  initial_flock + net_increase_per_year * years + other_flock = 300 := by
  sorry

end NUMINAMATH_CALUDE_combined_flock_size_l3146_314674


namespace NUMINAMATH_CALUDE_sqrt_450_simplification_l3146_314647

theorem sqrt_450_simplification : Real.sqrt 450 = 15 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_450_simplification_l3146_314647


namespace NUMINAMATH_CALUDE_percentage_problem_l3146_314619

theorem percentage_problem (P : ℝ) : 
  (P / 100) * 600 = (40 / 100) * 1050 → P = 70 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l3146_314619


namespace NUMINAMATH_CALUDE_hyperbola_n_range_l3146_314645

/-- Represents a hyperbola with parameters m and n -/
structure Hyperbola (m n : ℝ) where
  equation : ∀ x y : ℝ, x^2 / (m^2 + n) - y^2 / (3 * m^2 - n) = 1

/-- The distance between the foci of a hyperbola -/
def focal_distance (h : Hyperbola m n) : ℝ := 4

/-- Theorem stating the range of n for a hyperbola with given properties -/
theorem hyperbola_n_range (m n : ℝ) (h : Hyperbola m n) :
  focal_distance h = 4 → -1 < n ∧ n < 3 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_n_range_l3146_314645


namespace NUMINAMATH_CALUDE_limit_example_l3146_314637

open Real

/-- The limit of (9x^2 - 1) / (x + 1/3) as x approaches -1/3 is -6 -/
theorem limit_example : 
  ∀ ε > 0, ∃ δ > 0, ∀ x : ℝ, 
    0 < |x + 1/3| → |x + 1/3| < δ → 
    |(9*x^2 - 1) / (x + 1/3) + 6| < ε := by
sorry

end NUMINAMATH_CALUDE_limit_example_l3146_314637


namespace NUMINAMATH_CALUDE_elementary_classes_count_l3146_314629

/-- The number of schools -/
def num_schools : ℕ := 2

/-- The number of middle school classes per school -/
def middle_classes_per_school : ℕ := 5

/-- The number of soccer balls donated per class -/
def balls_per_class : ℕ := 5

/-- The total number of soccer balls donated -/
def total_balls : ℕ := 90

/-- The number of elementary school classes in each school -/
def elementary_classes_per_school : ℕ := 4

theorem elementary_classes_count : 
  num_schools * elementary_classes_per_school * balls_per_class + 
  num_schools * middle_classes_per_school * balls_per_class = total_balls :=
sorry

end NUMINAMATH_CALUDE_elementary_classes_count_l3146_314629


namespace NUMINAMATH_CALUDE_gcd_of_product_of_differences_l3146_314642

theorem gcd_of_product_of_differences (a b c d : ℤ) : 
  ∃ (k : ℤ), (12 : ℤ) ∣ (a - b) * (a - c) * (a - d) * (b - c) * (b - d) * (c - d) ∧
  ∀ (m : ℤ), (∀ (x y z w : ℤ), m ∣ (x - y) * (x - z) * (x - w) * (y - z) * (y - w) * (z - w)) → m ∣ 12 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_product_of_differences_l3146_314642


namespace NUMINAMATH_CALUDE_belle_weekly_treat_cost_l3146_314697

def cost_brand_a : ℚ := 0.25
def cost_brand_b : ℚ := 0.35
def cost_small_rawhide : ℚ := 1
def cost_large_rawhide : ℚ := 1.5

def odd_day_cost : ℚ :=
  3 * cost_brand_a + 2 * cost_brand_b + cost_small_rawhide + cost_large_rawhide

def even_day_cost : ℚ :=
  4 * cost_brand_a + 2 * cost_small_rawhide

def days_in_week : ℕ := 7
def odd_days_in_week : ℕ := 4
def even_days_in_week : ℕ := 3

theorem belle_weekly_treat_cost :
  odd_days_in_week * odd_day_cost + even_days_in_week * even_day_cost = 24.8 := by
  sorry

end NUMINAMATH_CALUDE_belle_weekly_treat_cost_l3146_314697


namespace NUMINAMATH_CALUDE_display_window_configurations_l3146_314681

theorem display_window_configurations :
  let fiction_books : ℕ := 3
  let nonfiction_books : ℕ := 3
  let fiction_arrangements := Nat.factorial fiction_books
  let nonfiction_arrangements := Nat.factorial nonfiction_books
  fiction_arrangements * nonfiction_arrangements = 36 :=
by sorry

end NUMINAMATH_CALUDE_display_window_configurations_l3146_314681


namespace NUMINAMATH_CALUDE_max_value_of_fraction_sum_l3146_314653

theorem max_value_of_fraction_sum (x y z : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) (hz : z ≥ 0) (hsum : x + y + z = 2) :
  (x * y) / (x + y) + (x * z) / (x + z) + (y * z) / (y + z) ≤ 1 ∧
  ∃ (a b c : ℝ), a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ a + b + c = 2 ∧
    (a * b) / (a + b) + (a * c) / (a + c) + (b * c) / (b + c) = 1 :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_fraction_sum_l3146_314653


namespace NUMINAMATH_CALUDE_airplane_seats_theorem_l3146_314630

/-- Represents the total number of seats on an airplane -/
def total_seats : ℕ := 300

/-- Represents the number of First Class seats -/
def first_class_seats : ℕ := 30

/-- Represents the percentage of Business Class seats -/
def business_class_percentage : ℚ := 20 / 100

/-- Represents the percentage of Economy Class seats -/
def economy_class_percentage : ℚ := 70 / 100

/-- Theorem stating that the total number of seats is 300 -/
theorem airplane_seats_theorem :
  (first_class_seats : ℚ) +
  (business_class_percentage * total_seats) +
  (economy_class_percentage * total_seats) =
  total_seats :=
sorry

end NUMINAMATH_CALUDE_airplane_seats_theorem_l3146_314630


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l3146_314652

-- Define the arithmetic sequence
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

-- State the theorem
theorem arithmetic_sequence_problem (a : ℕ → ℝ) 
  (h1 : arithmetic_sequence a) 
  (h2 : a 3 + a 13 = 20) 
  (h3 : a 2 = -2) : 
  a 15 = 24 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l3146_314652
