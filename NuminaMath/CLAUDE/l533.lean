import Mathlib

namespace NUMINAMATH_CALUDE_coin_flip_probability_difference_l533_53319

def fair_coin_probability (n k : ℕ) : ℚ :=
  (Nat.choose n k : ℚ) * (1 / 2) ^ k * (1 / 2) ^ (n - k)

theorem coin_flip_probability_difference :
  fair_coin_probability 4 3 - fair_coin_probability 4 4 = 7 / 16 := by
  sorry

end NUMINAMATH_CALUDE_coin_flip_probability_difference_l533_53319


namespace NUMINAMATH_CALUDE_divide_algebraic_expression_l533_53320

theorem divide_algebraic_expression (a b : ℝ) (h : a ≠ 0) :
  (8 * a * b) / (2 * a) = 4 * b := by
  sorry

end NUMINAMATH_CALUDE_divide_algebraic_expression_l533_53320


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l533_53321

theorem sqrt_equation_solution (x : ℝ) : 
  Real.sqrt (3 + Real.sqrt (x^2)) = 4 ↔ x = 13 ∨ x = -13 :=
by sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l533_53321


namespace NUMINAMATH_CALUDE_abc_fraction_simplification_l533_53358

theorem abc_fraction_simplification 
  (a b c : ℝ) 
  (distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c) 
  (sum_condition : a + b + c = 1) :
  let s := a * b + b * c + c * a
  (a^2 + b^2 + c^2) ≠ 0 ∧ 
  (ab+bc+ca) / (a^2+b^2+c^2) = s / (1 - 2*s) := by
sorry

end NUMINAMATH_CALUDE_abc_fraction_simplification_l533_53358


namespace NUMINAMATH_CALUDE_multiplication_division_equality_l533_53355

theorem multiplication_division_equality : (3.6 * 0.25) / 0.5 = 1.8 := by
  sorry

end NUMINAMATH_CALUDE_multiplication_division_equality_l533_53355


namespace NUMINAMATH_CALUDE_mushroom_collection_l533_53309

theorem mushroom_collection (a b v g : ℚ) 
  (eq1 : a / 2 + 2 * b = v + g) 
  (eq2 : a + b = v / 2 + 2 * g) : 
  v = 2 * b ∧ a = 2 * g := by
  sorry

end NUMINAMATH_CALUDE_mushroom_collection_l533_53309


namespace NUMINAMATH_CALUDE_quadratic_function_property_l533_53390

theorem quadratic_function_property (b c m n : ℝ) :
  let f := fun (x : ℝ) => x^2 + b*x + c
  (f m = n ∧ f (m + 1) = n) →
  (∀ x₁ x₂, x₁ < x₂ ∧ x₂ ≤ 2 → f x₁ > f x₂) →
  m ≥ 3/2 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_function_property_l533_53390


namespace NUMINAMATH_CALUDE_weight_of_replaced_person_l533_53362

theorem weight_of_replaced_person (initial_count : ℕ) (average_increase : ℝ) (new_person_weight : ℝ) : ℝ :=
  let replaced_person_weight := new_person_weight - initial_count * average_increase
  replaced_person_weight

#check weight_of_replaced_person 8 5 75 -- Should evaluate to 35

end NUMINAMATH_CALUDE_weight_of_replaced_person_l533_53362


namespace NUMINAMATH_CALUDE_hiking_resupply_percentage_l533_53328

/-- A hiking problem with resupply calculation -/
theorem hiking_resupply_percentage
  (supplies_per_mile : Real)
  (hiking_speed : Real)
  (hours_per_day : Real)
  (days : Real)
  (first_pack_weight : Real)
  (h1 : supplies_per_mile = 0.5)
  (h2 : hiking_speed = 2.5)
  (h3 : hours_per_day = 8)
  (h4 : days = 5)
  (h5 : first_pack_weight = 40) :
  let total_distance := hiking_speed * hours_per_day * days
  let total_supplies := total_distance * supplies_per_mile
  let resupply_weight := total_supplies - first_pack_weight
  resupply_weight / first_pack_weight * 100 = 25 := by
  sorry

#check hiking_resupply_percentage

end NUMINAMATH_CALUDE_hiking_resupply_percentage_l533_53328


namespace NUMINAMATH_CALUDE_angle_c_in_triangle_l533_53300

/-- In a triangle ABC, if the sum of angles A and B is 80°, then angle C is 100°. -/
theorem angle_c_in_triangle (A B C : ℝ) (h1 : A + B = 80) (h2 : A + B + C = 180) : C = 100 := by
  sorry

end NUMINAMATH_CALUDE_angle_c_in_triangle_l533_53300


namespace NUMINAMATH_CALUDE_fraction_who_say_dislike_but_like_l533_53351

/-- Represents the student population at Greendale College --/
structure StudentPopulation where
  total : ℝ
  likesSwimming : ℝ
  dislikesSwimming : ℝ
  likesSayLike : ℝ
  likesSayDislike : ℝ
  dislikesSayLike : ℝ
  dislikesSayDislike : ℝ

/-- Conditions of the problem --/
def greendaleCollege : StudentPopulation where
  total := 100
  likesSwimming := 70
  dislikesSwimming := 30
  likesSayLike := 0.75 * 70
  likesSayDislike := 0.25 * 70
  dislikesSayLike := 0.15 * 30
  dislikesSayDislike := 0.85 * 30

/-- The main theorem to prove --/
theorem fraction_who_say_dislike_but_like (ε : ℝ) (hε : ε > 0) :
  let totalSayDislike := greendaleCollege.likesSayDislike + greendaleCollege.dislikesSayDislike
  let fraction := greendaleCollege.likesSayDislike / totalSayDislike
  abs (fraction - 0.407) < ε := by
  sorry


end NUMINAMATH_CALUDE_fraction_who_say_dislike_but_like_l533_53351


namespace NUMINAMATH_CALUDE_inequality_holds_l533_53338

-- Define the function f
def f (x : ℝ) : ℝ := 2 * x^2 + 3 * x + 1

-- State the theorem
theorem inequality_holds (a b : ℝ) (ha : a > 2) (hb : b > 0) :
  ∀ x, |x + 1| < b → |2 * f x - 4| < a := by
  sorry

end NUMINAMATH_CALUDE_inequality_holds_l533_53338


namespace NUMINAMATH_CALUDE_parabola_equation_l533_53395

/-- Parabola with focus F and point M -/
structure Parabola where
  p : ℝ
  F : ℝ × ℝ
  M : ℝ × ℝ
  h_p_pos : p > 0
  h_F : F = (p/2, 0)
  h_M_on_C : M.2^2 = 2 * p * M.1
  h_MF_dist : Real.sqrt ((M.1 - F.1)^2 + (M.2 - F.2)^2) = 5

/-- Circle with diameter MF passing through (0,2) -/
def circle_passes_through (P : Parabola) : Prop :=
  let center := ((P.M.1 + P.F.1)/2, (P.M.2 + P.F.2)/2)
  Real.sqrt (center.1^2 + (center.2 - 2)^2) = Real.sqrt ((P.M.1 - P.F.1)^2 + (P.M.2 - P.F.2)^2) / 2

/-- Main theorem -/
theorem parabola_equation (P : Parabola) (h_circle : circle_passes_through P) :
  P.p = 2 ∨ P.p = 8 :=
sorry

end NUMINAMATH_CALUDE_parabola_equation_l533_53395


namespace NUMINAMATH_CALUDE_p_less_than_q_l533_53357

theorem p_less_than_q (a : ℝ) (h : a ≥ 0) : 
  Real.sqrt a + Real.sqrt (a + 5) < Real.sqrt (a + 2) + Real.sqrt (a + 3) := by
sorry

end NUMINAMATH_CALUDE_p_less_than_q_l533_53357


namespace NUMINAMATH_CALUDE_complex_expression_equality_l533_53385

theorem complex_expression_equality : ∀ (a b : ℂ), 
  a = 3 - 2*I ∧ b = -2 + 3*I → 3*a + 4*b = 1 + 6*I :=
by
  sorry

end NUMINAMATH_CALUDE_complex_expression_equality_l533_53385


namespace NUMINAMATH_CALUDE_speeding_ticket_percentage_l533_53383

theorem speeding_ticket_percentage
  (total_motorists : ℝ)
  (exceed_limit_percentage : ℝ)
  (no_ticket_percentage : ℝ)
  (h1 : exceed_limit_percentage = 0.5)
  (h2 : no_ticket_percentage = 0.2)
  (h3 : total_motorists > 0) :
  let speeding_motorists := total_motorists * exceed_limit_percentage
  let no_ticket_motorists := speeding_motorists * no_ticket_percentage
  let ticket_motorists := speeding_motorists - no_ticket_motorists
  ticket_motorists / total_motorists = 0.4 :=
sorry

end NUMINAMATH_CALUDE_speeding_ticket_percentage_l533_53383


namespace NUMINAMATH_CALUDE_min_value_on_interval_l533_53367

/-- The function f(x) = -x³ + 3x² + 9x + a -/
def f (a : ℝ) (x : ℝ) : ℝ := -x^3 + 3*x^2 + 9*x + a

theorem min_value_on_interval (a : ℝ) :
  (∃ x ∈ Set.Icc (-2 : ℝ) 2, f a x = 20 ∧ ∀ y ∈ Set.Icc (-2 : ℝ) 2, f a y ≤ f a x) →
  (∃ x ∈ Set.Icc (-2 : ℝ) 2, f a x = -7 ∧ ∀ y ∈ Set.Icc (-2 : ℝ) 2, f a x ≤ f a y) :=
by sorry

end NUMINAMATH_CALUDE_min_value_on_interval_l533_53367


namespace NUMINAMATH_CALUDE_smallest_product_factors_l533_53336

/-- A structure representing an arithmetic sequence -/
structure ArithmeticSequence where
  a : ℕ -- first term
  d : ℕ -- common difference

/-- A structure representing a geometric sequence -/
structure GeometricSequence where
  b : ℕ -- first term
  r : ℕ -- common ratio

/-- The product of the first four terms of an arithmetic sequence -/
def arithProduct (seq : ArithmeticSequence) : ℕ :=
  seq.a * (seq.a + seq.d) * (seq.a + 2*seq.d) * (seq.a + 3*seq.d)

/-- The product of the first four terms of a geometric sequence -/
def geoProduct (seq : GeometricSequence) : ℕ :=
  seq.b * (seq.b * seq.r) * (seq.b * seq.r^2) * (seq.b * seq.r^3)

/-- The number of positive factors of a natural number -/
def numPositiveFactors (n : ℕ) : ℕ := sorry

/-- The theorem to be proved -/
theorem smallest_product_factors : 
  ∃ (n : ℕ) (arith : ArithmeticSequence) (geo : GeometricSequence), 
    n > 500000 ∧ 
    n = arithProduct arith ∧ 
    n = geoProduct geo ∧
    (∀ m, m > 500000 → m = arithProduct arith → m = geoProduct geo → m ≥ n) ∧
    numPositiveFactors n = 56 := by
  sorry

end NUMINAMATH_CALUDE_smallest_product_factors_l533_53336


namespace NUMINAMATH_CALUDE_marble_draw_probability_l533_53333

/-- Represents a bag of marbles -/
structure MarbleBag where
  white : ℕ := 0
  black : ℕ := 0
  yellow : ℕ := 0
  blue : ℕ := 0
  red : ℕ := 0
  green : ℕ := 0

/-- Calculate the total number of marbles in a bag -/
def MarbleBag.total (bag : MarbleBag) : ℕ :=
  bag.white + bag.black + bag.yellow + bag.blue + bag.red + bag.green

/-- Definition of Bag A -/
def bagA : MarbleBag := { white := 5, black := 5 }

/-- Definition of Bag B -/
def bagB : MarbleBag := { yellow := 8, blue := 7 }

/-- Definition of Bag C -/
def bagC : MarbleBag := { yellow := 3, blue := 7 }

/-- Definition of Bag D -/
def bagD : MarbleBag := { red := 4, green := 6 }

/-- Probability of drawing a yellow marble from a bag -/
def probYellow (bag : MarbleBag) : ℚ :=
  bag.yellow / bag.total

/-- Probability of drawing a green marble from a bag -/
def probGreen (bag : MarbleBag) : ℚ :=
  bag.green / bag.total

/-- Main theorem: Probability of drawing yellow as second and green as third marble -/
theorem marble_draw_probability : 
  (1/2 * probYellow bagB + 1/2 * probYellow bagC) * probGreen bagD = 17/50 := by
  sorry

end NUMINAMATH_CALUDE_marble_draw_probability_l533_53333


namespace NUMINAMATH_CALUDE_fibonacci_closed_form_l533_53379

def fibonacci : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | (n + 2) => fibonacci (n + 1) + fibonacci n

theorem fibonacci_closed_form (a b : ℝ) (h1 : a + b = 1) (h2 : a * b = -1) (h3 : a > b) :
  ∀ n : ℕ, fibonacci n = (a^(n+1) - b^(n+1)) / Real.sqrt 5 :=
by sorry

end NUMINAMATH_CALUDE_fibonacci_closed_form_l533_53379


namespace NUMINAMATH_CALUDE_point_movement_theorem_l533_53380

theorem point_movement_theorem (A : ℝ) : 
  (A + 7 - 4 = 0) → A = -3 := by
  sorry

end NUMINAMATH_CALUDE_point_movement_theorem_l533_53380


namespace NUMINAMATH_CALUDE_min_sum_of_digits_3n2_plus_n_plus_1_l533_53388

/-- Sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- Theorem: The smallest sum of digits of 3n^2 + n + 1 for positive integer n is 3 -/
theorem min_sum_of_digits_3n2_plus_n_plus_1 :
  (∀ n : ℕ+, sum_of_digits (3 * n^2 + n + 1) ≥ 3) ∧
  (∃ n : ℕ+, sum_of_digits (3 * n^2 + n + 1) = 3) :=
sorry

end NUMINAMATH_CALUDE_min_sum_of_digits_3n2_plus_n_plus_1_l533_53388


namespace NUMINAMATH_CALUDE_triangle_circumcircle_intersection_l533_53342

theorem triangle_circumcircle_intersection (PQ QR RP : ℝ) (h1 : PQ = 39) (h2 : QR = 15) (h3 : RP = 50) : 
  ∃ (PS : ℝ), PS = 5 * Real.sqrt 61 ∧ 
  ⌊5 + Real.sqrt 61⌋ = 13 :=
by sorry

end NUMINAMATH_CALUDE_triangle_circumcircle_intersection_l533_53342


namespace NUMINAMATH_CALUDE_current_speed_l533_53343

theorem current_speed (speed_with_current speed_against_current : ℝ) 
  (h1 : speed_with_current = 15)
  (h2 : speed_against_current = 8.6) :
  ∃ (current_speed : ℝ), current_speed = 3.2 :=
by
  sorry

end NUMINAMATH_CALUDE_current_speed_l533_53343


namespace NUMINAMATH_CALUDE_beaus_age_is_42_l533_53359

/-- Beau's age today given his triplet sons' ages and a condition from the past -/
def beaus_age_today (sons_age_today : ℕ) : ℕ :=
  let sons_age_past := sons_age_today - 3
  let beaus_age_past := 3 * sons_age_past
  beaus_age_past + 3

theorem beaus_age_is_42 :
  beaus_age_today 16 = 42 := by sorry

end NUMINAMATH_CALUDE_beaus_age_is_42_l533_53359


namespace NUMINAMATH_CALUDE_simplify_sqrt_expression_l533_53350

theorem simplify_sqrt_expression (t : ℝ) : 
  Real.sqrt (t^6 + t^4) = t^2 * Real.sqrt (t^2 + 1) := by
  sorry

end NUMINAMATH_CALUDE_simplify_sqrt_expression_l533_53350


namespace NUMINAMATH_CALUDE_garage_sale_ratio_l533_53375

theorem garage_sale_ratio (treadmill_price chest_price tv_price total_sale : ℚ) : 
  treadmill_price = 100 →
  chest_price = treadmill_price / 2 →
  total_sale = 600 →
  total_sale = treadmill_price + chest_price + tv_price →
  tv_price / treadmill_price = 9 / 2 := by
  sorry

end NUMINAMATH_CALUDE_garage_sale_ratio_l533_53375


namespace NUMINAMATH_CALUDE_system_solution_existence_l533_53397

/-- The system of equations has at least one solution for some b iff a ≥ -√2 - 1/4 -/
theorem system_solution_existence (a : ℝ) : 
  (∃ (b x y : ℝ), y = x^2 - a ∧ x^2 + y^2 + 8*b^2 = 4*b*(y - x) + 1) ↔ 
  a ≥ -Real.sqrt 2 - 1/4 := by sorry

end NUMINAMATH_CALUDE_system_solution_existence_l533_53397


namespace NUMINAMATH_CALUDE_existence_of_equal_modulus_unequal_squares_l533_53345

theorem existence_of_equal_modulus_unequal_squares : ∃ (z₁ z₂ : ℂ), Complex.abs z₁ = Complex.abs z₂ ∧ z₁^2 ≠ z₂^2 := by
  sorry

end NUMINAMATH_CALUDE_existence_of_equal_modulus_unequal_squares_l533_53345


namespace NUMINAMATH_CALUDE_power_multiplication_equals_128_l533_53378

theorem power_multiplication_equals_128 : 
  ∀ b : ℕ, b = 2 → b^3 * b^4 = 128 := by
  sorry

end NUMINAMATH_CALUDE_power_multiplication_equals_128_l533_53378


namespace NUMINAMATH_CALUDE_experiment_is_conditional_control_l533_53315

-- Define the types of control experiments
inductive ControlType
  | Blank
  | Standard
  | Mutual
  | Conditional

-- Define the components of a culture medium
structure CultureMedium where
  urea : Bool
  nitrate : Bool
  otherComponents : Set String

-- Define an experimental group
structure ExperimentalGroup where
  medium : CultureMedium

-- Define the experiment
structure Experiment where
  groupA : ExperimentalGroup
  groupB : ExperimentalGroup
  sameOtherConditions : Bool

def isConditionalControl (exp : Experiment) : Prop :=
  exp.groupA.medium.urea = true ∧
  exp.groupA.medium.nitrate = false ∧
  exp.groupB.medium.urea = true ∧
  exp.groupB.medium.nitrate = true ∧
  exp.groupA.medium.otherComponents = exp.groupB.medium.otherComponents ∧
  exp.sameOtherConditions = true

theorem experiment_is_conditional_control (exp : Experiment) 
  (h1 : exp.groupA.medium.urea = true)
  (h2 : exp.groupA.medium.nitrate = false)
  (h3 : exp.groupB.medium.urea = true)
  (h4 : exp.groupB.medium.nitrate = true)
  (h5 : exp.groupA.medium.otherComponents = exp.groupB.medium.otherComponents)
  (h6 : exp.sameOtherConditions = true) :
  isConditionalControl exp :=
by sorry

end NUMINAMATH_CALUDE_experiment_is_conditional_control_l533_53315


namespace NUMINAMATH_CALUDE_hawks_score_l533_53393

theorem hawks_score (total_points eagles_margin hawks_min_score : ℕ) 
  (h1 : total_points = 82)
  (h2 : eagles_margin = 18)
  (h3 : hawks_min_score = 9)
  (h4 : ∃ (hawks_score : ℕ), 
    hawks_score ≥ hawks_min_score ∧ 
    hawks_score + (hawks_score + eagles_margin) = total_points) :
  ∃ (hawks_score : ℕ), hawks_score = 32 :=
by sorry

end NUMINAMATH_CALUDE_hawks_score_l533_53393


namespace NUMINAMATH_CALUDE_smallest_consecutive_odd_divisibility_l533_53356

theorem smallest_consecutive_odd_divisibility (n : ℕ+) :
  ∃ (u_n : ℕ+),
    (∀ (d : ℕ+) (a : ℕ),
      (∀ k : Fin u_n, ∃ m : ℕ, a + 2 * k.val = d * m) →
      (∀ k : Fin n, ∃ m : ℕ, 2 * k.val + 1 = d * m)) ∧
    (∀ (v : ℕ+),
      v < u_n →
      ∃ (d : ℕ+) (a : ℕ),
        (∀ k : Fin v, ∃ m : ℕ, a + 2 * k.val = d * m) ∧
        ¬(∀ k : Fin n, ∃ m : ℕ, 2 * k.val + 1 = d * m)) ∧
    u_n = 2 * n - 1 := by
  sorry

end NUMINAMATH_CALUDE_smallest_consecutive_odd_divisibility_l533_53356


namespace NUMINAMATH_CALUDE_set_relationship_l533_53324

def A : Set ℝ := {x | x^2 - 8*x + 15 = 0}
def B (a : ℝ) : Set ℝ := {x | a*x - 1 = 0}

theorem set_relationship :
  (¬(B (1/5) ⊆ A)) ∧
  (∀ a : ℝ, (B a ⊆ A) ↔ (a = 0 ∨ a = 1/3 ∨ a = 1/5)) :=
sorry

end NUMINAMATH_CALUDE_set_relationship_l533_53324


namespace NUMINAMATH_CALUDE_candy_shipment_proof_l533_53312

/-- Represents the number of cases of each candy type in a shipment -/
structure CandyShipment where
  chocolate : ℕ
  lollipops : ℕ
  gummy_bears : ℕ

/-- The ratio of chocolate bars to lollipops to gummy bears -/
def candy_ratio : CandyShipment := ⟨3, 2, 1⟩

/-- The actual shipment received -/
def actual_shipment : CandyShipment := ⟨36, 48, 24⟩

theorem candy_shipment_proof :
  (actual_shipment.chocolate / candy_ratio.chocolate = 
   actual_shipment.lollipops / candy_ratio.lollipops) ∧
  (actual_shipment.gummy_bears = 
   actual_shipment.chocolate / candy_ratio.chocolate * candy_ratio.gummy_bears) ∧
  (actual_shipment.chocolate + actual_shipment.lollipops + actual_shipment.gummy_bears = 108) :=
by sorry

#check candy_shipment_proof

end NUMINAMATH_CALUDE_candy_shipment_proof_l533_53312


namespace NUMINAMATH_CALUDE_domain_of_sqrt_tan_minus_sqrt3_l533_53301

/-- The domain of the function y = √(tan x - √3) -/
theorem domain_of_sqrt_tan_minus_sqrt3 (x : ℝ) :
  x ∈ {x : ℝ | ∃ k : ℤ, k * π + π / 3 ≤ x ∧ x < k * π + π / 2} ↔
  ∃ y : ℝ, y = Real.sqrt (Real.tan x - Real.sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_domain_of_sqrt_tan_minus_sqrt3_l533_53301


namespace NUMINAMATH_CALUDE_game_lives_per_player_l533_53376

theorem game_lives_per_player (initial_players : ℕ) (additional_players : ℕ) (total_lives : ℕ) :
  initial_players = 8 →
  additional_players = 2 →
  total_lives = 60 →
  (total_lives / (initial_players + additional_players) : ℚ) = 6 := by
  sorry

end NUMINAMATH_CALUDE_game_lives_per_player_l533_53376


namespace NUMINAMATH_CALUDE_cylinder_volume_l533_53311

/-- The volume of a cylinder with base radius 1 cm and height 2 cm is 2π cm³ -/
theorem cylinder_volume : 
  let r : ℝ := 1  -- base radius in cm
  let h : ℝ := 2  -- height in cm
  let V : ℝ := π * r^2 * h  -- volume formula
  V = 2 * π := by
  sorry

end NUMINAMATH_CALUDE_cylinder_volume_l533_53311


namespace NUMINAMATH_CALUDE_square_sum_equals_90_l533_53322

theorem square_sum_equals_90 (x y : ℝ) (h1 : x + 3*y = 6) (h2 : x*y = -9) : 
  x^2 + 9*y^2 = 90 := by
sorry

end NUMINAMATH_CALUDE_square_sum_equals_90_l533_53322


namespace NUMINAMATH_CALUDE_exponential_is_self_derivative_l533_53323

theorem exponential_is_self_derivative : 
  ∃ f : ℝ → ℝ, (∀ x, f x = Real.exp x) ∧ (∀ x, deriv f x = f x) :=
sorry

end NUMINAMATH_CALUDE_exponential_is_self_derivative_l533_53323


namespace NUMINAMATH_CALUDE_range_of_a_l533_53310

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, a * x^2 + a * x + 1 > 0) ∨ 
  (∃ x₀ : ℝ, x₀^2 - x₀ + a = 0) ∧ 
  ¬((∀ x : ℝ, a * x^2 + a * x + 1 > 0) ∧ 
    (∃ x₀ : ℝ, x₀^2 - x₀ + a = 0)) →
  a < 0 ∨ (1/4 < a ∧ a < 4) :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l533_53310


namespace NUMINAMATH_CALUDE_removed_sector_angle_l533_53391

/-- Given a circular piece of paper with radius 15 cm, if a cone is formed from the remaining sector
    after removing a part, and this cone has a radius of 10 cm and a volume of 500π cm³,
    then the angle measure of the removed sector is 120°. -/
theorem removed_sector_angle (paper_radius : ℝ) (cone_radius : ℝ) (cone_volume : ℝ) :
  paper_radius = 15 →
  cone_radius = 10 →
  cone_volume = 500 * Real.pi →
  ∃ (removed_angle : ℝ), removed_angle = 120 ∧ 0 ≤ removed_angle ∧ removed_angle ≤ 360 :=
by sorry

end NUMINAMATH_CALUDE_removed_sector_angle_l533_53391


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l533_53303

theorem quadratic_inequality_range (a : ℝ) : 
  (∀ x : ℝ, x^2 + 2*x + a > 0) → a > 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l533_53303


namespace NUMINAMATH_CALUDE_solution_triples_l533_53346

theorem solution_triples : 
  ∀ (x y : ℤ) (m : ℝ),
    x < 0 ∧ y > 0 ∧ 
    -2 * x + 3 * y = 2 * m ∧
    x - 5 * y = -11 →
    ((x = -6 ∧ y = 1 ∧ m = 7.5) ∨ (x = -1 ∧ y = 2 ∧ m = 4)) :=
by sorry

end NUMINAMATH_CALUDE_solution_triples_l533_53346


namespace NUMINAMATH_CALUDE_download_speed_scientific_notation_l533_53372

/-- The download speed of a 5G network in KB per second -/
def download_speed : ℝ := 1300000

/-- Scientific notation representation of the download speed -/
def scientific_notation : ℝ := 1.3 * (10 ^ 6)

theorem download_speed_scientific_notation : 
  download_speed = scientific_notation := by
  sorry

end NUMINAMATH_CALUDE_download_speed_scientific_notation_l533_53372


namespace NUMINAMATH_CALUDE_condition_relationship_l533_53349

theorem condition_relationship (x : ℝ) : 
  (∀ x, x = 1 → x^2 - 3*x + 2 = 0) ∧ 
  (∃ x, x^2 - 3*x + 2 = 0 ∧ x ≠ 1) := by
  sorry

end NUMINAMATH_CALUDE_condition_relationship_l533_53349


namespace NUMINAMATH_CALUDE_fraction_to_decimal_l533_53331

theorem fraction_to_decimal : (29 : ℚ) / 160 = 0.18125 := by
  sorry

end NUMINAMATH_CALUDE_fraction_to_decimal_l533_53331


namespace NUMINAMATH_CALUDE_leftover_books_l533_53399

/-- The number of leftover books when repacking from boxes of 45 to boxes of 47 -/
theorem leftover_books (initial_boxes : Nat) (books_per_initial_box : Nat) (books_per_new_box : Nat) :
  initial_boxes = 1500 →
  books_per_initial_box = 45 →
  books_per_new_box = 47 →
  (initial_boxes * books_per_initial_box) % books_per_new_box = 13 := by
  sorry

#eval (1500 * 45) % 47  -- This should output 13

end NUMINAMATH_CALUDE_leftover_books_l533_53399


namespace NUMINAMATH_CALUDE_height_average_comparison_l533_53334

theorem height_average_comparison 
  (h₁ : ℝ → ℝ → ℝ → ℝ → ℝ → Prop) 
  (a b c d : ℝ) 
  (h₂ : 3 * a + 2 * b = 2 * c + 3 * d) 
  (h₃ : a > d) : 
  |c + d| / 2 > |a + b| / 2 := by
sorry

end NUMINAMATH_CALUDE_height_average_comparison_l533_53334


namespace NUMINAMATH_CALUDE_book_selection_combinations_l533_53386

/-- The number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := Nat.choose n k

/-- The total number of books in the library -/
def total_books : ℕ := 15

/-- The number of books to be selected -/
def selected_books : ℕ := 3

/-- Theorem: The number of ways to choose 3 books from 15 books is 455 -/
theorem book_selection_combinations :
  choose total_books selected_books = 455 := by sorry

end NUMINAMATH_CALUDE_book_selection_combinations_l533_53386


namespace NUMINAMATH_CALUDE_sam_puppies_count_l533_53305

def final_puppies (initial bought given_away sold : ℕ) : ℕ :=
  initial - given_away + bought - sold

theorem sam_puppies_count : final_puppies 72 25 18 13 = 66 := by
  sorry

end NUMINAMATH_CALUDE_sam_puppies_count_l533_53305


namespace NUMINAMATH_CALUDE_triangle_area_l533_53392

theorem triangle_area (a b c : ℝ) (h1 : a = 4) (h2 : b = 4) (h3 : c = 6) :
  let s := (a + b + c) / 2
  Real.sqrt (s * (s - a) * (s - b) * (s - c)) = 3 * Real.sqrt 7 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_l533_53392


namespace NUMINAMATH_CALUDE_symmetric_points_sum_power_l533_53306

/-- Two points are symmetric about the y-axis if their y-coordinates are the same and their x-coordinates are opposites -/
def symmetric_about_y_axis (p1 p2 : ℝ × ℝ) : Prop :=
  p1.2 = p2.2 ∧ p1.1 = -p2.1

theorem symmetric_points_sum_power (a b : ℝ) :
  symmetric_about_y_axis (a, 3) (2, b) → (a + b)^2015 = 1 := by
  sorry

end NUMINAMATH_CALUDE_symmetric_points_sum_power_l533_53306


namespace NUMINAMATH_CALUDE_figure_b_cannot_be_formed_l533_53382

/-- A piece is represented by its width and height -/
structure Piece where
  width : ℕ
  height : ℕ

/-- A figure is represented by its width and height -/
structure Figure where
  width : ℕ
  height : ℕ

/-- The set of available pieces -/
def pieces : Finset Piece := sorry

/-- The set of figures to be formed -/
def figures : Finset Figure := sorry

/-- Function to check if a figure can be formed from the given pieces -/
def canFormFigure (p : Finset Piece) (f : Figure) : Prop := sorry

/-- Theorem stating that Figure B cannot be formed while others can -/
theorem figure_b_cannot_be_formed :
  ∃ (b : Figure),
    b ∈ figures ∧
    ¬(canFormFigure pieces b) ∧
    ∀ (f : Figure), f ∈ figures ∧ f ≠ b → canFormFigure pieces f :=
sorry

end NUMINAMATH_CALUDE_figure_b_cannot_be_formed_l533_53382


namespace NUMINAMATH_CALUDE_valleyball_club_members_l533_53348

/-- The cost of a pair of knee pads in dollars -/
def knee_pad_cost : ℕ := 6

/-- The cost of a jersey in dollars -/
def jersey_cost : ℕ := knee_pad_cost + 7

/-- The cost of a wristband in dollars -/
def wristband_cost : ℕ := jersey_cost + 3

/-- The total cost for one member's equipment (indoor and outdoor sets) -/
def member_cost : ℕ := 2 * (knee_pad_cost + jersey_cost + wristband_cost)

/-- The total cost for all members' equipment -/
def total_cost : ℕ := 4080

/-- The number of members in the Valleyball Volleyball Club -/
def club_members : ℕ := total_cost / member_cost

theorem valleyball_club_members : club_members = 58 := by
  sorry

end NUMINAMATH_CALUDE_valleyball_club_members_l533_53348


namespace NUMINAMATH_CALUDE_correct_option_is_valid_print_statement_l533_53314

-- Define an enum for the options
inductive ProgramOption
| A
| B
| C
| D

-- Define a function to check if an option is a valid print statement
def isValidPrintStatement (option : ProgramOption) : Prop :=
  match option with
  | ProgramOption.A => True  -- PRINT 4*x is valid
  | _ => False               -- Other options are not valid print statements

-- Theorem statement
theorem correct_option_is_valid_print_statement :
  ∃ (option : ProgramOption), isValidPrintStatement option :=
by
  sorry


end NUMINAMATH_CALUDE_correct_option_is_valid_print_statement_l533_53314


namespace NUMINAMATH_CALUDE_magazines_read_in_five_hours_l533_53368

/-- 
Proves that given a reading rate of 1 magazine per 20 minutes, 
the number of magazines that can be read in 5 hours is equal to 15.
-/
theorem magazines_read_in_five_hours 
  (reading_rate : ℚ) -- Reading rate in magazines per minute
  (hours : ℕ) -- Number of hours
  (h1 : reading_rate = 1 / 20) -- Reading rate is 1 magazine per 20 minutes
  (h2 : hours = 5) -- Time period is 5 hours
  : ⌊hours * 60 * reading_rate⌋ = 15 := by
  sorry

#check magazines_read_in_five_hours

end NUMINAMATH_CALUDE_magazines_read_in_five_hours_l533_53368


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l533_53353

/-- A geometric sequence -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = q * a n

/-- The theorem statement -/
theorem geometric_sequence_sum (a : ℕ → ℝ) :
  geometric_sequence a →
  a 3 + a 5 = -6 →
  a 2 * a 6 = 8 →
  a 1 + a 7 = -9 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l533_53353


namespace NUMINAMATH_CALUDE_problem_solution_l533_53325

theorem problem_solution : 
  |1 - Real.sqrt (4/3)| + (Real.sqrt 3 - 1/2)^0 = 2 * Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l533_53325


namespace NUMINAMATH_CALUDE_inequality_solution_set_l533_53317

theorem inequality_solution_set (x : ℝ) : 
  (x^2 - x - 6 < 0) ↔ (-2 < x ∧ x < 3) := by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l533_53317


namespace NUMINAMATH_CALUDE_not_all_greater_than_one_l533_53352

theorem not_all_greater_than_one (a b c : Real) 
  (ha : 0 < a ∧ a < 2) 
  (hb : 0 < b ∧ b < 2) 
  (hc : 0 < c ∧ c < 2) : 
  ¬((2 - a) * b > 1 ∧ (2 - b) * c > 1 ∧ (2 - c) * a > 1) := by
  sorry

end NUMINAMATH_CALUDE_not_all_greater_than_one_l533_53352


namespace NUMINAMATH_CALUDE_vector_sum_squared_l533_53337

variable (a b c m : ℝ × ℝ)

/-- m is the midpoint of a and b -/
def is_midpoint (m a b : ℝ × ℝ) : Prop :=
  m = ((a.1 + b.1) / 2, (a.2 + b.2) / 2)

/-- The dot product of two 2D vectors -/
def dot_product (v w : ℝ × ℝ) : ℝ :=
  v.1 * w.1 + v.2 * w.2

/-- The squared norm of a 2D vector -/
def norm_squared (v : ℝ × ℝ) : ℝ :=
  v.1 * v.1 + v.2 * v.2

theorem vector_sum_squared (a b : ℝ × ℝ) :
  is_midpoint m a b →
  m = (4, 5) →
  dot_product a b = 12 →
  dot_product c (a.1 + b.1, a.2 + b.2) = 0 →
  norm_squared a + norm_squared b = 140 := by
  sorry

end NUMINAMATH_CALUDE_vector_sum_squared_l533_53337


namespace NUMINAMATH_CALUDE_lane_length_correct_l533_53377

/-- Represents the length of a swimming lane in meters -/
def lane_length : ℝ := 100

/-- Represents the number of round trips swum -/
def round_trips : ℕ := 3

/-- Represents the total distance swum in meters -/
def total_distance : ℝ := 600

/-- Theorem stating that the lane length is correct given the conditions -/
theorem lane_length_correct : 
  lane_length * (2 * round_trips) = total_distance :=
by sorry

end NUMINAMATH_CALUDE_lane_length_correct_l533_53377


namespace NUMINAMATH_CALUDE_a_minus_b_value_l533_53374

theorem a_minus_b_value (a b : ℝ) (ha : |a| = 5) (hb : |b| = 3) (hab : a + b > 0) :
  a - b = 2 ∨ a - b = 8 := by
sorry

end NUMINAMATH_CALUDE_a_minus_b_value_l533_53374


namespace NUMINAMATH_CALUDE_second_attempt_score_l533_53371

/-- Represents the number of points scored in each attempt -/
structure Attempts where
  first : ℕ
  second : ℕ
  third : ℕ

/-- The minimum and maximum possible points for a single dart throw -/
def min_points : ℕ := 3
def max_points : ℕ := 9

/-- The number of darts thrown in each attempt -/
def num_darts : ℕ := 8

theorem second_attempt_score (a : Attempts) : 
  (a.second = 2 * a.first) → 
  (a.third = 3 * a.first) → 
  (a.first ≥ num_darts * min_points) → 
  (a.third ≤ num_darts * max_points) → 
  a.second = 48 := by
  sorry

end NUMINAMATH_CALUDE_second_attempt_score_l533_53371


namespace NUMINAMATH_CALUDE_otimes_equation_solution_l533_53369

-- Define the ⊗ operation
noncomputable def otimes (a b : ℝ) : ℝ :=
  a * Real.sqrt (b + Real.sqrt (b + 1 + Real.sqrt (b + 2 + Real.sqrt (b + 3 + Real.sqrt (b + 4)))))

-- Theorem statement
theorem otimes_equation_solution (h : ℝ) :
  otimes 3 h = 15 → h = 20 := by
  sorry

end NUMINAMATH_CALUDE_otimes_equation_solution_l533_53369


namespace NUMINAMATH_CALUDE_quadratic_intersection_point_l533_53316

/-- A quadratic function passing through given points -/
def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

theorem quadratic_intersection_point 
  (a b c : ℝ) 
  (h1 : f a b c (-3) = 16)
  (h2 : f a b c 0 = -5)
  (h3 : f a b c 3 = -8)
  (h4 : f a b c 5 = 0) :
  f a b c (-1) = 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_intersection_point_l533_53316


namespace NUMINAMATH_CALUDE_max_y_coord_sin_3theta_l533_53387

/-- The maximum y-coordinate of a point on the curve r = sin 3θ is 1 -/
theorem max_y_coord_sin_3theta :
  let r : ℝ → ℝ := λ θ ↦ Real.sin (3 * θ)
  let y : ℝ → ℝ := λ θ ↦ r θ * Real.sin θ
  ∃ (θ : ℝ), y θ = 1 ∧ ∀ (φ : ℝ), y φ ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_max_y_coord_sin_3theta_l533_53387


namespace NUMINAMATH_CALUDE_flag_raising_arrangements_l533_53363

/-- The number of classes in the first year of high school -/
def first_year_classes : ℕ := 8

/-- The number of classes in the second year of high school -/
def second_year_classes : ℕ := 6

/-- The total number of possible arrangements for selecting one class for flag-raising duty -/
def total_arrangements : ℕ := first_year_classes + second_year_classes

/-- Theorem stating that the total number of possible arrangements is 14 -/
theorem flag_raising_arrangements :
  total_arrangements = 14 := by sorry

end NUMINAMATH_CALUDE_flag_raising_arrangements_l533_53363


namespace NUMINAMATH_CALUDE_base_3_to_base_9_first_digit_l533_53332

def base_3_to_decimal (digits : List Nat) : Nat :=
  digits.enum.foldr (fun (i, d) acc => acc + d * (3 ^ i)) 0

def first_digit_base_9 (n : Nat) : Nat :=
  Nat.log 9 n

theorem base_3_to_base_9_first_digit :
  let y : Nat := base_3_to_decimal [2, 0, 2, 2, 1, 1, 2, 2, 2, 0, 1, 2, 0, 1, 0, 1, 1, 1]
  first_digit_base_9 y = 4 := by
  sorry

end NUMINAMATH_CALUDE_base_3_to_base_9_first_digit_l533_53332


namespace NUMINAMATH_CALUDE_polynomial_division_remainder_l533_53396

theorem polynomial_division_remainder : ∃ q : Polynomial ℤ, 
  3 * X^2 - 22 * X + 64 = (X - 3) * q + 25 := by sorry

end NUMINAMATH_CALUDE_polynomial_division_remainder_l533_53396


namespace NUMINAMATH_CALUDE_inequality_proof_l533_53384

theorem inequality_proof (k l m n : ℕ) 
  (h1 : k < l) (h2 : l < m) (h3 : m < n) (h4 : l * m = k * n) : 
  ((n - k) / 2 : ℚ)^2 ≥ k + 2 := by sorry

end NUMINAMATH_CALUDE_inequality_proof_l533_53384


namespace NUMINAMATH_CALUDE_cube_equation_solution_l533_53389

theorem cube_equation_solution (x : ℝ) : (x + 3)^3 = -64 → x = -7 := by
  sorry

end NUMINAMATH_CALUDE_cube_equation_solution_l533_53389


namespace NUMINAMATH_CALUDE_imaginary_part_of_2_minus_i_l533_53304

theorem imaginary_part_of_2_minus_i :
  let z : ℂ := 2 - I
  Complex.im z = -1 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_2_minus_i_l533_53304


namespace NUMINAMATH_CALUDE_max_value_fraction_l533_53364

theorem max_value_fraction (x y : ℝ) (hx : -4 ≤ x ∧ x ≤ -2) (hy : 2 ≤ y ∧ y ≤ 4) :
  (∀ a b : ℝ, -4 ≤ a ∧ a ≤ -2 → 2 ≤ b ∧ b ≤ 4 → (x + y) / x ≥ (a + b) / a) →
  (x + y) / x = 0 :=
by sorry

end NUMINAMATH_CALUDE_max_value_fraction_l533_53364


namespace NUMINAMATH_CALUDE_integral_problem_l533_53370

theorem integral_problem : ∫ x in (0)..(2 * Real.arctan (1/2)), (1 - Real.sin x) / (Real.cos x * (1 + Real.cos x)) = 2 * Real.log (3/2) - 1/2 := by
  sorry

end NUMINAMATH_CALUDE_integral_problem_l533_53370


namespace NUMINAMATH_CALUDE_circle_area_ratio_l533_53318

theorem circle_area_ratio (R : ℝ) (R_pos : R > 0) : 
  (π * (R/3)^2) / (π * R^2) = 1/9 := by
  sorry

end NUMINAMATH_CALUDE_circle_area_ratio_l533_53318


namespace NUMINAMATH_CALUDE_middle_frequency_is_32_l533_53344

/-- Represents a frequency distribution histogram -/
structure Histogram where
  n : ℕ  -- number of rectangles
  middle_area : ℕ  -- area of the middle rectangle
  total_area : ℕ  -- total area of the histogram
  h_area_sum : middle_area + (n - 1) * middle_area = total_area  -- area sum condition
  h_total_area : total_area = 160  -- total area is 160

/-- The frequency of the middle group in the histogram is 32 -/
theorem middle_frequency_is_32 (h : Histogram) : h.middle_area = 32 := by
  sorry

end NUMINAMATH_CALUDE_middle_frequency_is_32_l533_53344


namespace NUMINAMATH_CALUDE_area_of_similar_pentagons_l533_53398

/-- Theorem: Area of similar pentagons
  Given two similar pentagons with perimeters K₁ and K₂, and areas L₁ and L₂,
  if K₁ = 18, K₂ = 24, and L₁ = 8 7/16, then L₂ = 15.
-/
theorem area_of_similar_pentagons (K₁ K₂ L₁ L₂ : ℝ) : 
  K₁ = 18 → K₂ = 24 → L₁ = 8 + 7/16 → 
  (K₁ / K₂)^2 = L₁ / L₂ → 
  L₂ = 15 := by
  sorry


end NUMINAMATH_CALUDE_area_of_similar_pentagons_l533_53398


namespace NUMINAMATH_CALUDE_rectangle_area_change_l533_53354

theorem rectangle_area_change (initial_short : ℝ) (initial_long : ℝ) 
  (h1 : initial_short = 5)
  (h2 : initial_long = 7)
  (h3 : ∃ x, initial_short * (initial_long - x) = 24) :
  (initial_short * (initial_long - 2) = 25) := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_change_l533_53354


namespace NUMINAMATH_CALUDE_min_value_x_plus_y_l533_53327

theorem min_value_x_plus_y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 1/x + 9/y = 1) :
  x + y ≥ 16 ∧ ∃ x₀ y₀ : ℝ, x₀ > 0 ∧ y₀ > 0 ∧ 1/x₀ + 9/y₀ = 1 ∧ x₀ + y₀ = 16 := by
  sorry

end NUMINAMATH_CALUDE_min_value_x_plus_y_l533_53327


namespace NUMINAMATH_CALUDE_at_least_one_not_less_than_two_l533_53381

theorem at_least_one_not_less_than_two (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a + 1 / b ≥ 2) ∨ (b + 1 / c ≥ 2) ∨ (c + 1 / a ≥ 2) := by
  sorry

end NUMINAMATH_CALUDE_at_least_one_not_less_than_two_l533_53381


namespace NUMINAMATH_CALUDE_non_union_women_percentage_l533_53366

/-- Represents the composition of employees in a company -/
structure CompanyEmployees where
  total : ℝ
  men : ℝ
  unionized : ℝ
  unionized_men : ℝ

/-- Conditions given in the problem -/
def company_conditions (c : CompanyEmployees) : Prop :=
  c.men / c.total = 0.54 ∧
  c.unionized / c.total = 0.6 ∧
  c.unionized_men / c.unionized = 0.7

/-- The theorem to be proved -/
theorem non_union_women_percentage (c : CompanyEmployees) 
  (h : company_conditions c) : 
  (c.total - c.unionized - (c.men - c.unionized_men)) / (c.total - c.unionized) = 0.7 := by
  sorry

end NUMINAMATH_CALUDE_non_union_women_percentage_l533_53366


namespace NUMINAMATH_CALUDE_square_side_length_l533_53307

theorem square_side_length (area : ℝ) (side : ℝ) : 
  area = 64 → side * side = area → side = 8 := by
  sorry

end NUMINAMATH_CALUDE_square_side_length_l533_53307


namespace NUMINAMATH_CALUDE_choose_three_from_ten_l533_53365

theorem choose_three_from_ten (n : ℕ) (k : ℕ) :
  n = 10 → k = 3 → Nat.choose n k = 120 := by sorry

end NUMINAMATH_CALUDE_choose_three_from_ten_l533_53365


namespace NUMINAMATH_CALUDE_parabola_circle_theorem_trajectory_theorem_l533_53361

-- Define the parabola
def parabola (p : ℝ) (x y : ℝ) : Prop := y^2 = 2*p*x ∧ p > 0

-- Define the line passing through (1,0)
def line (k : ℝ) (x y : ℝ) : Prop := y = k*(x - 1)

-- Define the circle condition
def circle_condition (x₁ y₁ x₂ y₂ : ℝ) : Prop := x₁*x₂ + y₁*y₂ = 0

-- Define the vector equation
def vector_equation (x y x₁ y₁ x₂ y₂ : ℝ) : Prop := 
  x = x₁ + x₂ - 1/4 ∧ y = y₁ + y₂

-- Theorem 1
theorem parabola_circle_theorem (p : ℝ) :
  (∃ k x₁ y₁ x₂ y₂ : ℝ, 
    parabola p x₁ y₁ ∧ parabola p x₂ y₂ ∧
    line k x₁ y₁ ∧ line k x₂ y₂ ∧
    circle_condition x₁ y₁ x₂ y₂) →
  p = 1/2 :=
sorry

-- Theorem 2
theorem trajectory_theorem (p : ℝ) (x y : ℝ) :
  p = 1/2 →
  (∃ k x₁ y₁ x₂ y₂ : ℝ, 
    parabola p x₁ y₁ ∧ parabola p x₂ y₂ ∧
    line k x₁ y₁ ∧ line k x₂ y₂ ∧
    circle_condition x₁ y₁ x₂ y₂ ∧
    vector_equation x y x₁ y₁ x₂ y₂) →
  y^2 = x - 7/4 :=
sorry

end NUMINAMATH_CALUDE_parabola_circle_theorem_trajectory_theorem_l533_53361


namespace NUMINAMATH_CALUDE_unique_bisecting_line_l533_53302

/-- A triangle with sides 6, 8, and 10 -/
structure RightTriangle where
  side1 : ℝ
  side2 : ℝ
  hypotenuse : ℝ
  side1_eq : side1 = 6
  side2_eq : side2 = 8
  hypotenuse_eq : hypotenuse = 10
  pythagoras : side1^2 + side2^2 = hypotenuse^2

/-- A line that potentially bisects the area and perimeter of the triangle -/
structure BisectingLine (t : RightTriangle) where
  x : ℝ  -- distance from a vertex on one side
  y : ℝ  -- distance from the same vertex on another side
  bisects_area : x * y = 30  -- specific to this triangle
  bisects_perimeter : x + y = (t.side1 + t.side2 + t.hypotenuse) / 2

/-- There exists a unique bisecting line for the given right triangle -/
theorem unique_bisecting_line (t : RightTriangle) : 
  ∃! (l : BisectingLine t), True :=
sorry

end NUMINAMATH_CALUDE_unique_bisecting_line_l533_53302


namespace NUMINAMATH_CALUDE_min_sum_of_product_72_l533_53313

theorem min_sum_of_product_72 (a b : ℤ) (h : a * b = 72) : 
  (∀ x y : ℤ, x * y = 72 → a + b ≤ x + y) ∧ (∃ x y : ℤ, x * y = 72 ∧ x + y = -17) := by
  sorry

end NUMINAMATH_CALUDE_min_sum_of_product_72_l533_53313


namespace NUMINAMATH_CALUDE_remainder_sum_l533_53308

theorem remainder_sum (x y : ℤ) : 
  x % 80 = 75 → y % 120 = 115 → (x + y) % 40 = 30 := by sorry

end NUMINAMATH_CALUDE_remainder_sum_l533_53308


namespace NUMINAMATH_CALUDE_min_value_expression_l533_53373

theorem min_value_expression (x y : ℝ) 
  (h1 : x * y + 3 * x = 3)
  (h2 : 0 < x)
  (h3 : x < 1/2) :
  ∃ (m : ℝ), m = 8 ∧ ∀ (x' y' : ℝ), 
    x' * y' + 3 * x' = 3 → 
    0 < x' → 
    x' < 1/2 → 
    3 / x' + 1 / (y' - 3) ≥ m :=
sorry

end NUMINAMATH_CALUDE_min_value_expression_l533_53373


namespace NUMINAMATH_CALUDE_opposites_and_reciprocals_problem_l533_53394

theorem opposites_and_reciprocals_problem 
  (a b x y : ℝ) 
  (h1 : a + b = 0)      -- a and b are opposites
  (h2 : x * y = 1)      -- x and y are reciprocals
  : 5 * |a + b| - 5 * x * y = -5 := by
  sorry

end NUMINAMATH_CALUDE_opposites_and_reciprocals_problem_l533_53394


namespace NUMINAMATH_CALUDE_players_per_group_l533_53341

theorem players_per_group (new_players returning_players total_groups : ℕ) : 
  new_players = 48 → 
  returning_players = 6 → 
  total_groups = 9 → 
  (new_players + returning_players) / total_groups = 6 :=
by sorry

end NUMINAMATH_CALUDE_players_per_group_l533_53341


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l533_53330

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x^2 + x + 1 < 0) ↔ (∃ x : ℝ, x^2 + x + 1 ≥ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l533_53330


namespace NUMINAMATH_CALUDE_xy_equals_twelve_l533_53339

theorem xy_equals_twelve (x y : ℝ) (h : x * (x + y) = x^2 + 12) : x * y = 12 := by
  sorry

end NUMINAMATH_CALUDE_xy_equals_twelve_l533_53339


namespace NUMINAMATH_CALUDE_problem_statement_l533_53326

theorem problem_statement (a b : ℕ) (h_a : a ≠ 0) (h_b : b ≠ 0) 
  (h : ∀ n : ℕ, n ≥ 1 → (2^n * b + 1) ∣ (a^(2^n) - 1)) : a = 1 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l533_53326


namespace NUMINAMATH_CALUDE_shanghai_masters_matches_l533_53335

/-- Represents the tournament structure described in the problem -/
structure Tournament :=
  (num_players : Nat)
  (num_groups : Nat)
  (players_per_group : Nat)
  (advancing_per_group : Nat)

/-- Calculates the number of matches in a round-robin tournament -/
def round_robin_matches (n : Nat) : Nat :=
  n * (n - 1) / 2

/-- Calculates the total number of matches in the tournament -/
def total_matches (t : Tournament) : Nat :=
  let group_matches := t.num_groups * round_robin_matches t.players_per_group
  let elimination_matches := t.num_groups * t.advancing_per_group / 2
  let final_matches := 2
  group_matches + elimination_matches + final_matches

/-- Theorem stating that the total number of matches in the given tournament format is 16 -/
theorem shanghai_masters_matches :
  ∃ t : Tournament, t.num_players = 8 ∧ t.num_groups = 2 ∧ t.players_per_group = 4 ∧ t.advancing_per_group = 2 ∧ total_matches t = 16 :=
by
  sorry

end NUMINAMATH_CALUDE_shanghai_masters_matches_l533_53335


namespace NUMINAMATH_CALUDE_first_pipe_fill_time_l533_53340

/-- Given two pipes that can fill a tank, an outlet pipe that can empty it, 
    and the time it takes to fill the tank when all pipes are open, 
    this theorem proves the time it takes for the first pipe to fill the tank. -/
theorem first_pipe_fill_time (t : ℝ) (h1 : t > 0) 
  (h2 : 1/t + 1/30 - 1/45 = 1/15) : t = 18 := by
  sorry

end NUMINAMATH_CALUDE_first_pipe_fill_time_l533_53340


namespace NUMINAMATH_CALUDE_range_of_a_l533_53329

theorem range_of_a (a : ℝ) : (¬ ∃ x, x < 2023 ∧ x > a) → a ≥ 2023 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l533_53329


namespace NUMINAMATH_CALUDE_polynomial_correction_l533_53347

/-- If a polynomial P(x) satisfies P(x) - 3x² = x² - 2x + 1, 
    then -3x² * P(x) = -12x⁴ + 6x³ - 3x² -/
theorem polynomial_correction (x : ℝ) (P : ℝ → ℝ) 
  (h : P x - 3 * x^2 = x^2 - 2*x + 1) : 
  -3 * x^2 * P x = -12 * x^4 + 6 * x^3 - 3 * x^2 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_correction_l533_53347


namespace NUMINAMATH_CALUDE_hyperbolas_same_asymptotes_implies_M_64_l533_53360

-- Define the two hyperbolas
def hyperbola1 (x y : ℝ) : Prop := x^2 / 16 - y^2 / 25 = 1
def hyperbola2 (x y M : ℝ) : Prop := y^2 / 100 - x^2 / M = 1

-- Define the asymptotes of the hyperbolas
def asymptote1 (x y : ℝ) : Prop := y = (5/4) * x ∨ y = -(5/4) * x
def asymptote2 (x y M : ℝ) : Prop := y = (10/Real.sqrt M) * x ∨ y = -(10/Real.sqrt M) * x

-- Theorem statement
theorem hyperbolas_same_asymptotes_implies_M_64 :
  ∀ M : ℝ, (∀ x y : ℝ, asymptote1 x y ↔ asymptote2 x y M) → M = 64 := by
  sorry

end NUMINAMATH_CALUDE_hyperbolas_same_asymptotes_implies_M_64_l533_53360
