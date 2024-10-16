import Mathlib

namespace NUMINAMATH_CALUDE_beth_sold_coins_l2409_240947

theorem beth_sold_coins (initial_coins carl_gift kept_coins : ℕ) 
  (h1 : initial_coins = 250)
  (h2 : carl_gift = 75)
  (h3 : kept_coins = 135) :
  initial_coins + carl_gift - kept_coins = 190 :=
by
  sorry

end NUMINAMATH_CALUDE_beth_sold_coins_l2409_240947


namespace NUMINAMATH_CALUDE_expression_takes_many_values_l2409_240945

theorem expression_takes_many_values :
  ∀ (x : ℝ), x ≠ -2 → x ≠ 3 →
  ∃ (y : ℝ), y ≠ x ∧
    (3 + 6 / (3 - x)) ≠ (3 + 6 / (3 - y)) :=
by sorry

end NUMINAMATH_CALUDE_expression_takes_many_values_l2409_240945


namespace NUMINAMATH_CALUDE_apples_needed_proof_l2409_240902

/-- The number of additional apples Tessa needs to make a pie -/
def additional_apples_needed (initial : ℕ) (received : ℕ) (required : ℕ) : ℕ :=
  required - (initial + received)

/-- Theorem: Given Tessa's initial apples, apples received from Anita, and apples needed for a pie,
    the number of additional apples needed is equal to the apples required for a pie
    minus the sum of initial apples and received apples. -/
theorem apples_needed_proof (initial : ℕ) (received : ℕ) (required : ℕ)
    (h1 : initial = 4)
    (h2 : received = 5)
    (h3 : required = 10) :
  additional_apples_needed initial received required = 1 := by
  sorry

end NUMINAMATH_CALUDE_apples_needed_proof_l2409_240902


namespace NUMINAMATH_CALUDE_max_number_in_sample_l2409_240906

/-- Represents a systematic sample from a range of products -/
structure SystematicSample where
  total_products : ℕ
  sample_size : ℕ
  start : ℕ
  interval : ℕ

/-- Creates a systematic sample given total products and sample size -/
def create_systematic_sample (total_products sample_size : ℕ) : SystematicSample :=
  { total_products := total_products
  , sample_size := sample_size
  , start := 0  -- Assuming start is 0 for simplicity
  , interval := total_products / sample_size
  }

/-- Checks if a number is in the systematic sample -/
def is_in_sample (sample : SystematicSample) (n : ℕ) : Prop :=
  ∃ k, 0 ≤ k ∧ k < sample.sample_size ∧ n = sample.start + k * sample.interval

/-- Gets the maximum number in the systematic sample -/
def max_in_sample (sample : SystematicSample) : ℕ :=
  sample.start + (sample.sample_size - 1) * sample.interval

/-- Theorem: If 58 is in a systematic sample of size 10 from 80 products, 
    then the maximum number in the sample is 74 -/
theorem max_number_in_sample :
  let sample := create_systematic_sample 80 10
  is_in_sample sample 58 → max_in_sample sample = 74 := by
  sorry


end NUMINAMATH_CALUDE_max_number_in_sample_l2409_240906


namespace NUMINAMATH_CALUDE_synthetic_method_placement_l2409_240941

-- Define the structure elements
inductive KnowledgeElement
  | RationalReasoning
  | DeductiveReasoning
  | DirectProof
  | IndirectProof
  | SyntheticMethod

-- Define the relation "is a type of"
def isTypeOf : KnowledgeElement → KnowledgeElement → Prop :=
  fun x y => match x, y with
    | KnowledgeElement.SyntheticMethod, KnowledgeElement.DirectProof => True
    | _, _ => False

-- Define the relation "should be placed under"
def shouldBePlacedUnder : KnowledgeElement → KnowledgeElement → Prop :=
  fun x y => isTypeOf x y

-- Theorem statement
theorem synthetic_method_placement :
  shouldBePlacedUnder KnowledgeElement.SyntheticMethod KnowledgeElement.DirectProof :=
by sorry

end NUMINAMATH_CALUDE_synthetic_method_placement_l2409_240941


namespace NUMINAMATH_CALUDE_interval_equivalence_l2409_240904

theorem interval_equivalence (a : ℝ) : -1 < a ∧ a < 1 ↔ |a| < 1 := by
  sorry

end NUMINAMATH_CALUDE_interval_equivalence_l2409_240904


namespace NUMINAMATH_CALUDE_mr_mcpherson_contribution_l2409_240944

/-- Calculates the amount Mr. McPherson needs to raise for rent -/
theorem mr_mcpherson_contribution (total_rent : ℝ) (mrs_mcpherson_percentage : ℝ) :
  total_rent = 1200 →
  mrs_mcpherson_percentage = 30 →
  total_rent - (mrs_mcpherson_percentage / 100 * total_rent) = 840 := by
sorry

end NUMINAMATH_CALUDE_mr_mcpherson_contribution_l2409_240944


namespace NUMINAMATH_CALUDE_symmetric_point_y_axis_l2409_240959

/-- Given a point M(-5, 2), its symmetric point with respect to the y-axis has coordinates (5, 2) -/
theorem symmetric_point_y_axis :
  let M : ℝ × ℝ := (-5, 2)
  let symmetric_point (p : ℝ × ℝ) : ℝ × ℝ := (-p.1, p.2)
  symmetric_point M = (5, 2) := by
sorry

end NUMINAMATH_CALUDE_symmetric_point_y_axis_l2409_240959


namespace NUMINAMATH_CALUDE_intersection_implies_m_value_l2409_240989

def A (m : ℝ) : Set ℝ := {m - 1, -3}
def B (m : ℝ) : Set ℝ := {2*m - 1, m - 3}

theorem intersection_implies_m_value :
  ∀ m : ℝ, A m ∩ B m = {-3} → m = -1 := by
  sorry

end NUMINAMATH_CALUDE_intersection_implies_m_value_l2409_240989


namespace NUMINAMATH_CALUDE_total_marbles_count_l2409_240948

/-- The total number of marbles Mary and Joan have -/
def total_marbles : ℕ :=
  let mary_yellow := 9
  let mary_blue := 7
  let mary_green := 6
  let joan_yellow := 3
  let joan_blue := 5
  let joan_green := 4
  mary_yellow + mary_blue + mary_green + joan_yellow + joan_blue + joan_green

theorem total_marbles_count : total_marbles = 34 := by
  sorry

end NUMINAMATH_CALUDE_total_marbles_count_l2409_240948


namespace NUMINAMATH_CALUDE_cube_in_pyramid_volume_l2409_240912

/-- A pyramid with a square base and isosceles triangular lateral faces -/
structure Pyramid where
  base_side : ℝ
  lateral_height : ℝ

/-- A cube placed inside the pyramid -/
structure InsideCube where
  side_length : ℝ

/-- The volume of a cube -/
def cube_volume (c : InsideCube) : ℝ := c.side_length ^ 3

theorem cube_in_pyramid_volume 
  (p : Pyramid) 
  (c : InsideCube) 
  (h1 : p.base_side = 2) 
  (h2 : p.lateral_height = 4) 
  (h3 : c.side_length * 2 = p.lateral_height) : 
  cube_volume c = 8 := by
  sorry

#check cube_in_pyramid_volume

end NUMINAMATH_CALUDE_cube_in_pyramid_volume_l2409_240912


namespace NUMINAMATH_CALUDE_marble_count_l2409_240926

/-- Given a bag of marbles with red, blue, and yellow marbles in the ratio 2:3:4,
    and 36 yellow marbles, prove that there are 81 marbles in total. -/
theorem marble_count (red blue yellow total : ℕ) : 
  red + blue + yellow = total →
  red = 2 * n ∧ blue = 3 * n ∧ yellow = 4 * n →
  yellow = 36 →
  total = 81 :=
by
  sorry

#check marble_count

end NUMINAMATH_CALUDE_marble_count_l2409_240926


namespace NUMINAMATH_CALUDE_toucan_count_l2409_240966

theorem toucan_count (initial_toucans : ℕ) (joining_toucans : ℕ) : 
  initial_toucans = 2 → joining_toucans = 1 → initial_toucans + joining_toucans = 3 := by
  sorry

end NUMINAMATH_CALUDE_toucan_count_l2409_240966


namespace NUMINAMATH_CALUDE_smallest_n_congruence_two_satisfies_congruence_smallest_n_is_two_l2409_240981

theorem smallest_n_congruence (n : ℕ) : n > 0 ∧ 721 * n ≡ 1137 * n [ZMOD 30] → n ≥ 2 :=
sorry

theorem two_satisfies_congruence : 721 * 2 ≡ 1137 * 2 [ZMOD 30] :=
sorry

theorem smallest_n_is_two : 
  ∃ (n : ℕ), n > 0 ∧ 721 * n ≡ 1137 * n [ZMOD 30] ∧ 
  ∀ (m : ℕ), m > 0 ∧ 721 * m ≡ 1137 * m [ZMOD 30] → n ≤ m :=
sorry

end NUMINAMATH_CALUDE_smallest_n_congruence_two_satisfies_congruence_smallest_n_is_two_l2409_240981


namespace NUMINAMATH_CALUDE_pet_parasites_l2409_240968

theorem pet_parasites (dog_burrs : ℕ) : ℕ :=
  let dog_ticks := 6 * dog_burrs
  let dog_fleas := 3 * dog_ticks
  let cat_burrs := 2 * dog_burrs
  let cat_ticks := dog_ticks / 3
  let cat_fleas := 4 * cat_ticks
  let total_parasites := dog_burrs + dog_ticks + dog_fleas + cat_burrs + cat_ticks + cat_fleas
  
  by
  -- Assuming dog_burrs = 12
  have h : dog_burrs = 12 := by sorry
  -- Proof goes here
  sorry

-- The theorem states that given the number of burrs on the dog (which we know is 12),
-- we can calculate the total number of parasites on both pets.
-- The proof would show that this total is indeed 444.

end NUMINAMATH_CALUDE_pet_parasites_l2409_240968


namespace NUMINAMATH_CALUDE_jacket_selling_price_l2409_240942

/-- Calculates the total selling price of a jacket given the original price,
    discount rate, tax rate, and processing fee. -/
def total_selling_price (original_price discount_rate tax_rate processing_fee : ℝ) : ℝ :=
  let discounted_price := original_price * (1 - discount_rate)
  let price_with_tax := discounted_price * (1 + tax_rate)
  price_with_tax + processing_fee

/-- Theorem stating that the total selling price of the jacket is $95.72 -/
theorem jacket_selling_price :
  total_selling_price 120 0.30 0.08 5 = 95.72 := by
  sorry

#eval total_selling_price 120 0.30 0.08 5

end NUMINAMATH_CALUDE_jacket_selling_price_l2409_240942


namespace NUMINAMATH_CALUDE_integral_x_squared_plus_sin_x_l2409_240929

theorem integral_x_squared_plus_sin_x : ∫ x in (-1)..1, (x^2 + Real.sin x) = 2/3 := by sorry

end NUMINAMATH_CALUDE_integral_x_squared_plus_sin_x_l2409_240929


namespace NUMINAMATH_CALUDE_book_profit_percentage_l2409_240994

/-- Calculates the profit percentage on the cost price for a book sale --/
theorem book_profit_percentage 
  (cost_price : ℝ) 
  (marked_price : ℝ) 
  (discount_rate : ℝ) 
  (h1 : cost_price = 47.50)
  (h2 : marked_price = 69.85)
  (h3 : discount_rate = 0.15) : 
  ∃ (profit_percentage : ℝ), 
    abs (profit_percentage - 24.99) < 0.01 ∧ 
    profit_percentage = (marked_price * (1 - discount_rate) - cost_price) / cost_price * 100 := by
  sorry


end NUMINAMATH_CALUDE_book_profit_percentage_l2409_240994


namespace NUMINAMATH_CALUDE_circle_center_correct_l2409_240970

/-- The equation of a circle in the form ax^2 + bx + cy^2 + dy + e = 0 -/
structure CircleEquation where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ
  e : ℝ

/-- The center of a circle -/
structure CircleCenter where
  x : ℝ
  y : ℝ

/-- Given a circle equation, find its center -/
def findCircleCenter (eq : CircleEquation) : CircleCenter :=
  sorry

theorem circle_center_correct (eq : CircleEquation) :
  eq.a = 1 ∧ eq.b = -10 ∧ eq.c = 1 ∧ eq.d = -4 ∧ eq.e = -20 →
  let center := findCircleCenter eq
  center.x = 5 ∧ center.y = 2 :=
sorry

end NUMINAMATH_CALUDE_circle_center_correct_l2409_240970


namespace NUMINAMATH_CALUDE_decimal_to_fraction_l2409_240996

theorem decimal_to_fraction :
  (0.36 : ℚ) = 9 / 25 := by sorry

end NUMINAMATH_CALUDE_decimal_to_fraction_l2409_240996


namespace NUMINAMATH_CALUDE_probability_of_winning_pair_l2409_240976

def deck_size : ℕ := 10
def red_cards : ℕ := 5
def green_cards : ℕ := 5
def num_letters : ℕ := 5

def winning_pair_count : ℕ := num_letters + 2 * (red_cards.choose 2)

theorem probability_of_winning_pair :
  (winning_pair_count : ℚ) / (deck_size.choose 2) = 5 / 9 := by sorry

end NUMINAMATH_CALUDE_probability_of_winning_pair_l2409_240976


namespace NUMINAMATH_CALUDE_cost_price_of_ball_l2409_240950

/-- The cost price of a single ball -/
def cost_price : ℚ := 200/3

/-- The number of balls sold -/
def num_balls : ℕ := 17

/-- The selling price after discount -/
def selling_price_after_discount : ℚ := 720

/-- The discount rate -/
def discount_rate : ℚ := 1/10

/-- The selling price before discount -/
def selling_price_before_discount : ℚ := selling_price_after_discount / (1 - discount_rate)

/-- The theorem stating the cost price of each ball -/
theorem cost_price_of_ball :
  (num_balls * cost_price - selling_price_before_discount = 5 * cost_price) ∧
  (selling_price_after_discount = selling_price_before_discount * (1 - discount_rate)) ∧
  (cost_price = 200/3) :=
sorry

end NUMINAMATH_CALUDE_cost_price_of_ball_l2409_240950


namespace NUMINAMATH_CALUDE_correct_average_l2409_240935

theorem correct_average (n : ℕ) (initial_avg : ℚ) (incorrect_num correct_num : ℚ) :
  n = 10 ∧ initial_avg = 46 ∧ incorrect_num = 25 ∧ correct_num = 75 →
  (n * initial_avg + (correct_num - incorrect_num)) / n = 51 :=
by sorry

end NUMINAMATH_CALUDE_correct_average_l2409_240935


namespace NUMINAMATH_CALUDE_quadratic_inequality_implies_m_range_l2409_240991

theorem quadratic_inequality_implies_m_range (m : ℝ) : 
  (∀ x : ℝ, x^2 - 2*x - m > 0) → m < -1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_implies_m_range_l2409_240991


namespace NUMINAMATH_CALUDE_cos_alpha_for_point_in_third_quadrant_l2409_240921

theorem cos_alpha_for_point_in_third_quadrant (a : ℝ) (α : ℝ) :
  a < 0 →
  ∃ (P : ℝ × ℝ), P = (3*a, 4*a) ∧ 
  (∃ (r : ℝ), r > 0 ∧ P = (r * Real.cos α, r * Real.sin α)) →
  Real.cos α = -3/5 := by
  sorry

end NUMINAMATH_CALUDE_cos_alpha_for_point_in_third_quadrant_l2409_240921


namespace NUMINAMATH_CALUDE_line_classification_l2409_240913

-- Define the coordinate plane
def CoordinatePlane : Type := ℝ × ℝ

-- Define an integer point
def IntegerPoint (p : CoordinatePlane) : Prop :=
  ∃ (x y : ℤ), p = (↑x, ↑y)

-- Define a line on the coordinate plane
def Line : Type := CoordinatePlane → Prop

-- Define set I as the set of all lines
def I : Set Line := Set.univ

-- Define set M as the set of lines passing through exactly one integer point
def M : Set Line :=
  {l : Line | ∃! (p : CoordinatePlane), IntegerPoint p ∧ l p}

-- Define set N as the set of lines passing through no integer points
def N : Set Line :=
  {l : Line | ∀ (p : CoordinatePlane), l p → ¬IntegerPoint p}

-- Define set P as the set of lines passing through infinitely many integer points
def P : Set Line :=
  {l : Line | ∀ (n : ℕ), ∃ (S : Finset CoordinatePlane),
    Finset.card S = n ∧ (∀ (p : CoordinatePlane), p ∈ S → IntegerPoint p ∧ l p)}

theorem line_classification :
  (M ∪ N ∪ P = I) ∧ (N ≠ ∅) ∧ (M ≠ ∅) ∧ (P ≠ ∅) := by sorry

end NUMINAMATH_CALUDE_line_classification_l2409_240913


namespace NUMINAMATH_CALUDE_factor_calculation_l2409_240924

theorem factor_calculation (x : ℝ) (h : x = 36) : 
  ∃ f : ℝ, ((x + 10) * f / 2) - 2 = 88 / 2 ∧ f = 2 := by
  sorry

end NUMINAMATH_CALUDE_factor_calculation_l2409_240924


namespace NUMINAMATH_CALUDE_exists_valid_cylinder_arrangement_l2409_240972

/-- Represents a straight circular cylinder in 3D space -/
structure Cylinder where
  center : ℝ × ℝ × ℝ
  radius : ℝ
  height : ℝ

/-- Checks if two cylinders have a common boundary point -/
def havesCommonPoint (c1 c2 : Cylinder) : Prop := sorry

/-- Represents an arrangement of six cylinders -/
def CylinderArrangement := Fin 6 → Cylinder

/-- Checks if a given arrangement satisfies the condition that each cylinder
    has a common point with every other cylinder -/
def isValidArrangement (arr : CylinderArrangement) : Prop :=
  ∀ i j, i ≠ j → havesCommonPoint (arr i) (arr j)

/-- The main theorem stating that there exists a valid arrangement of six cylinders -/
theorem exists_valid_cylinder_arrangement :
  ∃ (arr : CylinderArrangement), isValidArrangement arr := by sorry

end NUMINAMATH_CALUDE_exists_valid_cylinder_arrangement_l2409_240972


namespace NUMINAMATH_CALUDE_morgans_mean_score_l2409_240905

def scores : List ℝ := [78, 82, 90, 95, 98, 102, 105]
def alex_count : ℕ := 4
def morgan_count : ℕ := 3
def alex_mean : ℝ := 91.5

theorem morgans_mean_score (h1 : scores.length = alex_count + morgan_count)
                            (h2 : alex_count * alex_mean = (scores.take alex_count).sum) :
  (scores.drop alex_count).sum / morgan_count = 94.67 := by
  sorry

end NUMINAMATH_CALUDE_morgans_mean_score_l2409_240905


namespace NUMINAMATH_CALUDE_tangent_line_at_e_l2409_240953

noncomputable def f (x : ℝ) := x * Real.log x

theorem tangent_line_at_e :
  let x₀ : ℝ := Real.exp 1
  let y₀ : ℝ := f x₀
  let m : ℝ := Real.exp 1 * (1 / Real.exp 1) + Real.log (Real.exp 1)
  (λ x y => y = m * (x - x₀) + y₀) = (λ x y => y = 2 * x - Real.exp 1) :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_at_e_l2409_240953


namespace NUMINAMATH_CALUDE_tileIV_in_rectangle_C_l2409_240907

-- Define the tile sides
inductive Side
| Top
| Right
| Bottom
| Left

-- Define the tiles
structure Tile :=
  (id : Nat)
  (top : Nat)
  (right : Nat)
  (bottom : Nat)
  (left : Nat)

-- Define the rectangles
inductive Rectangle
| A
| B
| C
| D

-- Define the placement of tiles
def Placement := Tile → Rectangle

-- Define the adjacency relation between rectangles
def Adjacent : Rectangle → Rectangle → Prop := sorry

-- Define the matching condition for adjacent tiles
def MatchingSides (t1 t2 : Tile) (s1 s2 : Side) : Prop := sorry

-- Define the validity of a placement
def ValidPlacement (p : Placement) : Prop := sorry

-- Define the tiles from the problem
def tileI : Tile := ⟨1, 6, 8, 3, 7⟩
def tileII : Tile := ⟨2, 7, 6, 2, 9⟩
def tileIII : Tile := ⟨3, 5, 1, 9, 0⟩
def tileIV : Tile := ⟨4, 0, 9, 4, 5⟩

-- Theorem statement
theorem tileIV_in_rectangle_C :
  ∀ (p : Placement), ValidPlacement p → p tileIV = Rectangle.C := by
  sorry

end NUMINAMATH_CALUDE_tileIV_in_rectangle_C_l2409_240907


namespace NUMINAMATH_CALUDE_one_child_truthful_l2409_240908

structure Child where
  name : String
  truthful : Bool

def grisha_claim (masha sasha natasha : Child) : Prop :=
  masha.truthful ∧ sasha.truthful ∧ natasha.truthful

def contradictions_exist (masha sasha natasha : Child) : Prop :=
  ¬(masha.truthful ∧ sasha.truthful ∧ natasha.truthful)

theorem one_child_truthful (masha sasha natasha : Child) :
  grisha_claim masha sasha natasha →
  contradictions_exist masha sasha natasha →
  ∃! c : Child, c ∈ [masha, sasha, natasha] ∧ c.truthful :=
by
  sorry

#check one_child_truthful

end NUMINAMATH_CALUDE_one_child_truthful_l2409_240908


namespace NUMINAMATH_CALUDE_hearing_aid_cost_proof_l2409_240998

/-- The cost of a single hearing aid -/
def hearing_aid_cost : ℝ := 2500

/-- The insurance coverage percentage -/
def insurance_coverage : ℝ := 0.80

/-- The amount John pays for both hearing aids -/
def john_payment : ℝ := 1000

/-- Theorem stating that the cost of each hearing aid is $2500 -/
theorem hearing_aid_cost_proof : 
  (1 - insurance_coverage) * (2 * hearing_aid_cost) = john_payment := by
  sorry

end NUMINAMATH_CALUDE_hearing_aid_cost_proof_l2409_240998


namespace NUMINAMATH_CALUDE_candy_jar_problem_l2409_240983

theorem candy_jar_problem (total : ℕ) (blue : ℕ) (red : ℕ) : 
  total = 3409 → 
  blue = 3264 → 
  total = red + blue → 
  red = 145 := by
sorry

end NUMINAMATH_CALUDE_candy_jar_problem_l2409_240983


namespace NUMINAMATH_CALUDE_yellow_balls_count_l2409_240943

theorem yellow_balls_count (red_balls : ℕ) (probability_red : ℚ) (yellow_balls : ℕ) : 
  red_balls = 10 →
  probability_red = 2/5 →
  (red_balls : ℚ) / ((red_balls : ℚ) + (yellow_balls : ℚ)) = probability_red →
  yellow_balls = 15 := by
sorry

end NUMINAMATH_CALUDE_yellow_balls_count_l2409_240943


namespace NUMINAMATH_CALUDE_inequalities_hold_l2409_240930

theorem inequalities_hold (a b c : ℝ) (h1 : a < 0) (h2 : a < b) (h3 : b < c) :
  (a^2 * b < b^2 * c) ∧ (a^2 * c < b^2 * c) ∧ (a^2 * b < a^2 * c) := by
  sorry

end NUMINAMATH_CALUDE_inequalities_hold_l2409_240930


namespace NUMINAMATH_CALUDE_ratio_12min_to_1hour_is_1_to_5_l2409_240955

/-- The ratio of 12 minutes to 1 hour -/
def ratio_12min_to_1hour : ℚ × ℚ :=
  sorry

/-- One hour in minutes -/
def minutes_per_hour : ℕ := 60

theorem ratio_12min_to_1hour_is_1_to_5 :
  ratio_12min_to_1hour = (1, 5) := by
  sorry

end NUMINAMATH_CALUDE_ratio_12min_to_1hour_is_1_to_5_l2409_240955


namespace NUMINAMATH_CALUDE_aloh3_molecular_weight_l2409_240967

/-- The molecular weight of a compound given its composition and atomic weights -/
def molecularWeight (alWeight oWeight hWeight : ℝ) (moles : ℝ) : ℝ :=
  moles * (alWeight + 3 * oWeight + 3 * hWeight)

/-- Theorem stating the molecular weight of 7 moles of Al(OH)3 -/
theorem aloh3_molecular_weight :
  molecularWeight 26.98 16.00 1.01 7 = 546.07 := by
  sorry

end NUMINAMATH_CALUDE_aloh3_molecular_weight_l2409_240967


namespace NUMINAMATH_CALUDE_travel_agency_comparison_l2409_240974

/-- Represents the fee calculation for a travel agency. -/
structure TravelAgency where
  parentDiscount : ℝ  -- Discount for parents (1 means no discount)
  studentDiscount : ℝ  -- Discount for students
  basePrice : ℝ        -- Base price per person

/-- Calculate the total fee for a travel agency given the number of students. -/
def calculateFee (agency : TravelAgency) (numStudents : ℝ) : ℝ :=
  agency.basePrice * (2 * agency.parentDiscount + numStudents * agency.studentDiscount)

/-- Travel Agency A with full price for parents and 70% for students. -/
def agencyA : TravelAgency :=
  { parentDiscount := 1
  , studentDiscount := 0.7
  , basePrice := 500 }

/-- Travel Agency B with 80% price for both parents and students. -/
def agencyB : TravelAgency :=
  { parentDiscount := 0.8
  , studentDiscount := 0.8
  , basePrice := 500 }

theorem travel_agency_comparison :
  ∀ x : ℝ,
    (calculateFee agencyA x = 350 * x + 1000) ∧
    (calculateFee agencyB x = 400 * x + 800) ∧
    (0 < x ∧ x < 4 → calculateFee agencyB x < calculateFee agencyA x) ∧
    (x = 4 → calculateFee agencyA x = calculateFee agencyB x) ∧
    (x > 4 → calculateFee agencyA x < calculateFee agencyB x) :=
by sorry

end NUMINAMATH_CALUDE_travel_agency_comparison_l2409_240974


namespace NUMINAMATH_CALUDE_average_speed_calculation_l2409_240934

theorem average_speed_calculation (total_distance : ℝ) (distance1 : ℝ) (speed1 : ℝ) (distance2 : ℝ) (speed2 : ℝ) :
  total_distance = 80 ∧
  distance1 = 30 ∧
  speed1 = 30 ∧
  distance2 = 50 ∧
  speed2 = 50 →
  (total_distance / (distance1 / speed1 + distance2 / speed2)) = 40 := by
  sorry

end NUMINAMATH_CALUDE_average_speed_calculation_l2409_240934


namespace NUMINAMATH_CALUDE_mixed_oil_rate_l2409_240909

/-- The rate of mixed oil per litre given specific quantities and prices of three types of oil -/
theorem mixed_oil_rate (quantity1 quantity2 quantity3 : ℚ) (price1 price2 price3 : ℚ) : 
  quantity1 = 12 ∧ quantity2 = 8 ∧ quantity3 = 4 ∧
  price1 = 55 ∧ price2 = 70 ∧ price3 = 82 →
  (quantity1 * price1 + quantity2 * price2 + quantity3 * price3) / (quantity1 + quantity2 + quantity3) = 64.5 := by
  sorry

#check mixed_oil_rate

end NUMINAMATH_CALUDE_mixed_oil_rate_l2409_240909


namespace NUMINAMATH_CALUDE_solutions_for_x_l2409_240928

theorem solutions_for_x : ∃ (x₁ x₂ x₃ : ℝ),
  ((x₁ + 1)^2 = 36 ∨ (x₁ + 10)^3 = -27) ∧
  ((x₂ + 1)^2 = 36 ∨ (x₂ + 10)^3 = -27) ∧
  ((x₃ + 1)^2 = 36 ∨ (x₃ + 10)^3 = -27) ∧
  x₁ = 5 ∧ x₂ = -7 ∧ x₃ = -13 :=
by sorry

end NUMINAMATH_CALUDE_solutions_for_x_l2409_240928


namespace NUMINAMATH_CALUDE_contrapositive_equivalence_l2409_240949

theorem contrapositive_equivalence :
  (¬(a^2 = 1) → ¬(a = -1)) ↔ (a = -1 → a^2 = 1) :=
by sorry

end NUMINAMATH_CALUDE_contrapositive_equivalence_l2409_240949


namespace NUMINAMATH_CALUDE_quadratic_rewrite_l2409_240961

theorem quadratic_rewrite (g h j : ℤ) :
  (∀ x : ℝ, 4 * x^2 - 16 * x - 21 = (g * x + h)^2 + j) →
  g * h = -8 := by
sorry

end NUMINAMATH_CALUDE_quadratic_rewrite_l2409_240961


namespace NUMINAMATH_CALUDE_inequality_theorem_equality_condition_l2409_240903

theorem inequality_theorem (a b c : ℝ) :
  Real.sqrt (a^2 + a*b + b^2) + Real.sqrt (a^2 + a*c + c^2) ≥ Real.sqrt (3*a^2 + (a+b+c)^2) :=
sorry

theorem equality_condition (a b c : ℝ) :
  (Real.sqrt (a^2 + a*b + b^2) + Real.sqrt (a^2 + a*c + c^2) = Real.sqrt (3*a^2 + (a+b+c)^2)) ↔
  (b = c ∨ (a = 0 ∧ b*c ≥ 0)) :=
sorry

end NUMINAMATH_CALUDE_inequality_theorem_equality_condition_l2409_240903


namespace NUMINAMATH_CALUDE_bus_ride_cost_l2409_240982

-- Define the cost of bus and train rides
def bus_cost : ℝ := sorry
def train_cost : ℝ := sorry

-- State the theorem
theorem bus_ride_cost :
  (train_cost = bus_cost + 6.85) →
  (train_cost + bus_cost = 9.65) →
  (bus_cost = 1.40) := by
  sorry

end NUMINAMATH_CALUDE_bus_ride_cost_l2409_240982


namespace NUMINAMATH_CALUDE_unique_pen_distribution_l2409_240932

/-- Represents a distribution of pens among students -/
structure PenDistribution where
  num_students : ℕ
  pens_per_student : ℕ → ℕ
  total_pens : ℕ

/-- The condition that among any four pens, at least two belong to the same person -/
def four_pens_condition (d : PenDistribution) : Prop :=
  ∀ (s : Finset ℕ), s.card = 4 → ∃ i ∈ s, d.pens_per_student i ≥ 2

/-- The condition that among any five pens, no more than three belong to the same person -/
def five_pens_condition (d : PenDistribution) : Prop :=
  ∀ (s : Finset ℕ), s.card = 5 → ∀ i ∈ s, d.pens_per_student i ≤ 3

/-- The theorem stating the unique distribution satisfying the given conditions -/
theorem unique_pen_distribution :
  ∀ (d : PenDistribution),
    d.total_pens = 9 →
    four_pens_condition d →
    five_pens_condition d →
    d.num_students = 3 ∧ (∀ i, i < d.num_students → d.pens_per_student i = 3) :=
by sorry

end NUMINAMATH_CALUDE_unique_pen_distribution_l2409_240932


namespace NUMINAMATH_CALUDE_net_gain_calculation_l2409_240978

def initial_value : ℝ := 500000

def first_sale_profit : ℝ := 0.15
def first_buyback_loss : ℝ := 0.05
def second_sale_profit : ℝ := 0.10
def final_buyback_loss : ℝ := 0.10

def first_sale (value : ℝ) : ℝ := value * (1 + first_sale_profit)
def first_buyback (value : ℝ) : ℝ := value * (1 - first_buyback_loss)
def second_sale (value : ℝ) : ℝ := value * (1 + second_sale_profit)
def final_buyback (value : ℝ) : ℝ := value * (1 - final_buyback_loss)

def total_sales (v : ℝ) : ℝ := first_sale v + second_sale (first_buyback (first_sale v))
def total_purchases (v : ℝ) : ℝ := first_buyback (first_sale v) + final_buyback (second_sale (first_buyback (first_sale v)))

theorem net_gain_calculation (v : ℝ) : 
  total_sales v - total_purchases v = 88837.50 :=
by sorry

end NUMINAMATH_CALUDE_net_gain_calculation_l2409_240978


namespace NUMINAMATH_CALUDE_subset_of_all_implies_zero_l2409_240923

theorem subset_of_all_implies_zero (a : ℝ) :
  (∀ S : Set ℝ, {x : ℝ | a * x = 1} ⊆ S) → a = 0 := by
  sorry

end NUMINAMATH_CALUDE_subset_of_all_implies_zero_l2409_240923


namespace NUMINAMATH_CALUDE_min_value_at_3_l2409_240900

def S (n : ℕ) : ℤ := n^2 - 10*n

def a (n : ℕ) : ℤ :=
  if n = 1 then S 1
  else S n - S (n-1)

def na (n : ℕ) : ℤ := n * (a n)

theorem min_value_at_3 :
  ∀ k : ℕ, k ≥ 1 → na 3 ≤ na k :=
by sorry

end NUMINAMATH_CALUDE_min_value_at_3_l2409_240900


namespace NUMINAMATH_CALUDE_two_roots_implication_l2409_240975

/-- If a quadratic trinomial ax^2 + bx + c has two roots, 
    then the trinomial 3ax^2 + 2(a + b)x + (b + c) also has two roots. -/
theorem two_roots_implication (a b c : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ a * x^2 + b * x + c = 0 ∧ a * y^2 + b * y + c = 0) →
  (∃ u v : ℝ, u ≠ v ∧ 3 * a * u^2 + 2 * (a + b) * u + (b + c) = 0 ∧ 
                    3 * a * v^2 + 2 * (a + b) * v + (b + c) = 0) :=
by sorry

end NUMINAMATH_CALUDE_two_roots_implication_l2409_240975


namespace NUMINAMATH_CALUDE_sequence_divisibility_l2409_240933

theorem sequence_divisibility (n : ℕ) : 
  (∃ k, k > 0 ∧ k * (k + 1) ≤ 14520 ∧ 120 ∣ (k * (k + 1))) ↔ 
  (∃ m, m ≥ 1 ∧ m ≤ 8 ∧ 120 ∣ (n * (n + 1))) :=
by sorry

end NUMINAMATH_CALUDE_sequence_divisibility_l2409_240933


namespace NUMINAMATH_CALUDE_thursday_dogs_l2409_240980

/-- The number of dogs Harry walks on Monday, Wednesday, and Friday -/
def dogs_mon_wed_fri : ℕ := 7

/-- The number of dogs Harry walks on Tuesday -/
def dogs_tuesday : ℕ := 12

/-- The amount Harry is paid per dog -/
def pay_per_dog : ℕ := 5

/-- Harry's total earnings for the week -/
def total_earnings : ℕ := 210

/-- The number of days Harry walks 7 dogs -/
def days_with_seven_dogs : ℕ := 3

theorem thursday_dogs :
  ∃ (dogs_thursday : ℕ),
    dogs_thursday * pay_per_dog =
      total_earnings -
      (days_with_seven_dogs * dogs_mon_wed_fri + dogs_tuesday) * pay_per_dog ∧
    dogs_thursday = 9 :=
sorry

end NUMINAMATH_CALUDE_thursday_dogs_l2409_240980


namespace NUMINAMATH_CALUDE_software_hours_calculation_l2409_240938

def total_hours : ℝ := 68.33333333333333
def help_user_hours : ℝ := 17
def other_services_percentage : ℝ := 0.4

theorem software_hours_calculation :
  let other_services_hours := total_hours * other_services_percentage
  let software_hours := total_hours - help_user_hours - other_services_hours
  software_hours = 24 := by sorry

end NUMINAMATH_CALUDE_software_hours_calculation_l2409_240938


namespace NUMINAMATH_CALUDE_inequality_solution_set_l2409_240937

def solution_set (x : ℝ) : Prop := x ≥ 3 ∨ x ≤ 1

theorem inequality_solution_set
  (f : ℝ → ℝ)
  (f_even : ∀ x, f x = f (-x))
  (f_increasing : ∀ x y, 0 ≤ x → x ≤ y → f x ≤ f y)
  (f_one_eq_zero : f 1 = 0) :
  ∀ x, f (x - 2) ≥ 0 ↔ solution_set x :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l2409_240937


namespace NUMINAMATH_CALUDE_star_not_associative_l2409_240965

/-- Definition of the binary operation ⋆ -/
def star (x y : ℝ) : ℝ := (x + 2) * (y + 2) - x - y

/-- Theorem stating that the binary operation ⋆ is not associative -/
theorem star_not_associative : ¬ ∀ x y z : ℝ, star (star x y) z = star x (star y z) := by
  sorry

end NUMINAMATH_CALUDE_star_not_associative_l2409_240965


namespace NUMINAMATH_CALUDE_max_min_product_l2409_240990

theorem max_min_product (a b c : ℝ) (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : 0 ≤ c)
  (h4 : a + b + c = 8) (h5 : a * b + b * c + c * a = 16) :
  ∃ m : ℝ, m = min (a * b) (min (b * c) (c * a)) ∧ m ≤ 16 / 9 ∧
  ∃ a' b' c' : ℝ, 0 ≤ a' ∧ 0 ≤ b' ∧ 0 ≤ c' ∧
  a' + b' + c' = 8 ∧ a' * b' + b' * c' + c' * a' = 16 ∧
  min (a' * b') (min (b' * c') (c' * a')) = 16 / 9 := by
  sorry

end NUMINAMATH_CALUDE_max_min_product_l2409_240990


namespace NUMINAMATH_CALUDE_two_digit_number_swap_sum_theorem_l2409_240964

/-- Represents a two-digit number with distinct non-zero digits -/
structure TwoDigitNumber where
  tens : Nat
  units : Nat
  tens_not_zero : tens ≠ 0
  units_not_zero : units ≠ 0
  distinct_digits : tens ≠ units
  is_two_digit : tens < 10 ∧ units < 10

/-- The value of a TwoDigitNumber -/
def TwoDigitNumber.value (n : TwoDigitNumber) : Nat :=
  10 * n.tens + n.units

/-- The value of a TwoDigitNumber with swapped digits -/
def TwoDigitNumber.swapped_value (n : TwoDigitNumber) : Nat :=
  10 * n.units + n.tens

theorem two_digit_number_swap_sum_theorem 
  (a b c : TwoDigitNumber) 
  (h : a.value + b.value + c.value = 41) :
  a.swapped_value + b.swapped_value + c.swapped_value = 113 := by
  sorry

end NUMINAMATH_CALUDE_two_digit_number_swap_sum_theorem_l2409_240964


namespace NUMINAMATH_CALUDE_raviraj_cycling_journey_l2409_240910

/-- Raviraj's cycling journey --/
theorem raviraj_cycling_journey (initial_south distance_west_1 distance_north distance_west_2 distance_to_home : ℝ) :
  distance_west_1 = 10 ∧
  distance_north = 20 ∧
  distance_west_2 = 20 ∧
  distance_to_home = 30 ∧
  distance_west_1 + distance_west_2 = distance_to_home ∧
  initial_south + distance_north = distance_to_home →
  initial_south = 10 := by sorry

end NUMINAMATH_CALUDE_raviraj_cycling_journey_l2409_240910


namespace NUMINAMATH_CALUDE_alice_and_bob_money_l2409_240931

theorem alice_and_bob_money : (5 : ℚ) / 8 + (3 : ℚ) / 5 = 1.225 := by sorry

end NUMINAMATH_CALUDE_alice_and_bob_money_l2409_240931


namespace NUMINAMATH_CALUDE_incorrect_statement_about_parallelogram_l2409_240986

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

end NUMINAMATH_CALUDE_incorrect_statement_about_parallelogram_l2409_240986


namespace NUMINAMATH_CALUDE_polynomial_bound_l2409_240936

theorem polynomial_bound (a b c : ℝ) : 
  ∃ x : ℝ, x ∈ Set.Icc (-1 : ℝ) 1 ∧ 
    |x^3 + a*x^2 + b*x + c| ≥ (1/4 : ℝ) ∧
    ∀ ε > 0, ∃ a' b' c' x', 
      x' ∈ Set.Icc (-1 : ℝ) 1 ∧ 
      |x'^3 + a'*x'^2 + b'*x' + c'| < 1/4 + ε :=
by sorry

end NUMINAMATH_CALUDE_polynomial_bound_l2409_240936


namespace NUMINAMATH_CALUDE_janet_total_earnings_l2409_240997

/-- Calculates Janet's total earnings from exterminator work and sculpture sales -/
def janet_earnings (exterminator_rate : ℕ) (sculpture_rate : ℕ) (hours_worked : ℕ) (sculpture1_weight : ℕ) (sculpture2_weight : ℕ) : ℕ :=
  exterminator_rate * hours_worked + sculpture_rate * (sculpture1_weight + sculpture2_weight)

theorem janet_total_earnings :
  janet_earnings 70 20 20 5 7 = 1640 := by
  sorry

end NUMINAMATH_CALUDE_janet_total_earnings_l2409_240997


namespace NUMINAMATH_CALUDE_exponential_inequality_l2409_240984

theorem exponential_inequality (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h : Real.exp a + 2 * a = Real.exp b + 3 * b) : a < b := by
  sorry

end NUMINAMATH_CALUDE_exponential_inequality_l2409_240984


namespace NUMINAMATH_CALUDE_ellipse_eccentricity_l2409_240946

/-- Given an ellipse C and a line l, prove the eccentricity e --/
theorem ellipse_eccentricity (a b : ℝ) (h1 : a > b) (h2 : b > 0) : 
  ∃ (e : ℝ), 
    (∀ (x y : ℝ), x^2 / a^2 + y^2 / b^2 = 1 → 
      ∃ (A B M : ℝ × ℝ), 
        (A.2 = 0 ∧ B.1 = 0) ∧ 
        (M.2 = e * M.1 + a) ∧
        (A.1 = -a / e ∧ B.2 = a) ∧
        (M.1^2 / a^2 + M.2^2 / b^2 = 1) ∧
        ((M.1 - A.1)^2 + (M.2 - A.2)^2 = e^2 * ((B.1 - A.1)^2 + (B.2 - A.2)^2))) →
    e = (Real.sqrt 5 - 1) / 2 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_eccentricity_l2409_240946


namespace NUMINAMATH_CALUDE_inequality_relationship_l2409_240917

theorem inequality_relationship (a b : ℝ) (h1 : a + b < 0) (h2 : b > 0) :
  a^2 > -a*b ∧ -a*b > b^2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_relationship_l2409_240917


namespace NUMINAMATH_CALUDE_desired_average_grade_l2409_240971

def first_test_score : ℚ := 95
def second_test_score : ℚ := 80
def third_test_score : ℚ := 95

def average_grade : ℚ := (first_test_score + second_test_score + third_test_score) / 3

theorem desired_average_grade :
  average_grade = 90 := by sorry

end NUMINAMATH_CALUDE_desired_average_grade_l2409_240971


namespace NUMINAMATH_CALUDE_rectangle_side_length_l2409_240963

/-- A rectangle with given area and perimeter has a side of length 9 -/
theorem rectangle_side_length
  (area : ℝ)
  (perimeter : ℝ)
  (h_area : area = 117)
  (h_perimeter : perimeter = 44) :
  ∃ (length width : ℝ),
    length * width = area ∧
    2 * (length + width) = perimeter ∧
    (length = 9 ∨ width = 9) :=
sorry

end NUMINAMATH_CALUDE_rectangle_side_length_l2409_240963


namespace NUMINAMATH_CALUDE_pat_earned_stickers_l2409_240922

/-- The number of stickers Pat had at the start of the week -/
def initial_stickers : ℕ := 39

/-- The number of stickers Pat had at the end of the week -/
def final_stickers : ℕ := 61

/-- The number of stickers Pat earned during the week -/
def earned_stickers : ℕ := final_stickers - initial_stickers

theorem pat_earned_stickers : earned_stickers = 22 := by
  sorry

end NUMINAMATH_CALUDE_pat_earned_stickers_l2409_240922


namespace NUMINAMATH_CALUDE_equation_solution_l2409_240957

theorem equation_solution : ∃ x : ℝ, 14*x + 15*x + 18*x + 11 = 152 ∧ x = 3 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2409_240957


namespace NUMINAMATH_CALUDE_chromatic_number_le_max_degree_plus_one_l2409_240940

/-- A graph is represented by its vertex set and an adjacency relation -/
structure Graph (V : Type*) where
  adj : V → V → Prop

/-- The degree of a vertex in a graph -/
def degree (G : Graph V) (v : V) : ℕ := sorry

/-- The maximum degree of a graph -/
def maxDegree (G : Graph V) : ℕ := sorry

/-- A coloring of a graph is a function from vertices to colors -/
def isColoring (G : Graph V) (f : V → ℕ) : Prop :=
  ∀ u v : V, G.adj u v → f u ≠ f v

/-- The chromatic number of a graph -/
def chromaticNumber (G : Graph V) : ℕ := sorry

/-- Theorem: The chromatic number of a graph is at most one more than its maximum degree -/
theorem chromatic_number_le_max_degree_plus_one (V : Type*) (G : Graph V) :
  chromaticNumber G ≤ maxDegree G + 1 := by sorry

end NUMINAMATH_CALUDE_chromatic_number_le_max_degree_plus_one_l2409_240940


namespace NUMINAMATH_CALUDE_transportation_optimization_l2409_240952

/-- Demand function for transportation --/
def demand (p : ℝ) : ℝ := 3000 - 20 * p

/-- Transportation cost function for the bus company --/
def transportCost (y : ℝ) : ℝ := y + 5

/-- Fixed train fare --/
def trainFare : ℝ := 10

/-- Maximum train capacity --/
def trainCapacity : ℝ := 1000

/-- Optimal bus fare when train is operating --/
def optimalBusFare : ℝ := 50.5

/-- Decrease in total passengers after train closure --/
def passengerDecrease : ℝ := 500

theorem transportation_optimization :
  let busDemand (p : ℝ) := max 0 (demand p - trainCapacity)
  let busProfit (p : ℝ) := p * busDemand p - transportCost (busDemand p)
  let totalDemandWithTrain := trainCapacity + busDemand optimalBusFare
  let totalDemandWithoutTrain := demand 75.5
  (∀ p, p > trainFare → busProfit p ≤ busProfit optimalBusFare) ∧
  (totalDemandWithTrain - totalDemandWithoutTrain = passengerDecrease) := by
  sorry

end NUMINAMATH_CALUDE_transportation_optimization_l2409_240952


namespace NUMINAMATH_CALUDE_larger_number_is_fifty_l2409_240973

theorem larger_number_is_fifty (a b : ℝ) : 
  (4 * b = 5 * a) → (b - a = 10) → b = 50 := by
  sorry

end NUMINAMATH_CALUDE_larger_number_is_fifty_l2409_240973


namespace NUMINAMATH_CALUDE_hyperbola_dot_product_bound_l2409_240992

/-- The hyperbola with center at origin, left focus at (-2,0), and equation x²/a² - y² = 1 where a > 0 -/
structure Hyperbola where
  a : ℝ
  h_a_pos : a > 0

/-- A point on the right branch of the hyperbola -/
structure HyperbolaPoint (h : Hyperbola) where
  x : ℝ
  y : ℝ
  h_on_hyperbola : x^2 / h.a^2 - y^2 = 1
  h_right_branch : x ≥ h.a

/-- The theorem stating that the dot product of OP and FP is bounded below -/
theorem hyperbola_dot_product_bound (h : Hyperbola) (p : HyperbolaPoint h) :
  p.x * (p.x + 2) + p.y * p.y ≥ 3 + 2 * Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_hyperbola_dot_product_bound_l2409_240992


namespace NUMINAMATH_CALUDE_intersection_condition_l2409_240995

def A (a : ℝ) : Set ℝ := {-1, 0, a}

def B : Set ℝ := {x : ℝ | 1/3 < x ∧ x < 1}

theorem intersection_condition (a : ℝ) :
  (A a ∩ B).Nonempty → 1/3 < a ∧ a < 1 := by sorry

end NUMINAMATH_CALUDE_intersection_condition_l2409_240995


namespace NUMINAMATH_CALUDE_simplify_expression_l2409_240979

theorem simplify_expression : 
  2 - (2 / (1 + Real.sqrt 2)) - (2 / (1 - Real.sqrt 2)) = -2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2409_240979


namespace NUMINAMATH_CALUDE_min_guests_for_cheaper_second_planner_l2409_240969

/-- Represents the pricing model of an event planner -/
structure EventPlanner where
  flatFee : ℕ
  perGuestFee : ℕ

/-- Calculates the total cost for a given number of guests -/
def totalCost (planner : EventPlanner) (guests : ℕ) : ℕ :=
  planner.flatFee + planner.perGuestFee * guests

/-- Defines the two event planners -/
def planner1 : EventPlanner := { flatFee := 120, perGuestFee := 18 }
def planner2 : EventPlanner := { flatFee := 250, perGuestFee := 15 }

/-- Theorem stating the minimum number of guests for the second planner to be less expensive -/
theorem min_guests_for_cheaper_second_planner :
  ∀ n : ℕ, (n ≥ 44 → totalCost planner2 n < totalCost planner1 n) ∧
           (n < 44 → totalCost planner2 n ≥ totalCost planner1 n) := by
  sorry

end NUMINAMATH_CALUDE_min_guests_for_cheaper_second_planner_l2409_240969


namespace NUMINAMATH_CALUDE_perimeter_region_with_270_degree_arc_l2409_240956

/-- The perimeter of a region formed by two radii and a 270° arc in a circle -/
theorem perimeter_region_with_270_degree_arc (r : ℝ) (h : r = 7) :
  2 * r + (3/4) * (2 * Real.pi * r) = 14 + (21 * Real.pi / 2) := by
  sorry

end NUMINAMATH_CALUDE_perimeter_region_with_270_degree_arc_l2409_240956


namespace NUMINAMATH_CALUDE_number_of_siblings_l2409_240999

def total_spent : ℕ := 150
def cost_per_sibling : ℕ := 30
def cost_per_parent : ℕ := 30
def num_parents : ℕ := 2

theorem number_of_siblings :
  (total_spent - num_parents * cost_per_parent) / cost_per_sibling = 3 := by
  sorry

end NUMINAMATH_CALUDE_number_of_siblings_l2409_240999


namespace NUMINAMATH_CALUDE_nested_rectangles_l2409_240918

/-- A rectangle with integer side lengths -/
structure Rectangle where
  width : Nat
  height : Nat
  width_pos : width > 0
  height_pos : height > 0
  width_bound : width ≤ 100
  height_bound : height ≤ 100

/-- Predicate to check if one rectangle fits inside another -/
def fits_inside (r1 r2 : Rectangle) : Prop :=
  (r1.width ≤ r2.width ∧ r1.height ≤ r2.height) ∨
  (r1.width ≤ r2.height ∧ r1.height ≤ r2.width)

/-- Main theorem: Given 101 rectangles, there exist 3 that fit inside each other -/
theorem nested_rectangles (rectangles : Finset Rectangle) 
    (h : rectangles.card = 101) : 
    ∃ (A B C : Rectangle), A ∈ rectangles ∧ B ∈ rectangles ∧ C ∈ rectangles ∧
    fits_inside A B ∧ fits_inside B C := by
  sorry

end NUMINAMATH_CALUDE_nested_rectangles_l2409_240918


namespace NUMINAMATH_CALUDE_regular_polygon_150_degree_angles_l2409_240958

/-- A regular polygon with interior angles measuring 150 degrees has 12 sides -/
theorem regular_polygon_150_degree_angles (n : ℕ) : 
  (n ≥ 3) →                          -- A polygon has at least 3 sides
  (∀ i : ℕ, i < n → 150 = (n - 2) * 180 / n) →  -- Each interior angle is 150 degrees
  n = 12 := by
sorry

end NUMINAMATH_CALUDE_regular_polygon_150_degree_angles_l2409_240958


namespace NUMINAMATH_CALUDE_canoe_upstream_speed_l2409_240962

/-- The speed of a canoe rowing upstream, given its downstream speed and the stream speed -/
theorem canoe_upstream_speed (downstream_speed stream_speed : ℝ) :
  downstream_speed = 12 →
  stream_speed = 4 →
  downstream_speed - 2 * stream_speed = 4 := by
  sorry

end NUMINAMATH_CALUDE_canoe_upstream_speed_l2409_240962


namespace NUMINAMATH_CALUDE_data_mode_is_neg_one_l2409_240939

def data : List Int := [-1, 0, 2, -1, 3]

def mode (l : List α) [DecidableEq α] : Option α :=
  l.foldl (λ acc x => 
    match acc with
    | none => some x
    | some y => if l.count x > l.count y then some x else some y
  ) none

theorem data_mode_is_neg_one : mode data = some (-1) := by
  sorry

end NUMINAMATH_CALUDE_data_mode_is_neg_one_l2409_240939


namespace NUMINAMATH_CALUDE_polygon_sides_l2409_240985

theorem polygon_sides (n : ℕ) : 
  (n ≥ 3) → 
  ((n - 2) * 180 = 3 * 360) → 
  n = 8 := by
sorry

end NUMINAMATH_CALUDE_polygon_sides_l2409_240985


namespace NUMINAMATH_CALUDE_tangent_line_equation_l2409_240987

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 + Real.log x

theorem tangent_line_equation (a : ℝ) :
  (∀ ε > 0, ∃ δ > 0, ∀ x, 0 < |x - 1| ∧ |x - 1| < δ → 
    |((f a x - f a 1) / (x - 1)) - 3| < ε) →
  ∃ b c : ℝ, ∀ x y : ℝ, y = f a x → (x = 1 ∧ y = f a 1) → 
    3 * x - y - 2 = 0 := by sorry

end NUMINAMATH_CALUDE_tangent_line_equation_l2409_240987


namespace NUMINAMATH_CALUDE_smallest_difference_2010_l2409_240915

theorem smallest_difference_2010 (a b : ℕ+) : 
  a * b = 2010 → a > b → 
  ∀ (c d : ℕ+), c * d = 2010 → c > d → a - b ≤ c - d → a - b = 37 := by
  sorry

end NUMINAMATH_CALUDE_smallest_difference_2010_l2409_240915


namespace NUMINAMATH_CALUDE_total_cards_l2409_240951

theorem total_cards (deck_a deck_b deck_c deck_d : ℕ)
  (ha : deck_a = 52)
  (hb : deck_b = 40)
  (hc : deck_c = 50)
  (hd : deck_d = 48) :
  deck_a + deck_b + deck_c + deck_d = 190 := by
  sorry

end NUMINAMATH_CALUDE_total_cards_l2409_240951


namespace NUMINAMATH_CALUDE_expression_evaluation_l2409_240988

theorem expression_evaluation : 
  let x : ℝ := -3
  (5 + x * (4 + x) - 4^2 + (x^2 - 3*x + 2)) / (x^2 - 4 + x - 1) = 6 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2409_240988


namespace NUMINAMATH_CALUDE_article_cost_price_l2409_240927

/-- The cost price of an article given its marked price and profit percentages -/
theorem article_cost_price (marked_price : ℝ) (discount_percent : ℝ) (profit_percent : ℝ) : 
  marked_price = 87.5 → 
  discount_percent = 5 → 
  profit_percent = 25 → 
  (1 - discount_percent / 100) * marked_price = (1 + profit_percent / 100) * (marked_price * (1 - discount_percent / 100) / (1 + profit_percent / 100)) → 
  marked_price * (1 - discount_percent / 100) / (1 + profit_percent / 100) = 66.5 := by
sorry

end NUMINAMATH_CALUDE_article_cost_price_l2409_240927


namespace NUMINAMATH_CALUDE_satisfactory_fraction_is_eleven_fifteenths_l2409_240954

/-- Represents the distribution of grades in a class assessment. -/
structure GradeDistribution where
  a : Nat -- Number of A's
  b : Nat -- Number of B's
  c : Nat -- Number of C's
  d : Nat -- Number of D's
  ef : Nat -- Number of E's and F's combined

/-- Calculates the fraction of satisfactory grades given a grade distribution. -/
def fractionSatisfactory (grades : GradeDistribution) : Rat :=
  let satisfactory := grades.a + grades.b + grades.c + grades.d
  let total := satisfactory + grades.ef
  satisfactory / total

/-- Theorem stating that for the given grade distribution, 
    the fraction of satisfactory grades is 11/15. -/
theorem satisfactory_fraction_is_eleven_fifteenths : 
  fractionSatisfactory { a := 7, b := 6, c := 5, d := 4, ef := 8 } = 11 / 15 := by
  sorry


end NUMINAMATH_CALUDE_satisfactory_fraction_is_eleven_fifteenths_l2409_240954


namespace NUMINAMATH_CALUDE_unique_intersection_point_l2409_240993

open Real

noncomputable def f (m : ℝ) (x : ℝ) : ℝ := (1/2) * x^2 - m * log x

noncomputable def g (m : ℝ) (x : ℝ) : ℝ := x^2 - (m + 1) * x

noncomputable def h (m : ℝ) (x : ℝ) : ℝ := f m x - g m x

theorem unique_intersection_point (m : ℝ) (hm : m ≥ 1) :
  ∃! x, x > 0 ∧ h m x = 0 := by sorry

end NUMINAMATH_CALUDE_unique_intersection_point_l2409_240993


namespace NUMINAMATH_CALUDE_meaningful_expression_l2409_240920

theorem meaningful_expression (x : ℝ) : 
  (∃ y : ℝ, y = 1 / (x - 2023)) ↔ x ≠ 2023 := by
sorry

end NUMINAMATH_CALUDE_meaningful_expression_l2409_240920


namespace NUMINAMATH_CALUDE_right_triangle_abc_area_l2409_240960

/-- A right triangle ABC in the xy-plane with specific properties -/
structure RightTriangleABC where
  -- Point A
  a : ℝ × ℝ
  -- Point B
  b : ℝ × ℝ
  -- Point C (right angle)
  c : ℝ × ℝ
  -- Hypotenuse length
  ab_length : ℝ
  -- Median through A equation
  median_a_slope : ℝ
  median_a_intercept : ℝ
  -- Median through B equation
  median_b_slope : ℝ
  median_b_intercept : ℝ
  -- Conditions
  right_angle_at_c : (a.1 - c.1) * (b.1 - c.1) + (a.2 - c.2) * (b.2 - c.2) = 0
  hypotenuse_length : (a.1 - b.1)^2 + (a.2 - b.2)^2 = ab_length^2
  median_a_equation : ∀ x y, y = median_a_slope * x + median_a_intercept → 
    2 * x = a.1 + c.1 ∧ 2 * y = a.2 + c.2
  median_b_equation : ∀ x y, y = median_b_slope * x + median_b_intercept → 
    2 * x = b.1 + c.1 ∧ 2 * y = b.2 + c.2

/-- The area of the right triangle ABC with given properties is 175 -/
theorem right_triangle_abc_area 
  (t : RightTriangleABC) 
  (h1 : t.ab_length = 50) 
  (h2 : t.median_a_slope = 1 ∧ t.median_a_intercept = 5)
  (h3 : t.median_b_slope = 2 ∧ t.median_b_intercept = 6) :
  abs ((t.a.1 * t.b.2 - t.b.1 * t.a.2) / 2) = 175 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_abc_area_l2409_240960


namespace NUMINAMATH_CALUDE_coefficient_x_cubed_in_expansion_l2409_240901

/-- The coefficient of x^3 in the expansion of (2x + √x)^5 is 10 -/
theorem coefficient_x_cubed_in_expansion : ℕ := by
  sorry

end NUMINAMATH_CALUDE_coefficient_x_cubed_in_expansion_l2409_240901


namespace NUMINAMATH_CALUDE_combination_sum_equals_466_l2409_240925

theorem combination_sum_equals_466 (n : ℕ) 
  (h1 : 38 ≥ n) 
  (h2 : 3 * n ≥ 38 - n) 
  (h3 : n + 21 ≥ 3 * n) : 
  Nat.choose (3 * n) (38 - n) + Nat.choose (n + 21) (3 * n) = 466 := by
  sorry

end NUMINAMATH_CALUDE_combination_sum_equals_466_l2409_240925


namespace NUMINAMATH_CALUDE_greg_marbles_l2409_240911

/-- The number of marbles Adam has -/
def adam_marbles : ℕ := 29

/-- The number of additional marbles Greg has compared to Adam -/
def greg_additional_marbles : ℕ := 14

/-- Theorem: Greg has 43 marbles -/
theorem greg_marbles : adam_marbles + greg_additional_marbles = 43 := by
  sorry

end NUMINAMATH_CALUDE_greg_marbles_l2409_240911


namespace NUMINAMATH_CALUDE_toms_beef_quantity_tom_has_ten_pounds_beef_l2409_240914

/-- Represents the problem of determining Tom's beef quantity for lasagna -/
theorem toms_beef_quantity (noodles_to_beef_ratio : ℕ) 
  (existing_noodles : ℕ) (package_size : ℕ) (packages_to_buy : ℕ) : ℕ :=
  let total_noodles := existing_noodles + package_size * packages_to_buy
  total_noodles / noodles_to_beef_ratio

/-- Proves that Tom has 10 pounds of beef given the problem conditions -/
theorem tom_has_ten_pounds_beef : 
  toms_beef_quantity 2 4 2 8 = 10 := by
  sorry

end NUMINAMATH_CALUDE_toms_beef_quantity_tom_has_ten_pounds_beef_l2409_240914


namespace NUMINAMATH_CALUDE_biology_magnet_problem_l2409_240977

def word : Finset Char := {'B', 'I', 'O', 'L', 'O', 'G', 'Y'}
def vowels : Finset Char := {'I', 'O', 'Y'}
def consonants : Finset Char := {'B', 'L', 'G'}

def distinct_collections : ℕ := sorry

theorem biology_magnet_problem :
  (word.card = 7) →
  (vowels ⊆ word) →
  (consonants ⊆ word) →
  (vowels ∩ consonants = ∅) →
  (vowels ∪ consonants = word) →
  (distinct_collections = 12) := by sorry

end NUMINAMATH_CALUDE_biology_magnet_problem_l2409_240977


namespace NUMINAMATH_CALUDE_john_skateboard_distance_l2409_240916

/-- Represents the distance John traveled in miles -/
structure JohnTrip where
  skateboard_to_park : ℕ
  walk_to_park : ℕ

/-- Calculates the total distance John skateboarded -/
def total_skateboard_distance (trip : JohnTrip) : ℕ :=
  2 * trip.skateboard_to_park + trip.skateboard_to_park

theorem john_skateboard_distance :
  ∀ (trip : JohnTrip),
    trip.skateboard_to_park = 10 ∧ trip.walk_to_park = 4 →
    total_skateboard_distance trip = 24 :=
by
  sorry

#check john_skateboard_distance

end NUMINAMATH_CALUDE_john_skateboard_distance_l2409_240916


namespace NUMINAMATH_CALUDE_original_average_rent_l2409_240919

theorem original_average_rent
  (num_friends : ℕ)
  (original_rent : ℝ)
  (increased_rent : ℝ)
  (new_average : ℝ)
  (h1 : num_friends = 4)
  (h2 : original_rent = 1250)
  (h3 : increased_rent = 1250 * 1.16)
  (h4 : new_average = 850)
  : (num_friends * new_average - increased_rent + original_rent) / num_friends = 800 := by
  sorry

end NUMINAMATH_CALUDE_original_average_rent_l2409_240919
