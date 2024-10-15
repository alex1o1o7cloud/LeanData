import Mathlib

namespace NUMINAMATH_CALUDE_partnership_profit_calculation_l3664_366429

/-- Represents the investment and profit information for a partnership business --/
structure PartnershipBusiness where
  a_initial : ℕ
  a_additional : ℕ
  a_additional_time : ℕ
  b_initial : ℕ
  b_withdrawal : ℕ
  b_withdrawal_time : ℕ
  c_initial : ℕ
  c_additional : ℕ
  c_additional_time : ℕ
  total_time : ℕ
  c_profit : ℕ

/-- Calculates the total profit of the partnership business --/
def calculate_total_profit (pb : PartnershipBusiness) : ℕ :=
  sorry

/-- Theorem stating that given the specific investment conditions, 
    if C's profit is 45000, then the total profit is 103571 --/
theorem partnership_profit_calculation 
  (pb : PartnershipBusiness)
  (h1 : pb.a_initial = 5000)
  (h2 : pb.a_additional = 2000)
  (h3 : pb.a_additional_time = 4)
  (h4 : pb.b_initial = 8000)
  (h5 : pb.b_withdrawal = 1000)
  (h6 : pb.b_withdrawal_time = 4)
  (h7 : pb.c_initial = 9000)
  (h8 : pb.c_additional = 3000)
  (h9 : pb.c_additional_time = 6)
  (h10 : pb.total_time = 12)
  (h11 : pb.c_profit = 45000) :
  calculate_total_profit pb = 103571 :=
sorry

end NUMINAMATH_CALUDE_partnership_profit_calculation_l3664_366429


namespace NUMINAMATH_CALUDE_product_remainder_l3664_366407

theorem product_remainder (a b m : ℕ) (ha : a = 1492) (hb : b = 1999) (hm : m = 500) :
  (a * b) % m = 8 := by
  sorry

end NUMINAMATH_CALUDE_product_remainder_l3664_366407


namespace NUMINAMATH_CALUDE_stock_price_theorem_l3664_366417

/-- The face value of the stock (assumed to be $100) -/
def faceValue : ℝ := 100

/-- A's stock interest rate -/
def interestRateA : ℝ := 0.10

/-- B's stock interest rate -/
def interestRateB : ℝ := 0.12

/-- The amount B must invest to get an equally good investment -/
def bInvestment : ℝ := 115.2

/-- The price of the stock A invested in -/
def stockPriceA : ℝ := 138.24

/-- Theorem stating that given the conditions, the price of A's stock is $138.24 -/
theorem stock_price_theorem :
  let incomeA := faceValue * interestRateA
  let requiredInvestmentB := incomeA / interestRateB
  let marketPriceB := bInvestment * (faceValue / requiredInvestmentB)
  marketPriceB = stockPriceA := by
  sorry

#check stock_price_theorem

end NUMINAMATH_CALUDE_stock_price_theorem_l3664_366417


namespace NUMINAMATH_CALUDE_field_height_rise_l3664_366435

/-- Calculates the rise in height of a field after digging a pit and spreading the removed earth --/
theorem field_height_rise (field_length field_width pit_length pit_width pit_depth : ℝ) 
  (h_field_length : field_length = 20)
  (h_field_width : field_width = 10)
  (h_pit_length : pit_length = 8)
  (h_pit_width : pit_width = 5)
  (h_pit_depth : pit_depth = 2) :
  let total_area := field_length * field_width
  let pit_area := pit_length * pit_width
  let remaining_area := total_area - pit_area
  let pit_volume := pit_length * pit_width * pit_depth
  pit_volume / remaining_area = 0.5 := by sorry

end NUMINAMATH_CALUDE_field_height_rise_l3664_366435


namespace NUMINAMATH_CALUDE_johns_allowance_l3664_366444

/-- John's weekly allowance problem -/
theorem johns_allowance (A : ℚ) : A = 32.4 ↔
  ∃ (arcade toy book candy : ℚ),
    -- Spending at the arcade
    arcade = 7 / 12 * A ∧
    -- Spending at the toy store
    toy = 5 / 9 * (A - arcade) ∧
    -- Spending at the bookstore
    book = 3 / 4 * (A - arcade - toy) ∧
    -- Spending at the candy store
    candy = 3 / 2 ∧
    -- Total spending equals the allowance
    arcade + toy + book + candy = A := by
  sorry

end NUMINAMATH_CALUDE_johns_allowance_l3664_366444


namespace NUMINAMATH_CALUDE_interest_rate_frequency_relationship_l3664_366492

/-- The nominal annual interest rate -/
def nominal_rate : ℝ := 0.16

/-- The effective annual interest rate -/
def effective_rate : ℝ := 0.1664

/-- The frequency of interest payments per year -/
def frequency : ℕ := 2

/-- Theorem stating that the given frequency satisfies the relationship between nominal and effective rates -/
theorem interest_rate_frequency_relationship : 
  (1 + nominal_rate / frequency)^frequency - 1 = effective_rate := by sorry

end NUMINAMATH_CALUDE_interest_rate_frequency_relationship_l3664_366492


namespace NUMINAMATH_CALUDE_total_toothpicks_needed_l3664_366447

/-- The number of small triangles in the base row of the large equilateral triangle. -/
def base_triangles : ℕ := 2004

/-- The total number of small triangles in the large equilateral triangle. -/
def total_triangles : ℕ := base_triangles * (base_triangles + 1) / 2

/-- The number of toothpicks needed if each side of each small triangle was unique. -/
def total_sides : ℕ := 3 * total_triangles

/-- The number of toothpicks on the boundary of the large triangle. -/
def boundary_toothpicks : ℕ := 3 * base_triangles

/-- Theorem: The total number of toothpicks needed to construct the large equilateral triangle. -/
theorem total_toothpicks_needed : 
  (total_sides / 2) + boundary_toothpicks = 3021042 := by
  sorry

end NUMINAMATH_CALUDE_total_toothpicks_needed_l3664_366447


namespace NUMINAMATH_CALUDE_even_function_implies_a_equals_one_l3664_366450

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := (a - 2) * x^2 + (a - 1) * x + 3

-- State the theorem
theorem even_function_implies_a_equals_one :
  (∀ x : ℝ, f a x = f a (-x)) → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_even_function_implies_a_equals_one_l3664_366450


namespace NUMINAMATH_CALUDE_tax_revenue_change_l3664_366452

theorem tax_revenue_change (T C : ℝ) (hT : T > 0) (hC : C > 0) :
  let new_tax := 0.86 * T
  let new_consumption := 1.15 * C
  let original_revenue := T * C
  let new_revenue := new_tax * new_consumption
  (new_revenue / original_revenue - 1) * 100 = -1.1 := by
sorry

end NUMINAMATH_CALUDE_tax_revenue_change_l3664_366452


namespace NUMINAMATH_CALUDE_evaluate_expression_l3664_366485

theorem evaluate_expression : 150 * (150 - 4) - 2 * (150 * 150 - 4) = -23092 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l3664_366485


namespace NUMINAMATH_CALUDE_olivia_soda_purchase_l3664_366472

/-- The number of quarters Olivia spent on a soda -/
def quarters_spent (initial : ℕ) (remaining : ℕ) : ℕ :=
  initial - remaining

/-- Theorem stating that Olivia spent 4 quarters on the soda -/
theorem olivia_soda_purchase : quarters_spent 11 7 = 4 := by
  sorry

end NUMINAMATH_CALUDE_olivia_soda_purchase_l3664_366472


namespace NUMINAMATH_CALUDE_reflection_across_origin_l3664_366414

/-- Reflects a point across the origin -/
def reflect_origin (p : ℝ × ℝ) : ℝ × ℝ :=
  (-p.1, -p.2)

/-- The original point P -/
def P : ℝ × ℝ := (-2, -3)

/-- The reflected point Q -/
def Q : ℝ × ℝ := (2, 3)

theorem reflection_across_origin :
  reflect_origin P = Q := by sorry

end NUMINAMATH_CALUDE_reflection_across_origin_l3664_366414


namespace NUMINAMATH_CALUDE_inscribed_sphere_radius_specific_prism_l3664_366416

/-- An equilateral triangular prism -/
structure EquilateralTriangularPrism where
  /-- The base side length of the prism -/
  baseSideLength : ℝ
  /-- The height of the prism -/
  height : ℝ

/-- The radius of the inscribed sphere in an equilateral triangular prism -/
def inscribedSphereRadius (prism : EquilateralTriangularPrism) : ℝ :=
  sorry

/-- Theorem: The radius of the inscribed sphere in an equilateral triangular prism
    with base side length 1 and height √2 is equal to √2/6 -/
theorem inscribed_sphere_radius_specific_prism :
  let prism : EquilateralTriangularPrism := { baseSideLength := 1, height := Real.sqrt 2 }
  inscribedSphereRadius prism = Real.sqrt 2 / 6 :=
by sorry

end NUMINAMATH_CALUDE_inscribed_sphere_radius_specific_prism_l3664_366416


namespace NUMINAMATH_CALUDE_mary_flour_calculation_l3664_366415

/-- The number of cups of flour Mary put in -/
def flour_put_in : ℕ := 2

/-- The total number of cups of flour required by the recipe -/
def total_flour : ℕ := 10

/-- The number of cups of sugar required by the recipe -/
def sugar : ℕ := 3

/-- The additional cups of flour needed compared to sugar -/
def extra_flour : ℕ := 5

theorem mary_flour_calculation :
  flour_put_in = total_flour - (sugar + extra_flour) :=
by sorry

end NUMINAMATH_CALUDE_mary_flour_calculation_l3664_366415


namespace NUMINAMATH_CALUDE_invalid_league_schedule_l3664_366428

/-- Represents a league schedule --/
structure LeagueSchedule where
  num_teams : Nat
  num_dates : Nat
  max_games_per_date : Nat

/-- Calculate the total number of games in a round-robin tournament --/
def total_games (schedule : LeagueSchedule) : Nat :=
  schedule.num_teams * (schedule.num_teams - 1) / 2

/-- Check if a schedule is valid --/
def is_valid_schedule (schedule : LeagueSchedule) : Prop :=
  total_games schedule ≤ schedule.num_dates * schedule.max_games_per_date

/-- Theorem stating that the given schedule is invalid --/
theorem invalid_league_schedule : 
  ¬ is_valid_schedule ⟨20, 5, 8⟩ := by
  sorry

#eval total_games ⟨20, 5, 8⟩

end NUMINAMATH_CALUDE_invalid_league_schedule_l3664_366428


namespace NUMINAMATH_CALUDE_abc_inequality_l3664_366402

theorem abc_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h : a^2 + b^2 + c^2 + a*b*c = 4) :
  a^2 * b^2 + b^2 * c^2 + c^2 * a^2 + a*b*c ≤ 4 := by
  sorry

end NUMINAMATH_CALUDE_abc_inequality_l3664_366402


namespace NUMINAMATH_CALUDE_jay_and_paul_distance_l3664_366461

/-- Calculates the distance traveled given a speed and time --/
def distance (speed : ℚ) (time : ℚ) : ℚ := speed * time

/-- Proves that Jay and Paul will be 20 miles apart after walking in opposite directions for 2 hours --/
theorem jay_and_paul_distance : 
  let jay_speed : ℚ := 1 / 15  -- 1 mile per 15 minutes
  let paul_speed : ℚ := 3 / 30 -- 3 miles per 30 minutes
  let time : ℚ := 2 * 60      -- 2 hours in minutes
  distance jay_speed time + distance paul_speed time = 20 := by
sorry

end NUMINAMATH_CALUDE_jay_and_paul_distance_l3664_366461


namespace NUMINAMATH_CALUDE_total_sales_proof_l3664_366451

def robyn_sales : ℕ := 55
def lucy_sales : ℕ := 43

theorem total_sales_proof : robyn_sales + lucy_sales = 98 := by
  sorry

end NUMINAMATH_CALUDE_total_sales_proof_l3664_366451


namespace NUMINAMATH_CALUDE_consecutive_sum_product_l3664_366453

theorem consecutive_sum_product (a : ℤ) : (3*a + 3) * (3*a + 12) ≠ 111111111 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_sum_product_l3664_366453


namespace NUMINAMATH_CALUDE_jim_purchase_cost_l3664_366401

/-- The cost of a lamp in dollars -/
def lamp_cost : ℝ := 7

/-- The cost difference between a lamp and a bulb in dollars -/
def cost_difference : ℝ := 4

/-- The number of lamps bought -/
def num_lamps : ℕ := 2

/-- The number of bulbs bought -/
def num_bulbs : ℕ := 6

/-- The total cost of Jim's purchase -/
def total_cost : ℝ := num_lamps * lamp_cost + num_bulbs * (lamp_cost - cost_difference)

theorem jim_purchase_cost :
  total_cost = 32 := by sorry

end NUMINAMATH_CALUDE_jim_purchase_cost_l3664_366401


namespace NUMINAMATH_CALUDE_class_average_score_l3664_366436

theorem class_average_score (total_students : ℕ) (present_students : ℕ) (initial_average : ℚ) (makeup_score : ℚ) :
  total_students = 40 →
  present_students = 38 →
  initial_average = 92 →
  makeup_score = 100 →
  ((initial_average * present_students + makeup_score * (total_students - present_students)) / total_students) = 92.4 := by
  sorry

end NUMINAMATH_CALUDE_class_average_score_l3664_366436


namespace NUMINAMATH_CALUDE_absolute_value_inequality_l3664_366483

theorem absolute_value_inequality (k : ℝ) : 
  (∀ x : ℝ, |x + 1| + |x - 2| > k) ↔ k < 3 := by sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_l3664_366483


namespace NUMINAMATH_CALUDE_bonsai_earnings_proof_l3664_366498

/-- Calculates the total earnings from selling bonsai. -/
def total_earnings (small_cost big_cost : ℕ) (small_sold big_sold : ℕ) : ℕ :=
  small_cost * small_sold + big_cost * big_sold

/-- Proves that the total earnings from selling 3 small bonsai at $30 each
    and 5 big bonsai at $20 each is equal to $190. -/
theorem bonsai_earnings_proof :
  total_earnings 30 20 3 5 = 190 := by
  sorry

end NUMINAMATH_CALUDE_bonsai_earnings_proof_l3664_366498


namespace NUMINAMATH_CALUDE_pedal_triangles_common_circumcircle_l3664_366456

/-- Triangle type -/
structure Triangle where
  A : Point
  B : Point
  C : Point

/-- Isotomic conjugates with respect to a triangle -/
def IsotomicConjugates (P₁ P₂ : Point) (T : Triangle) : Prop := sorry

/-- Pedal triangle of a point with respect to a triangle, given an angle -/
def PedalTriangle (P : Point) (T : Triangle) (angle : ℝ) : Triangle := sorry

/-- Circumcircle of a triangle -/
def Circumcircle (T : Triangle) : Circle := sorry

/-- Center of a circle -/
def Center (C : Circle) : Point := sorry

/-- Midpoint of a segment -/
def Midpoint (A B : Point) : Point := sorry

theorem pedal_triangles_common_circumcircle 
  (T : Triangle) (P₁ P₂ : Point) (angle : ℝ) :
  IsotomicConjugates P₁ P₂ T →
  ∃ (C : Circle), 
    Circumcircle (PedalTriangle P₁ T angle) = C ∧
    Circumcircle (PedalTriangle P₂ T angle) = C ∧
    Center C = Midpoint P₁ P₂ := by
  sorry

end NUMINAMATH_CALUDE_pedal_triangles_common_circumcircle_l3664_366456


namespace NUMINAMATH_CALUDE_son_work_time_l3664_366443

theorem son_work_time (man_time son_father_time : ℝ) 
  (h1 : man_time = 5)
  (h2 : son_father_time = 3) : 
  let man_rate := 1 / man_time
  let combined_rate := 1 / son_father_time
  let son_rate := combined_rate - man_rate
  1 / son_rate = 7.5 := by sorry

end NUMINAMATH_CALUDE_son_work_time_l3664_366443


namespace NUMINAMATH_CALUDE_quadratic_coefficient_unique_l3664_366423

/-- A quadratic function with integer coefficients -/
structure QuadraticFunction where
  a : ℤ
  b : ℤ
  c : ℤ

/-- The y-value of a quadratic function at a given x -/
def QuadraticFunction.evaluate (f : QuadraticFunction) (x : ℚ) : ℚ :=
  f.a * x^2 + f.b * x + f.c

/-- The x-coordinate of the vertex of a quadratic function -/
def QuadraticFunction.vertexX (f : QuadraticFunction) : ℚ :=
  -f.b / (2 * f.a)

/-- The y-coordinate of the vertex of a quadratic function -/
def QuadraticFunction.vertexY (f : QuadraticFunction) : ℚ :=
  f.evaluate (f.vertexX)

theorem quadratic_coefficient_unique (f : QuadraticFunction) :
    f.vertexX = 2 ∧ f.vertexY = -3 ∧ f.evaluate 1 = -2 → f.a = 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_coefficient_unique_l3664_366423


namespace NUMINAMATH_CALUDE_correct_verb_forms_l3664_366424

/-- Represents the grammatical number of a subject --/
inductive GrammaticalNumber
| Singular
| Plural

/-- Represents a subject in a sentence --/
structure Subject where
  text : String
  number : GrammaticalNumber

/-- Represents a verb in a sentence --/
structure Verb where
  singular_form : String
  plural_form : String

/-- Checks if a verb agrees with a subject --/
def verb_agrees (s : Subject) (v : Verb) : Prop :=
  match s.number with
  | GrammaticalNumber.Singular => v.singular_form = "is"
  | GrammaticalNumber.Plural => v.plural_form = "want"

/-- The main theorem stating the correct verb forms for the given subjects --/
theorem correct_verb_forms 
  (subject1 : Subject)
  (subject2 : Subject)
  (h1 : subject1.text = "The number of the stamps")
  (h2 : subject2.text = "a number of people")
  (h3 : subject1.number = GrammaticalNumber.Singular)
  (h4 : subject2.number = GrammaticalNumber.Plural) :
  ∃ (v1 v2 : Verb), 
    verb_agrees subject1 v1 ∧ 
    verb_agrees subject2 v2 ∧ 
    v1.singular_form = "is" ∧ 
    v2.plural_form = "want" := by
  sorry


end NUMINAMATH_CALUDE_correct_verb_forms_l3664_366424


namespace NUMINAMATH_CALUDE_encyclopedia_monthly_payment_l3664_366405

/-- Proves that the monthly payment for the encyclopedia purchase is $57 -/
theorem encyclopedia_monthly_payment
  (total_cost : ℝ)
  (down_payment : ℝ)
  (num_monthly_payments : ℕ)
  (final_payment : ℝ)
  (interest_rate : ℝ)
  (h_total_cost : total_cost = 750)
  (h_down_payment : down_payment = 300)
  (h_num_monthly_payments : num_monthly_payments = 9)
  (h_final_payment : final_payment = 21)
  (h_interest_rate : interest_rate = 0.18666666666666668)
  : ∃ (monthly_payment : ℝ),
    monthly_payment = 57 ∧
    total_cost - down_payment + (total_cost - down_payment) * interest_rate =
    monthly_payment * num_monthly_payments + final_payment := by
  sorry

end NUMINAMATH_CALUDE_encyclopedia_monthly_payment_l3664_366405


namespace NUMINAMATH_CALUDE_parabola_chord_through_focus_l3664_366468

/-- Given a parabola y² = 2px with p > 0, if a chord AB passes through the focus F
    such that |AF| = 2 and |BF| = 3, then p = 12/5 -/
theorem parabola_chord_through_focus (p : ℝ) (A B F : ℝ × ℝ) :
  p > 0 →
  (∀ x y, y^2 = 2*p*x) →
  F.1 = p/2 ∧ F.2 = 0 →
  (A.1 - F.1)^2 + (A.2 - F.2)^2 = 4 →
  (B.1 - F.1)^2 + (B.2 - F.2)^2 = 9 →
  p = 12/5 := by
sorry

end NUMINAMATH_CALUDE_parabola_chord_through_focus_l3664_366468


namespace NUMINAMATH_CALUDE_odd_function_inequality_l3664_366486

-- Define the properties of the function f
def IsOddFunction (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

def HasPositiveProduct (f : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂, x₁ > 0 → x₂ > 0 → x₁ ≠ x₂ → (x₁ - x₂) * (f x₁ - f x₂) > 0

-- State the theorem
theorem odd_function_inequality (f : ℝ → ℝ) 
  (h_odd : IsOddFunction f) (h_pos : HasPositiveProduct f) : 
  f 4 < f (-6) := by
  sorry

end NUMINAMATH_CALUDE_odd_function_inequality_l3664_366486


namespace NUMINAMATH_CALUDE_chess_tournament_participants_l3664_366454

theorem chess_tournament_participants (n : ℕ) : 
  (n * (n - 1)) / 2 = 105 → n = 15 := by sorry

end NUMINAMATH_CALUDE_chess_tournament_participants_l3664_366454


namespace NUMINAMATH_CALUDE_frank_first_half_correct_l3664_366411

/-- Represents the trivia game scenario -/
structure TriviaGame where
  points_per_question : ℕ
  final_score : ℕ
  second_half_correct : ℕ

/-- Calculates the number of questions answered correctly in the first half -/
def first_half_correct (game : TriviaGame) : ℕ :=
  (game.final_score - game.second_half_correct * game.points_per_question) / game.points_per_question

/-- Theorem stating that Frank answered 3 questions correctly in the first half -/
theorem frank_first_half_correct :
  let game : TriviaGame := {
    points_per_question := 3,
    final_score := 15,
    second_half_correct := 2
  }
  first_half_correct game = 3 := by
  sorry

end NUMINAMATH_CALUDE_frank_first_half_correct_l3664_366411


namespace NUMINAMATH_CALUDE_vector_operation_result_l3664_366493

theorem vector_operation_result :
  let v1 : Fin 3 → ℝ := ![(-3), 4, 2]
  let v2 : Fin 3 → ℝ := ![1, 6, (-3)]
  2 • v1 + v2 = ![-5, 14, 1] := by sorry

end NUMINAMATH_CALUDE_vector_operation_result_l3664_366493


namespace NUMINAMATH_CALUDE_number_divided_by_16_equals_16_times_8_l3664_366473

theorem number_divided_by_16_equals_16_times_8 : 
  2048 / 16 = 16 * 8 := by sorry

end NUMINAMATH_CALUDE_number_divided_by_16_equals_16_times_8_l3664_366473


namespace NUMINAMATH_CALUDE_small_tile_position_l3664_366434

/-- Represents a tile in the square --/
inductive Tile
| Large : Tile  -- 1×3 tile
| Small : Tile  -- 1×1 tile

/-- Represents a position in the 7×7 square --/
structure Position :=
(row : Fin 7)
(col : Fin 7)

/-- Defines if a position is in the center or adjacent to the border --/
def is_center_or_border (p : Position) : Prop :=
  (p.row = 3 ∧ p.col = 3) ∨ 
  p.row = 0 ∨ p.row = 6 ∨ p.col = 0 ∨ p.col = 6

/-- Represents the arrangement of tiles in the square --/
def Arrangement := Position → Tile

/-- The theorem to be proved --/
theorem small_tile_position 
  (arr : Arrangement) 
  (h1 : ∃! p, arr p = Tile.Small) 
  (h2 : ∀ p, arr p = Tile.Large → 
       ∃ p1 p2, p1 ≠ p ∧ p2 ≠ p ∧ p1 ≠ p2 ∧ 
       arr p1 = Tile.Large ∧ arr p2 = Tile.Large) 
  (h3 : ∀ p, arr p = Tile.Large ∨ arr p = Tile.Small) :
  ∃ p, arr p = Tile.Small ∧ is_center_or_border p :=
sorry

end NUMINAMATH_CALUDE_small_tile_position_l3664_366434


namespace NUMINAMATH_CALUDE_f_not_monotonic_iff_a_in_range_l3664_366479

def f (a : ℝ) (x : ℝ) : ℝ := x^3 + (1-a)*x^2 - a*(a+2)*x

def is_not_monotonic (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∃ x y z, a < x ∧ x < y ∧ y < z ∧ z < b ∧ 
  ((f x < f y ∧ f y > f z) ∨ (f x > f y ∧ f y < f z))

theorem f_not_monotonic_iff_a_in_range (a : ℝ) :
  is_not_monotonic (f a) (-1) 1 ↔ 
  (a > -5 ∧ a < -1/2) ∨ (a > -1/2 ∧ a < 1) :=
sorry

end NUMINAMATH_CALUDE_f_not_monotonic_iff_a_in_range_l3664_366479


namespace NUMINAMATH_CALUDE_parallelogram_area_l3664_366489

/-- The area of a parallelogram with base 20 cm and height 16 cm is 320 cm². -/
theorem parallelogram_area : 
  ∀ (base height area : ℝ), 
  base = 20 → height = 16 → area = base * height → area = 320 :=
by sorry

end NUMINAMATH_CALUDE_parallelogram_area_l3664_366489


namespace NUMINAMATH_CALUDE_binomial_7_choose_4_l3664_366490

theorem binomial_7_choose_4 : Nat.choose 7 4 = 35 := by
  sorry

end NUMINAMATH_CALUDE_binomial_7_choose_4_l3664_366490


namespace NUMINAMATH_CALUDE_dabbie_turkey_cost_l3664_366406

/-- The cost of Dabbie's turkeys -/
def turkey_cost : ℕ → ℕ
| 0 => 6  -- weight of first turkey
| 1 => 9  -- weight of second turkey
| 2 => 2 * turkey_cost 1  -- weight of third turkey
| _ => 0  -- for completeness

/-- The total weight of all turkeys -/
def total_weight : ℕ := turkey_cost 0 + turkey_cost 1 + turkey_cost 2

/-- The cost per kilogram of turkey -/
def cost_per_kg : ℕ := 2

/-- The theorem stating the total cost of Dabbie's turkeys -/
theorem dabbie_turkey_cost : total_weight * cost_per_kg = 66 := by
  sorry

end NUMINAMATH_CALUDE_dabbie_turkey_cost_l3664_366406


namespace NUMINAMATH_CALUDE_hot_sauce_servings_per_day_l3664_366404

/-- Proves the number of hot sauce servings used per day -/
theorem hot_sauce_servings_per_day 
  (serving_size : Real) 
  (jar_size : Real) 
  (duration : Nat) 
  (h1 : serving_size = 0.5)
  (h2 : jar_size = 32 - 2)
  (h3 : duration = 20) :
  (jar_size / duration) / serving_size = 3 := by
  sorry

end NUMINAMATH_CALUDE_hot_sauce_servings_per_day_l3664_366404


namespace NUMINAMATH_CALUDE_trigonometric_identity_l3664_366462

theorem trigonometric_identity (x y : ℝ) : 
  Real.sin (x - y) * Real.cos y + Real.cos (x - y) * Real.sin y = Real.sin x := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identity_l3664_366462


namespace NUMINAMATH_CALUDE_new_cube_edge_length_l3664_366484

-- Define the edge lengths of the original cubes
def edge1 : ℝ := 6
def edge2 : ℝ := 8
def edge3 : ℝ := 10

-- Define the volume of a cube given its edge length
def cubeVolume (edge : ℝ) : ℝ := edge ^ 3

-- Define the total volume of the three original cubes
def totalVolume : ℝ := cubeVolume edge1 + cubeVolume edge2 + cubeVolume edge3

-- Define the edge length of the new cube
def newEdge : ℝ := totalVolume ^ (1/3)

-- Theorem statement
theorem new_cube_edge_length : newEdge = 12 := by
  sorry

end NUMINAMATH_CALUDE_new_cube_edge_length_l3664_366484


namespace NUMINAMATH_CALUDE_fibonacci_determinant_identity_fibonacci_1002_1004_minus_1003_squared_l3664_366497

def fib : ℕ → ℤ
  | 0 => 0
  | 1 => 1
  | n + 2 => fib (n + 1) + fib n

theorem fibonacci_determinant_identity (n : ℕ) (h : n > 0) :
  fib (n - 1) * fib (n + 1) - fib n ^ 2 = (-1) ^ n := by
  sorry

-- The specific case for n = 1003
theorem fibonacci_1002_1004_minus_1003_squared :
  fib 1002 * fib 1004 - fib 1003 ^ 2 = -1 := by
  sorry

end NUMINAMATH_CALUDE_fibonacci_determinant_identity_fibonacci_1002_1004_minus_1003_squared_l3664_366497


namespace NUMINAMATH_CALUDE_trapezoid_shorter_base_l3664_366491

/-- A trapezoid with the given properties -/
structure Trapezoid where
  longer_base : ℝ
  shorter_base : ℝ
  midpoint_line : ℝ
  longer_base_length : longer_base = 97
  midpoint_line_length : midpoint_line = 3
  midpoint_property : midpoint_line = (longer_base - shorter_base) / 2

theorem trapezoid_shorter_base (t : Trapezoid) : t.shorter_base = 91 := by
  sorry

end NUMINAMATH_CALUDE_trapezoid_shorter_base_l3664_366491


namespace NUMINAMATH_CALUDE_production_calculation_l3664_366409

-- Define the production rate for 6 machines
def production_rate_6 : ℕ := 300

-- Define the number of machines in the original setup
def original_machines : ℕ := 6

-- Define the number of machines in the new setup
def new_machines : ℕ := 10

-- Define the duration in minutes
def duration : ℕ := 4

-- Theorem to prove
theorem production_calculation :
  (new_machines * duration * production_rate_6) / original_machines = 2000 :=
by
  sorry


end NUMINAMATH_CALUDE_production_calculation_l3664_366409


namespace NUMINAMATH_CALUDE_matrix_determinant_zero_l3664_366474

theorem matrix_determinant_zero (a b c : ℝ) : 
  Matrix.det !![1, a+b, b+c; 1, a+2*b, b+2*c; 1, a+3*b, b+3*c] = 0 := by
  sorry

end NUMINAMATH_CALUDE_matrix_determinant_zero_l3664_366474


namespace NUMINAMATH_CALUDE_simplify_expression_l3664_366425

theorem simplify_expression (a b c : ℝ) :
  (15*a + 45*b + 20*c) + (25*a - 35*b - 10*c) - (10*a + 55*b + 30*c) = 30*a - 45*b - 20*c :=
by sorry

end NUMINAMATH_CALUDE_simplify_expression_l3664_366425


namespace NUMINAMATH_CALUDE_recurring_decimal_to_fraction_l3664_366457

theorem recurring_decimal_to_fraction : 
  ∃ (x : ℚ), x = 4 + 56 / 99 ∧ x = 452 / 99 := by sorry

end NUMINAMATH_CALUDE_recurring_decimal_to_fraction_l3664_366457


namespace NUMINAMATH_CALUDE_extended_parallelepiped_volume_2_5_6_l3664_366460

/-- The volume of points within or exactly one unit from a rectangular parallelepiped -/
def extended_parallelepiped_volume (length width height : ℝ) : ℝ :=
  (length + 2) * (width + 2) * (height + 2) - length * width * height

/-- The volume of the set of points within or exactly one unit from a 2x5x6 parallelepiped -/
theorem extended_parallelepiped_volume_2_5_6 :
  extended_parallelepiped_volume 2 5 6 = (1008 + 44 * Real.pi) / 3 := by
  sorry

end NUMINAMATH_CALUDE_extended_parallelepiped_volume_2_5_6_l3664_366460


namespace NUMINAMATH_CALUDE_solve_equation_l3664_366476

theorem solve_equation : ∃ x : ℚ, (3 * x - 4) / 7 = 15 ∧ x = 109 / 3 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l3664_366476


namespace NUMINAMATH_CALUDE_thomas_daniel_equation_l3664_366410

theorem thomas_daniel_equation (b c : ℝ) : 
  (∀ x : ℝ, |x - 4| = 3 ↔ x^2 + b*x + c = 0) → 
  b = -8 ∧ c = 7 := by
sorry

end NUMINAMATH_CALUDE_thomas_daniel_equation_l3664_366410


namespace NUMINAMATH_CALUDE_tim_stored_26_bales_l3664_366487

/-- The number of bales Tim stored in the barn -/
def bales_stored (initial_bales final_bales : ℕ) : ℕ :=
  final_bales - initial_bales

/-- Proof that Tim stored 26 bales in the barn -/
theorem tim_stored_26_bales : bales_stored 28 54 = 26 := by
  sorry

end NUMINAMATH_CALUDE_tim_stored_26_bales_l3664_366487


namespace NUMINAMATH_CALUDE_diagonal_cut_square_dimensions_l3664_366432

/-- Given a square with side length 10 units that is cut diagonally,
    prove that the resulting triangles have dimensions 10, 10, and 10√2 units. -/
theorem diagonal_cut_square_dimensions :
  let square_side : ℝ := 10
  let diagonal : ℝ := square_side * Real.sqrt 2
  ∀ triangle : Set (ℝ × ℝ × ℝ),
    (∃ (a b c : ℝ), triangle = {(a, b, c)} ∧
      a = square_side ∧
      b = square_side ∧
      c = diagonal) →
    triangle = {(10, 10, 10 * Real.sqrt 2)} :=
by sorry

end NUMINAMATH_CALUDE_diagonal_cut_square_dimensions_l3664_366432


namespace NUMINAMATH_CALUDE_not_p_sufficient_not_necessary_for_not_q_l3664_366433

-- Define the conditions p and q
def p (x : ℝ) : Prop := |x + 1| > 2
def q (x : ℝ) : Prop := x > 2

-- Theorem statement
theorem not_p_sufficient_not_necessary_for_not_q :
  (∀ x : ℝ, ¬(p x) → ¬(q x)) ∧ 
  (∃ x : ℝ, ¬(q x) ∧ p x) :=
sorry

end NUMINAMATH_CALUDE_not_p_sufficient_not_necessary_for_not_q_l3664_366433


namespace NUMINAMATH_CALUDE_smaller_cuboid_height_l3664_366419

/-- Given a large cuboid and smaller cuboids with specified dimensions,
    prove that the height of each smaller cuboid is 3 meters. -/
theorem smaller_cuboid_height
  (large_length large_width large_height : ℝ)
  (small_length small_width : ℝ)
  (num_small_cuboids : ℝ)
  (h_large_length : large_length = 18)
  (h_large_width : large_width = 15)
  (h_large_height : large_height = 2)
  (h_small_length : small_length = 6)
  (h_small_width : small_width = 4)
  (h_num_small_cuboids : num_small_cuboids = 7.5)
  (h_volume_conservation : large_length * large_width * large_height =
    num_small_cuboids * small_length * small_width * (large_length * large_width * large_height / (num_small_cuboids * small_length * small_width))) :
  large_length * large_width * large_height / (num_small_cuboids * small_length * small_width) = 3 := by
  sorry

end NUMINAMATH_CALUDE_smaller_cuboid_height_l3664_366419


namespace NUMINAMATH_CALUDE_complex_square_l3664_366426

theorem complex_square (z : ℂ) (h : z * Complex.I = 2 + Complex.I) : z^2 = -3 - 4*Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_square_l3664_366426


namespace NUMINAMATH_CALUDE_science_study_time_l3664_366470

/-- The number of minutes in an hour -/
def minutes_per_hour : ℕ := 60

/-- The total time Sam spends studying in hours -/
def total_study_time_hours : ℕ := 3

/-- The time Sam spends studying Math in minutes -/
def math_study_time : ℕ := 80

/-- The time Sam spends studying Literature in minutes -/
def literature_study_time : ℕ := 40

/-- Theorem: Sam spends 60 minutes studying Science -/
theorem science_study_time : ℕ := by
  sorry

end NUMINAMATH_CALUDE_science_study_time_l3664_366470


namespace NUMINAMATH_CALUDE_sum_of_digits_successor_l3664_366495

/-- Sum of digits function -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- Theorem: If S(n) = 4387, then S(n+1) = 4388 -/
theorem sum_of_digits_successor (n : ℕ) (h : sum_of_digits n = 4387) : 
  sum_of_digits (n + 1) = 4388 := by sorry

end NUMINAMATH_CALUDE_sum_of_digits_successor_l3664_366495


namespace NUMINAMATH_CALUDE_dog_kennel_problem_l3664_366475

theorem dog_kennel_problem (total long_fur brown neither : ℕ) 
  (h_total : total = 45)
  (h_long_fur : long_fur = 26)
  (h_brown : brown = 30)
  (h_neither : neither = 8)
  : long_fur + brown - (total - neither) = 19 := by
  sorry

end NUMINAMATH_CALUDE_dog_kennel_problem_l3664_366475


namespace NUMINAMATH_CALUDE_cubic_intersection_line_l3664_366467

theorem cubic_intersection_line (a b c M : ℝ) : 
  a < b ∧ b < c ∧ 
  2 * (b - a) = c - b ∧
  a^3 - 84*a = M ∧
  b^3 - 84*b = M ∧
  c^3 - 84*c = M →
  M = 160 := by
sorry

end NUMINAMATH_CALUDE_cubic_intersection_line_l3664_366467


namespace NUMINAMATH_CALUDE_min_trig_expression_l3664_366459

theorem min_trig_expression (x : ℝ) : 
  (Real.sin x)^8 + (Real.cos x)^8 + 1 ≥ 7/18 * ((Real.sin x)^6 + (Real.cos x)^6 + 1) := by
  sorry

end NUMINAMATH_CALUDE_min_trig_expression_l3664_366459


namespace NUMINAMATH_CALUDE_cone_volume_proof_l3664_366455

noncomputable def cone_volume (slant_height : ℝ) (lateral_surface_is_semicircle : Prop) : ℝ :=
  (Real.sqrt 3 / 3) * Real.pi

theorem cone_volume_proof (slant_height : ℝ) (lateral_surface_is_semicircle : Prop) 
  (h1 : slant_height = 2)
  (h2 : lateral_surface_is_semicircle) :
  cone_volume slant_height lateral_surface_is_semicircle = (Real.sqrt 3 / 3) * Real.pi :=
by
  sorry

end NUMINAMATH_CALUDE_cone_volume_proof_l3664_366455


namespace NUMINAMATH_CALUDE_altitude_properties_l3664_366438

-- Define the triangle ABC
def A : ℝ × ℝ := (2, -1)
def B : ℝ × ℝ := (3, 2)
def C : ℝ × ℝ := (-3, -1)

-- Define vector BC
def BC : ℝ × ℝ := (C.1 - B.1, C.2 - B.2)

-- Define the altitude AD
def AD : ℝ × ℝ → Prop := λ D => 
  -- AD is perpendicular to BC
  (D.1 - A.1) * BC.1 + (D.2 - A.2) * BC.2 = 0 ∧
  -- D lies on line BC
  ∃ t : ℝ, D = (B.1 + t * BC.1, B.2 + t * BC.2)

-- Theorem statement
theorem altitude_properties : 
  ∃ D : ℝ × ℝ, AD D ∧ 
    ((D.1 - A.1)^2 + (D.2 - A.2)^2 = 5) ∧ 
    D = (1, 1) :=
sorry

end NUMINAMATH_CALUDE_altitude_properties_l3664_366438


namespace NUMINAMATH_CALUDE_function_equation_zero_l3664_366408

theorem function_equation_zero (f : ℝ → ℝ) 
  (h : ∀ x y : ℝ, f x + f y = f (f x * f y)) : 
  ∀ x : ℝ, f x = 0 := by sorry

end NUMINAMATH_CALUDE_function_equation_zero_l3664_366408


namespace NUMINAMATH_CALUDE_apple_harvest_per_section_l3664_366496

theorem apple_harvest_per_section 
  (total_sections : ℕ) 
  (total_sacks : ℕ) 
  (h1 : total_sections = 8) 
  (h2 : total_sacks = 360) : 
  total_sacks / total_sections = 45 := by
  sorry

end NUMINAMATH_CALUDE_apple_harvest_per_section_l3664_366496


namespace NUMINAMATH_CALUDE_degenerate_ellipse_max_y_coordinate_l3664_366494

theorem degenerate_ellipse_max_y_coordinate :
  ∀ x y : ℝ, (x^2 / 49) + ((y - 3)^2 / 25) = 0 → y ≤ 3 :=
by sorry

end NUMINAMATH_CALUDE_degenerate_ellipse_max_y_coordinate_l3664_366494


namespace NUMINAMATH_CALUDE_expression_simplification_and_evaluation_l3664_366440

theorem expression_simplification_and_evaluation (a b : ℝ) 
  (h : |a + 1| + (b - 1/2)^2 = 0) : 
  5 * (a^2 * b - a * b^2) - (a * b^2 + 3 * a^2 * b) = 5/2 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_and_evaluation_l3664_366440


namespace NUMINAMATH_CALUDE_football_count_proof_l3664_366412

/-- The cost of a single soccer ball in dollars -/
def soccer_ball_cost : ℕ := 50

/-- The cost of some footballs and 3 soccer balls in dollars -/
def first_set_cost : ℕ := 220

/-- The cost of 3 footballs and 1 soccer ball in dollars -/
def second_set_cost : ℕ := 155

/-- The number of footballs in the second set -/
def footballs_in_second_set : ℕ := 3

theorem football_count_proof : 
  ∃ (football_cost : ℕ) (footballs_in_first_set : ℕ),
    footballs_in_first_set * football_cost + 3 * soccer_ball_cost = first_set_cost ∧
    3 * football_cost + soccer_ball_cost = second_set_cost ∧
    footballs_in_second_set = 3 :=
sorry

end NUMINAMATH_CALUDE_football_count_proof_l3664_366412


namespace NUMINAMATH_CALUDE_overlap_length_l3664_366441

theorem overlap_length (L D : ℝ) (n : ℕ) (h1 : L = 98) (h2 : D = 83) (h3 : n = 6) :
  ∃ x : ℝ, x = (L - D) / n ∧ x = 2.5 := by
  sorry

end NUMINAMATH_CALUDE_overlap_length_l3664_366441


namespace NUMINAMATH_CALUDE_max_gcd_sum_l3664_366480

theorem max_gcd_sum (a b c : ℕ) (h1 : 1 ≤ a) (h2 : a < b) (h3 : b < c) (h4 : c ≤ 3000) :
  (∃ (x y z : ℕ), 1 ≤ x ∧ x < y ∧ y < z ∧ z ≤ 3000 ∧
    Nat.gcd x y + Nat.gcd y z + Nat.gcd z x = 3000) ∧
  (∀ (x y z : ℕ), 1 ≤ x → x < y → y < z → z ≤ 3000 →
    Nat.gcd x y + Nat.gcd y z + Nat.gcd z x ≤ 3000) :=
by
  sorry

end NUMINAMATH_CALUDE_max_gcd_sum_l3664_366480


namespace NUMINAMATH_CALUDE_quadrilateral_area_l3664_366422

/-- The area of a quadrilateral with given diagonal and offsets -/
theorem quadrilateral_area (d h₁ h₂ : ℝ) (hd : d = 40) (hh₁ : h₁ = 9) (hh₂ : h₂ = 6) :
  (1 / 2 : ℝ) * d * h₁ + (1 / 2 : ℝ) * d * h₂ = 300 := by
  sorry

end NUMINAMATH_CALUDE_quadrilateral_area_l3664_366422


namespace NUMINAMATH_CALUDE_solution_verification_l3664_366448

theorem solution_verification :
  let x : ℚ := 425
  let y : ℝ := (270 + 90 * Real.sqrt 2) / 7
  (x - (11/17) * x = 150) ∧ (y - ((Real.sqrt 2)/3) * y = 90) := by sorry

end NUMINAMATH_CALUDE_solution_verification_l3664_366448


namespace NUMINAMATH_CALUDE_smallest_k_square_root_diff_l3664_366458

/-- Represents a card with a number from 1 to 2016 -/
def Card := {n : ℕ // 1 ≤ n ∧ n ≤ 2016}

/-- The property that two cards have numbers whose square roots differ by less than 1 -/
def SquareRootDiffLessThanOne (a b : Card) : Prop :=
  |Real.sqrt a.val - Real.sqrt b.val| < 1

/-- The theorem stating that 45 is the smallest number of cards guaranteeing
    two cards with square root difference less than 1 -/
theorem smallest_k_square_root_diff : 
  (∀ (S : Finset Card), S.card = 45 → 
    ∃ (a b : Card), a ∈ S ∧ b ∈ S ∧ a ≠ b ∧ SquareRootDiffLessThanOne a b) ∧
  (∀ (k : ℕ), k < 45 → 
    ∃ (S : Finset Card), S.card = k ∧
      ∀ (a b : Card), a ∈ S → b ∈ S → a ≠ b → ¬SquareRootDiffLessThanOne a b) :=
by sorry

end NUMINAMATH_CALUDE_smallest_k_square_root_diff_l3664_366458


namespace NUMINAMATH_CALUDE_plane_line_perpendicular_l3664_366488

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (parallel : Plane → Plane → Prop)
variable (perpendicular : Line → Plane → Prop)

-- State the theorem
theorem plane_line_perpendicular 
  (m : Line) (α β γ : Plane) :
  parallel α β → parallel β γ → perpendicular m α → perpendicular m γ :=
sorry

end NUMINAMATH_CALUDE_plane_line_perpendicular_l3664_366488


namespace NUMINAMATH_CALUDE_koala_fiber_intake_l3664_366418

/-- Proves that if a koala absorbs 20% of the fiber it eats and it absorbed 8 ounces of fiber in one day, then the total amount of fiber the koala ate that day was 40 ounces. -/
theorem koala_fiber_intake (absorption_rate : Real) (absorbed_amount : Real) (total_intake : Real) :
  absorption_rate = 0.20 →
  absorbed_amount = 8 →
  absorbed_amount = absorption_rate * total_intake →
  total_intake = 40 := by
  sorry

end NUMINAMATH_CALUDE_koala_fiber_intake_l3664_366418


namespace NUMINAMATH_CALUDE_square_equals_1369_l3664_366439

theorem square_equals_1369 (x : ℤ) (h : x^2 = 1369) : (x + 1) * (x - 1) = 1368 := by
  sorry

end NUMINAMATH_CALUDE_square_equals_1369_l3664_366439


namespace NUMINAMATH_CALUDE_largest_consecutive_sum_30_l3664_366446

def consecutive_sum (start : ℕ) (count : ℕ) : ℕ :=
  count * start + count * (count - 1) / 2

theorem largest_consecutive_sum_30 :
  (∃ (n : ℕ), n > 0 ∧ ∃ (start : ℕ), start > 0 ∧ consecutive_sum start n = 30) ∧
  (∀ (m : ℕ), m > 5 → ¬∃ (start : ℕ), start > 0 ∧ consecutive_sum start m = 30) :=
sorry

end NUMINAMATH_CALUDE_largest_consecutive_sum_30_l3664_366446


namespace NUMINAMATH_CALUDE_car_catching_truck_l3664_366481

/-- A problem about a car catching up to a truck on a highway. -/
theorem car_catching_truck (truck_speed : ℝ) (head_start : ℝ) (catch_up_time : ℝ) :
  truck_speed = 45 →
  head_start = 1 →
  catch_up_time = 4 →
  let car_speed := (truck_speed * (catch_up_time + head_start)) / catch_up_time
  car_speed = 56.25 := by
sorry


end NUMINAMATH_CALUDE_car_catching_truck_l3664_366481


namespace NUMINAMATH_CALUDE_combined_friends_list_l3664_366420

theorem combined_friends_list (james_friends : ℕ) (susan_friends : ℕ) (maria_friends : ℕ)
  (james_john_shared : ℕ) (james_john_maria_shared : ℕ)
  (h1 : james_friends = 90)
  (h2 : susan_friends = 50)
  (h3 : maria_friends = 80)
  (h4 : james_john_shared = 35)
  (h5 : james_john_maria_shared = 10) :
  james_friends + 4 * susan_friends - james_john_shared + maria_friends - james_john_maria_shared = 325 := by
  sorry

end NUMINAMATH_CALUDE_combined_friends_list_l3664_366420


namespace NUMINAMATH_CALUDE_condition_type_1_condition_type_2_condition_type_3_l3664_366477

-- Statement 1
theorem condition_type_1 :
  (∀ x : ℝ, 0 < x ∧ x < 3 → |x - 1| < 2) ∧
  ¬(∀ x : ℝ, |x - 1| < 2 → 0 < x ∧ x < 3) := by sorry

-- Statement 2
theorem condition_type_2 :
  (∀ x : ℝ, x = 2 → (x - 2) * (x - 3) = 0) ∧
  ¬(∀ x : ℝ, (x - 2) * (x - 3) = 0 → x = 2) := by sorry

-- Statement 3
theorem condition_type_3 :
  ∀ (a b c : ℝ), c = 0 ↔ a * 0^2 + b * 0 + c = 0 := by sorry

end NUMINAMATH_CALUDE_condition_type_1_condition_type_2_condition_type_3_l3664_366477


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3664_366449

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The statement of the problem -/
theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  arithmetic_sequence a →
  (a 3)^2 + 3 * (a 3) - 18 = 0 →
  (a 8)^2 + 3 * (a 8) - 18 = 0 →
  a 5 + a 6 = 3 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3664_366449


namespace NUMINAMATH_CALUDE_expansion_properties_l3664_366463

/-- The expansion of (x^(1/4) + x^(3/2))^n where the third-to-last term's coefficient is 45 -/
def expansion (x : ℝ) (n : ℕ) := (x^(1/4) + x^(3/2))^n

/-- The coefficient of the third-to-last term in the expansion -/
def third_to_last_coeff (n : ℕ) := Nat.choose n (n - 2)

theorem expansion_properties (x : ℝ) (n : ℕ) 
  (h : third_to_last_coeff n = 45) : 
  ∃ (k : ℕ), 
    (Nat.choose n k * x^5 = 45 * x^5) ∧ 
    (∀ (j : ℕ), j ≤ n → Nat.choose n j ≤ 252) ∧
    (Nat.choose n 5 * x^(35/4) = 252 * x^(35/4)) := by
  sorry

end NUMINAMATH_CALUDE_expansion_properties_l3664_366463


namespace NUMINAMATH_CALUDE_fuel_station_problem_l3664_366482

/-- Represents the number of trucks filled up at a fuel station. -/
def num_trucks : ℕ := 2

theorem fuel_station_problem :
  let service_cost : ℚ := 21/10
  let fuel_cost_per_liter : ℚ := 7/10
  let num_minivans : ℕ := 3
  let total_cost : ℚ := 3472/10
  let minivan_capacity : ℚ := 65
  let truck_capacity : ℚ := minivan_capacity * 220/100
  
  let minivan_fuel_cost : ℚ := num_minivans * minivan_capacity * fuel_cost_per_liter
  let minivan_service_cost : ℚ := num_minivans * service_cost
  let total_minivan_cost : ℚ := minivan_fuel_cost + minivan_service_cost
  
  let truck_cost : ℚ := total_cost - total_minivan_cost
  let single_truck_fuel_cost : ℚ := truck_capacity * fuel_cost_per_liter
  let single_truck_total_cost : ℚ := single_truck_fuel_cost + service_cost
  
  num_trucks = (truck_cost / single_truck_total_cost).num :=
by sorry

#check fuel_station_problem

end NUMINAMATH_CALUDE_fuel_station_problem_l3664_366482


namespace NUMINAMATH_CALUDE_max_value_of_m_over_n_l3664_366437

theorem max_value_of_m_over_n (n : ℝ) (m : ℝ) (h_n : n > 0) :
  (∀ x > 0, Real.log x + 1 ≥ m - n / x) →
  m / n ≤ Real.exp 1 :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_m_over_n_l3664_366437


namespace NUMINAMATH_CALUDE_hikers_distribution_theorem_l3664_366471

/-- The number of ways to distribute 5 people into three rooms -/
def distribution_ways : ℕ := 20

/-- The number of hikers -/
def num_hikers : ℕ := 5

/-- The number of available rooms -/
def num_rooms : ℕ := 3

/-- The capacity of the largest room -/
def large_room_capacity : ℕ := 3

/-- The capacity of each of the smaller rooms -/
def small_room_capacity : ℕ := 2

/-- Theorem stating that the number of ways to distribute the hikers is correct -/
theorem hikers_distribution_theorem :
  distribution_ways = (num_hikers.choose large_room_capacity) * 2 :=
by sorry

end NUMINAMATH_CALUDE_hikers_distribution_theorem_l3664_366471


namespace NUMINAMATH_CALUDE_f_cos_x_equals_two_plus_cos_two_x_l3664_366478

-- Define the function f
def f : ℝ → ℝ := sorry

-- State the theorem
theorem f_cos_x_equals_two_plus_cos_two_x (x : ℝ) : 
  (∀ y : ℝ, f (Real.sin y) = 2 - Real.cos (2 * y)) → 
  f (Real.cos x) = 2 + Real.cos (2 * x) := by
  sorry

end NUMINAMATH_CALUDE_f_cos_x_equals_two_plus_cos_two_x_l3664_366478


namespace NUMINAMATH_CALUDE_subtraction_result_l3664_366413

def largest_3digit_number : ℕ := 999
def smallest_5digit_number : ℕ := 10000

theorem subtraction_result : 
  smallest_5digit_number - largest_3digit_number = 9001 := by sorry

end NUMINAMATH_CALUDE_subtraction_result_l3664_366413


namespace NUMINAMATH_CALUDE_orchard_harvest_l3664_366466

/-- Calculates the total mass of fruit harvested in an orchard -/
def total_fruit_mass (apple_trees : ℕ) (apple_yield : ℕ) (peach_trees : ℕ) (peach_yield : ℕ) : ℕ :=
  apple_trees * apple_yield + peach_trees * peach_yield

/-- Theorem stating the total mass of fruit harvested in the specific orchard -/
theorem orchard_harvest :
  total_fruit_mass 30 150 45 65 = 7425 := by
  sorry

end NUMINAMATH_CALUDE_orchard_harvest_l3664_366466


namespace NUMINAMATH_CALUDE_arithmetic_sequence_50th_term_l3664_366464

theorem arithmetic_sequence_50th_term : 
  let start : ℤ := -48
  let diff : ℤ := 2
  let n : ℕ := 50
  let sequence := fun i : ℕ => start + diff * (i - 1)
  sequence n = 50 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_50th_term_l3664_366464


namespace NUMINAMATH_CALUDE_integer_solution_fifth_power_minus_three_times_square_l3664_366469

theorem integer_solution_fifth_power_minus_three_times_square : ∃ x : ℤ, x^5 - 3*x^2 = 216 ∧ x = 3 := by
  sorry

end NUMINAMATH_CALUDE_integer_solution_fifth_power_minus_three_times_square_l3664_366469


namespace NUMINAMATH_CALUDE_collect_all_blocks_time_l3664_366403

/-- Represents the block collection problem --/
structure BlockCollection where
  totalBlocks : ℕ := 50
  dadPuts : ℕ := 5
  miaRemoves : ℕ := 3
  brotherRemoves : ℕ := 1
  cycleTime : ℕ := 30  -- in seconds

/-- Calculates the time in minutes to collect all blocks --/
def timeToCollectAll (bc : BlockCollection) : ℕ :=
  let netBlocksPerCycle := bc.dadPuts - (bc.miaRemoves + bc.brotherRemoves)
  let cyclesToReachAlmostAll := (bc.totalBlocks - bc.dadPuts) / netBlocksPerCycle
  let totalSeconds := (cyclesToReachAlmostAll + 1) * bc.cycleTime
  totalSeconds / 60

/-- Theorem stating that the time to collect all blocks is 23 minutes --/
theorem collect_all_blocks_time (bc : BlockCollection) :
  timeToCollectAll bc = 23 := by
  sorry

end NUMINAMATH_CALUDE_collect_all_blocks_time_l3664_366403


namespace NUMINAMATH_CALUDE_toll_constant_is_half_dollar_l3664_366431

/-- The number of axles on an 18-wheel truck with 2 wheels on its front axle and 4 wheels on each other axle -/
def truck_axles : ℕ := 5

/-- The toll formula for a truck -/
def toll (constant : ℝ) (x : ℕ) : ℝ := 2.50 + constant * (x - 2)

/-- The theorem stating that the constant in the toll formula is 0.50 -/
theorem toll_constant_is_half_dollar :
  ∃ (constant : ℝ), toll constant truck_axles = 4 ∧ constant = 0.50 := by
  sorry

end NUMINAMATH_CALUDE_toll_constant_is_half_dollar_l3664_366431


namespace NUMINAMATH_CALUDE_number_order_l3664_366499

theorem number_order : 
  let a : ℝ := 30.5
  let b : ℝ := 0.53
  let c : ℝ := Real.log 0.53
  c < b ∧ b < a := by sorry

end NUMINAMATH_CALUDE_number_order_l3664_366499


namespace NUMINAMATH_CALUDE_smallest_three_digit_congruence_l3664_366442

theorem smallest_three_digit_congruence :
  ∃ n : ℕ, 
    (100 ≤ n ∧ n < 1000) ∧ 
    (60 * n ≡ 180 [MOD 300]) ∧ 
    (∀ m : ℕ, (100 ≤ m ∧ m < 1000) ∧ (60 * m ≡ 180 [MOD 300]) → n ≤ m) ∧
    n = 103 := by
  sorry

end NUMINAMATH_CALUDE_smallest_three_digit_congruence_l3664_366442


namespace NUMINAMATH_CALUDE_linear_equation_solution_l3664_366445

theorem linear_equation_solution : ∀ (x y : ℝ), x = 3 ∧ y = -2 → 2 * x + 3 * y = 0 := by
  sorry

end NUMINAMATH_CALUDE_linear_equation_solution_l3664_366445


namespace NUMINAMATH_CALUDE_product_odd_probability_l3664_366400

def range_start : ℕ := 5
def range_end : ℕ := 19

def is_in_range (n : ℕ) : Prop := range_start ≤ n ∧ n ≤ range_end

def total_integers : ℕ := range_end - range_start + 1

def odd_integers : ℕ := (total_integers + 1) / 2

theorem product_odd_probability :
  (odd_integers.choose 2 : ℚ) / (total_integers.choose 2) = 4 / 15 :=
sorry

end NUMINAMATH_CALUDE_product_odd_probability_l3664_366400


namespace NUMINAMATH_CALUDE_smallest_next_divisor_after_427_l3664_366427

theorem smallest_next_divisor_after_427 (m : ℕ) (h1 : 1000 ≤ m ∧ m ≤ 9999) 
  (h2 : m % 2 = 0) (h3 : m % 427 = 0) :
  ∃ (d : ℕ), d > 427 ∧ m % d = 0 ∧ d = 434 ∧ 
  ∀ (x : ℕ), 427 < x ∧ x < 434 → m % x ≠ 0 := by
sorry

end NUMINAMATH_CALUDE_smallest_next_divisor_after_427_l3664_366427


namespace NUMINAMATH_CALUDE_complement_determines_set_l3664_366421

def U : Set Nat := {1, 2, 3, 4}

theorem complement_determines_set (B : Set Nat) (h : Set.compl B = {2, 3}) : B = {1, 4} := by
  sorry

end NUMINAMATH_CALUDE_complement_determines_set_l3664_366421


namespace NUMINAMATH_CALUDE_max_value_is_72_l3664_366465

/-- Represents a type of rock with its weight and value -/
structure Rock where
  weight : ℕ
  value : ℕ

/-- The maximum weight Carl can carry -/
def maxWeight : ℕ := 24

/-- The available types of rocks -/
def rocks : List Rock := [
  { weight := 6, value := 18 },
  { weight := 3, value := 9 },
  { weight := 2, value := 5 }
]

/-- A function to calculate the maximum value of rocks that can be carried -/
def maxValue (rocks : List Rock) (maxWeight : ℕ) : ℕ :=
  sorry

/-- Theorem stating that the maximum value Carl can transport is $72 -/
theorem max_value_is_72 : maxValue rocks maxWeight = 72 := by
  sorry

end NUMINAMATH_CALUDE_max_value_is_72_l3664_366465


namespace NUMINAMATH_CALUDE_max_distance_point_to_line_l3664_366430

/-- The maximum distance from a point to a line --/
theorem max_distance_point_to_line : 
  let P : ℝ × ℝ := (-1, 3)
  let line_equation (k x : ℝ) := k * (x - 2)
  ∀ k : ℝ, 
  (∃ x : ℝ, abs (P.2 - line_equation k P.1) / Real.sqrt (k^2 + 1) ≤ 3 * Real.sqrt 2) ∧ 
  (∃ k₀ : ℝ, abs (P.2 - line_equation k₀ P.1) / Real.sqrt (k₀^2 + 1) = 3 * Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_max_distance_point_to_line_l3664_366430
